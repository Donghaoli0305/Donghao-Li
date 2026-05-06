#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lora_evaluation.py — Evaluate a base Qwen + LoRA adapter model
on PubMedQA-style HF datasets using constrained first-token scoring.

This is the LoRA version of eval_full_ft.py, i.e.:
- data format: load_from_disk(...) with 'messages' and 'label'
- scoring: only score the first generated token among [" yes", " no", " maybe"]
- support calibration (grid search) or direct bias+margin eval
- difference: model is (base) + (LoRA adapter), not a single finetuned directory
"""

import argparse
import contextlib
import os
from typing import Dict, List, Tuple

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ============================================================
# Argparse
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser("Evaluate a LoRA-augmented Qwen model")

    # data / model
    ap.add_argument("--data", required=True, help="HF dataset dir, e.g. processed_data/pqal/fold0/hf_dataset")
    ap.add_argument("--base-model", required=True, help="Base Qwen path, e.g. model/Qwen2.5-7B")
    ap.add_argument("--adapter", required=True, help="LoRA adapter dir, e.g. outputs/stage2/fold0/final")
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])

    # inference
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--answer-tag", type=str, default="Answer:")

    # direct bias/margin
    ap.add_argument("--logit-bias", type=str, default="yes=0,no=0,maybe=0")
    ap.add_argument("--no-margin", type=float, default=0.0)

    # calibration
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--grid-yes", type=str, default="0,0.1,0.2")
    ap.add_argument("--grid-no", type=str, default="0.5,1.0,1.5,2.0")
    ap.add_argument("--grid-maybe", type=str, default="-0.4,-0.2,0,0.2")
    ap.add_argument("--grid-margin", type=str, default="0.3,0.5,0.7")
    ap.add_argument("--calib-metric", choices=["macro_f1", "accuracy", "f1_no"], default="accuracy")

    # misc
    ap.add_argument("--debug", action="store_true")
    return ap.parse_args()


# ============================================================
# Helpers (copied/adapted from eval_full_ft.py)
# ============================================================

def parse_kv_bias(s: str, keys=("yes", "no", "maybe"), default=0.0) -> Dict[str, float]:
    d = {k: default for k in keys}
    if not s:
        return d
    for kv in s.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            k = k.strip().lower()
            try:
                d[k] = float(v.strip())
            except Exception:
                pass
    return d


def parse_list(s: str) -> List[float]:
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]


def safe_div(a, b):
    return (a / b) if b else 0.0


def bf16_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def compute_metrics(golds: List[str], preds: List[str]):
    labels = ["yes", "no", "maybe"]
    idx = {lab: i for i, lab in enumerate(labels)}

    cm = [[0] * 3 for _ in range(3)]
    tp = {lab: 0 for lab in labels}
    fp = {lab: 0 for lab in labels}
    fn = {lab: 0 for lab in labels}

    used = 0
    for y_true, y_pred in zip(golds, preds):
        if y_true not in labels:
            continue
        used += 1
        cm[idx[y_true]][idx[y_pred]] += 1
        if y_true == y_pred:
            tp[y_true] += 1
        else:
            fp[y_pred] += 1
            fn[y_true] += 1

    prec = {lab: safe_div(tp[lab], tp[lab] + fp[lab]) for lab in labels}
    rec = {lab: safe_div(tp[lab], tp[lab] + fn[lab]) for lab in labels}
    f1 = {lab: safe_div(2 * prec[lab] * rec[lab], prec[lab] + rec[lab]) for lab in labels}

    acc = safe_div(sum(tp.values()), sum(sum(r) for r in cm))
    macro_f1 = sum(f1.values()) / 3.0

    stats = {
        "used": used,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "p_yes": prec["yes"], "r_yes": rec["yes"], "f1_yes": f1["yes"],
        "p_no": prec["no"], "r_no": rec["no"], "f1_no": f1["no"],
        "p_maybe": prec["maybe"], "r_maybe": rec["maybe"], "f1_maybe": f1["maybe"],
    }

    return stats, cm


@torch.no_grad()
def precompute_first_token_logprobs(
    model,
    tok,
    prompts: List[str],
    answer_tag: str,
    target_dtype: torch.dtype,
    bsz: int = 16,
    max_len: int = 1024,
):
    device = next(model.parameters()).device

    tok_ids = {
        "yes": tok("yes", add_special_tokens=False)["input_ids"][0],
        "no": tok("no", add_special_tokens=False)["input_ids"][0],
        "maybe": tok("maybe", add_special_tokens=False)["input_ids"][0],
    }

    scores = torch.empty((len(prompts), 3), dtype=torch.float32)
    i0 = 0

    ctx = (
        torch.amp.autocast("cuda", dtype=target_dtype)
        if torch.cuda.is_available() and target_dtype in (torch.float16, torch.bfloat16)
        else contextlib.nullcontext()
    )

    with ctx:
        for batch_idx in batched(list(range(len(prompts))), bsz):
            batch_prompts = []
            for i in batch_idx:
                p = prompts[i]
                if answer_tag:
                    if not p.rstrip().endswith(answer_tag):
                        p = p.rstrip() + "\n" + answer_tag + " "
                batch_prompts.append(p)

            enc = tok(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            out = model(**enc)
            last_logits = out.logits[:, -1, :]
            logp = torch.log_softmax(last_logits, dim=-1)

            yes_logp = logp[:, tok_ids["yes"]]
            no_logp = logp[:, tok_ids["no"]]
            may_logp = logp[:, tok_ids["maybe"]]

            S = torch.stack([yes_logp, no_logp, may_logp], dim=1).to(torch.float32)
            scores[i0 : i0 + len(batch_idx), :] = S
            i0 += len(batch_idx)

    return scores


def predict_from_scores(
    scores: torch.Tensor,
    logit_bias: Dict[str, float],
    no_margin: float,
) -> List[str]:
    bias_vec = torch.tensor(
        [logit_bias.get("yes", 0.0), logit_bias.get("no", 0.0), logit_bias.get("maybe", 0.0)],
        dtype=torch.float32,
        device=scores.device,
    )
    S = scores + bias_vec
    best_vals, best_idx = S.max(dim=1)

    if no_margin > 0:
        no_scores = S[:, 1]
        pick_no = no_scores >= (best_vals - no_margin)
        best_idx[pick_no] = 1

    idx2lab = {0: "yes", 1: "no", 2: "maybe"}
    return [idx2lab[int(i)] for i in best_idx.tolist()]


# ============================================================
# main
# ============================================================

def main():
    args = parse_args()

    print("=" * 60)
    print("LoRA Evaluation")
    print("=" * 60)
    print(f"data:       {args.data}")
    print(f"base model: {args.base-model if hasattr(args, 'base-model') else args.base_model}")
    print(f"adapter:    {args.adapter}")
    print(f"split:      {args.split}")
    print("=" * 60)

    # 1) load dataset
    ds = load_from_disk(args.data)
    if args.split not in ds:
        raise ValueError(f"split '{args.split}' not found in dataset")
    dset = ds[args.split]
    print(f"✓ loaded {len(dset)} samples from {args.split}")

        # 2) tokenizer
    tok = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        trust_remote_code=True,
    )
    tok.padding_side = "left"
    if tok.pad_token is None and getattr(tok, "eos_token", None):
        tok.pad_token = tok.eos_token

    # 3) load base + adapter
    target_dtype = torch.bfloat16 if bf16_available() else torch.float16

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=target_dtype,
        trust_remote_code=True,
    )
    base.config.use_cache = True

    # ✅ 关键补丁：跟训练时一样，把 base 也 resize 到 tokenizer 的长度
    try:
        base.resize_token_embeddings(len(tok))
    except Exception as e:
        print("[WARN] resize_token_embeddings during eval failed:", e)

    # 再加载 LoRA
    model = PeftModel.from_pretrained(
        base,
        args.adapter,
        is_trainable=False,
    )
    model.eval()


    device = next(model.parameters()).device
    print(f"✓ model ready on {device}, dtype={target_dtype}")

    # 4) build prompts + golds
    prompts = []
    golds = []
    uses_messages = "messages" in dset.column_names
    for ex in dset:
        golds.append(str(ex.get("label", "")).lower().strip())
        if not uses_messages:
            raise ValueError("dataset must have 'messages' column")
        msgs = ex["messages"]
        user_side = msgs[:-1]
        p = tok.apply_chat_template(
            user_side,
            tokenize=False,
            add_generation_prompt=True,
        )
        if args.answer_tag:
            p = p.rstrip() + "\n" + args.answer_tag + " "
        prompts.append(p)

    # 5) precompute scores
    scores = precompute_first_token_logprobs(
        model=model,
        tok=tok,
        prompts=prompts,
        answer_tag=args.answer_tag,
        target_dtype=target_dtype,
        bsz=args.bsz,
        max_len=args.max_len,
    )
    print(f"✓ scores shape: {scores.shape}")

    # 6) calibrate or direct eval
    if args.calibrate:
        grid_yes = parse_list(args.grid_yes)
        grid_no = parse_list(args.grid_no)
        grid_maybe = parse_list(args.grid_maybe)
        grid_margin = parse_list(args.grid_margin)

        best = None
        total = len(grid_yes) * len(grid_no) * len(grid_maybe) * len(grid_margin)
        it = 0

        for yb in grid_yes:
            for nb in grid_no:
                for mb in grid_maybe:
                    for mg in grid_margin:
                        it += 1
                        lb = {"yes": yb, "no": nb, "maybe": mb}
                        preds = predict_from_scores(scores, lb, mg)
                        stats, cm = compute_metrics(golds, preds)

                        if args.calib_metric == "accuracy":
                            metric = stats["accuracy"]
                        elif args.calib_metric == "f1_no":
                            metric = stats["f1_no"]
                        else:
                            metric = stats["macro_f1"]

                        if (best is None) or (metric > best[0]):
                            best = (metric, (yb, nb, mb, mg), stats, cm)
                            print(f"[CALIB] {it}/{total} NEW best={metric:.4f} "
                                  f"(yes={yb}, no={nb}, maybe={mb}, margin={mg})")

        best_metric, (best_y, best_n, best_m, best_mg), best_stats, best_cm = best
        print("\n=== CALIB RESULT ===")
        print(f"best {args.calib_metric}: {best_metric:.4f}")
        print(f'--logit-bias "yes={best_y},no={best_n},maybe={best_m}" --no-margin {best_mg}')
    else:
        log_bias = parse_kv_bias(args.logit_bias)
        preds = predict_from_scores(scores, log_bias, args.no_margin)
        stats, cm = compute_metrics(golds, preds)

        print("\n=== EVAL RESULT ===")
        print(f"N={len(golds)}  split={args.split}")
        print(f"accuracy:  {stats['accuracy']:.4f}")
        print(f"macro_f1:  {stats['macro_f1']:.4f}")
        print("per-class F1:")
        print(f"  yes   {stats['f1_yes']:.4f}")
        print(f"  no    {stats['f1_no']:.4f}")
        print(f"  maybe {stats['f1_maybe']:.4f}")
        print("confusion matrix (rows=true, cols=pred):")
        labs = ["yes", "no", "maybe"]
        print("         yes   no   maybe")
        for i, r in enumerate(cm):
            print(f"{labs[i]:6s} {r[0]:4d} {r[1]:4d} {r[2]:6d}")


if __name__ == "__main__":
    main()