#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import argparse
from dataclasses import dataclass
from typing import Any, List, Optional, Dict
import traceback

import torch
from datasets import load_from_disk, DatasetDict
from torch.utils.data import DataLoader, WeightedRandomSampler

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import set_seed
from transformers.trainer_callback import EarlyStoppingCallback

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
)

# LoRA 要挂的模块，可以按需缩减
TARGET_MODULES = [
    "q_proj",  "v_proj",
]


def parse_args():
    ap = argparse.ArgumentParser()

    # 必需参数
    ap.add_argument("--data", required=True, help="load_from_disk 的目录，必须包含 train / validation")
    ap.add_argument("--model", required=True, help="基础模型路径或 HF 名")
    ap.add_argument("--out", required=True, help="输出目录")

    # 训练超参
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--eval-bsz", type=int, default=8)
    ap.add_argument("--grad-acc", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--logging-steps", type=int, default=100)

    # dataloader & encode 并行
    ap.add_argument("--num-workers", type=int, default=1, help="PyTorch DataLoader workers")
    ap.add_argument("--encode-proc", type=int, default=2, help="datasets.map 并行进程数")

    # checkpoint / adapter
    ap.add_argument("--resume", type=str, default=None, help="resume from checkpoint")
    ap.add_argument("--init-adapter", type=str, default=None, help="从已有 LoRA adapter 继续训练")

    # 训练行为
    ap.add_argument("--oversample", action="store_true", help="对 label 做加权采样")
    ap.add_argument("--gc", action="store_true", help="开启 gradient checkpointing")
    ap.add_argument("--selfcheck", action="store_true", help="打印参数调试用")

    # early stopping
    ap.add_argument("--early-stop", action="store_true")
    ap.add_argument("--early-metric", type=str, default="eval_loss")
    ap.add_argument("--early-patience", type=int, default=2)
    ap.add_argument("--early-mode", type=str, choices=["min", "max"], default="min")

    return ap.parse_args()


def bf16_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


@dataclass
class AnswerOnlyCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features):
        # padding 到同一长度
        batch = self.tokenizer.pad(
            {
                "input_ids": [f["input_ids"] for f in features],
                "attention_mask": [f["attention_mask"] for f in features],
            },
            padding=True,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].shape[1]

        padded_labels = []
        for f in features:
            lab = f["labels"]
            if len(lab) < max_len:
                lab = lab + [-100] * (max_len - len(lab))
            else:
                lab = lab[:max_len]
            padded_labels.append(lab)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


class OversampleTrainer(Trainer):
    """Trainer 带 weighted sampler"""

    def __init__(self, *args, weights_for_train: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._weights_for_train = weights_for_train

    def get_train_dataloader(self) -> DataLoader:
        if self._weights_for_train is None:
            return super().get_train_dataloader()
        sampler = WeightedRandomSampler(
            weights=self._weights_for_train.double(),
            num_samples=len(self._weights_for_train),
            replacement=True,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def compute_inverse_freq_weights(labels: List[Any]) -> torch.Tensor:
    """label 不平衡时做 1/freq 加权"""
    from collections import Counter
    labs = [str(l).lower().strip() for l in labels]
    uniq = sorted(set(labs))
    lab2idx = {lab: i for i, lab in enumerate(uniq)}
    idxs = [lab2idx[l] for l in labs]
    cnt = Counter(idxs)
    n = len(idxs)
    K = max(1, len(cnt))
    class_w = {c: n / (K * cnt[c]) for c in cnt}
    return torch.tensor([class_w[i] for i in idxs], dtype=torch.float)


def main():
    args = parse_args()
    set_seed(42)

    if args.selfcheck:
        print("[SELF-CHECK]", args)

    # ==========================================================
    # 1. 加载数据
    # ==========================================================
    ds: DatasetDict = load_from_disk(args.data)
    if "train" not in ds or "validation" not in ds:
        raise ValueError("dataset must contain train and validation splits")

    # ==========================================================
    # 2. tokenizer
    # ==========================================================
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    tok.padding_side = "right"
    if tok.pad_token is None and getattr(tok, "eos_token", None):
        tok.pad_token = tok.eos_token

    # ==========================================================
    # 3. 编码函数（加速版）
    # ==========================================================
    def encode_answer_only(batch: Dict[str, Any]) -> Dict[str, Any]:
        prompts = []
        answers = []
        for msgs, lab in zip(batch["messages"], batch["label"]):
            # 去掉最后一条 assistant，保留对话上下文
            msgs_no_assist = msgs[:-1]
            prompt = tok.apply_chat_template(
                msgs_no_assist,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)
            answers.append(" " + str(lab).lower())

        # 一批编码 prompt
        enc_p = tok(
            prompts,
            padding=False,
            truncation=True,
            max_length=args.max_len,
        )
        # 一批编码答案
        enc_a = tok(
            answers,
            padding=False,
            truncation=True,
            max_length=8,
        )

        input_ids_list = []
        attn_list = []
        labels_list = []

        for p_ids, p_attn, a_ids in zip(
            enc_p["input_ids"], enc_p["attention_mask"], enc_a["input_ids"]
        ):
            # 去掉答案开头的 BOS
            if len(a_ids) > 0 and tok.bos_token_id is not None and a_ids[0] == tok.bos_token_id:
                a_ids = a_ids[1:]

            full_ids = p_ids + a_ids
            full_attn = p_attn + [1] * len(a_ids)
            lab = [-100] * len(p_ids) + a_ids

            # 截断
            if len(full_ids) > args.max_len:
                over = len(full_ids) - args.max_len
                keep_p = max(0, len(p_ids) - over)
                full_ids = full_ids[:keep_p] + a_ids
                full_attn = full_attn[:keep_p] + [1] * len(a_ids)
                lab = [-100] * keep_p + a_ids

            input_ids_list.append(full_ids)
            attn_list.append(full_attn)
            labels_list.append(lab)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attn_list,
            "labels": labels_list,
        }

    # ==========================================================
    # 4. 对两个 split 都做 map，并行 num_proc
    # ==========================================================
    tokenized = DatasetDict()
    for split in ds.keys():
        tokenized[split] = ds[split].map(
            encode_answer_only,
            batched=True,
            num_proc=args.encode_proc,  # 这行就是加速点
            remove_columns=[c for c in ds[split].column_names if c not in ("messages", "label")],
            desc=f"encode [{split}]",
        )

    data_collator = AnswerOnlyCollator(tok)

    # ==========================================================
    # 5. oversample 权重
    # ==========================================================
    weights_for_train = None
    if args.oversample:
        labels = list(ds["train"]["label"])
        weights_for_train = compute_inverse_freq_weights(labels)
        print("[INFO] oversample enabled")

    # ==========================================================
    # 6. 加载基础模型 + LoRA
    # ==========================================================
    use_bf16 = bf16_available()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    try:
        model.resize_token_embeddings(len(tok))
    except Exception as e:
        print("[WARN] resize_token_embeddings skipped:", e)

    if args.init_adapter:
        print(f"[INFO] init adapter from {args.init_adapter}")
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=TARGET_MODULES,
            bias="none",
            use_dora=False,
            inference_mode=False,
        )
        model = get_peft_model(model, lora_cfg)

    # gc 开着的话要让输入也可求导
    if args.gc:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # ==========================================================
    # 7. TrainingArguments
    # ==========================================================
    load_best = args.early_stop
    greater_is_better = None
    if args.early_stop:
        greater_is_better = False if args.early_mode == "min" else True

    train_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.eval_bsz,
        gradient_accumulation_steps=args.grad_acc,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=args.gc,
        report_to="none",
        optim="adamw_torch",
        weight_decay=0.0,
        max_grad_norm=1.0,
        dataloader_num_workers=args.num_workers,
        load_best_model_at_end=load_best,
        metric_for_best_model=args.early_metric if args.early_stop else None,
        greater_is_better=greater_is_better,
    )

    # ==========================================================
    # 8. callbacks
    # ==========================================================
    callbacks = []
    if args.early_stop:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_patience,
                early_stopping_threshold=0.0,
            )
        )
        print(f"[INFO] early stopping enabled: metric={args.early_metric}, mode={args.early_mode}")

    # ==========================================================
    # 9. Trainer
    # ==========================================================
    trainer = OversampleTrainer(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        weights_for_train=weights_for_train,
        callbacks=callbacks,
    )

    # ==========================================================
    # 10. 训练
    # ==========================================================
    try:
        trainer.train(resume_from_checkpoint=args.resume)
    except Exception:
        traceback.print_exc()
        raise

    # ==========================================================
    # 11. 保存模型
    # ==========================================================
    final_dir = os.path.join(args.out, "final")
    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)
    print(f"[OK] saved to {final_dir}")


if __name__ == "__main__":

    main()

