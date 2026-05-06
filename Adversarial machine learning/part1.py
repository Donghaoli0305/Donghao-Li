#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" CIS6261TML -- Project Option 1 -- part1.py

# This file contains the part1 code
"""

import sys
import os

import time

import numpy as np

import sklearn
from sklearn.metrics import confusion_matrix
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import scipy.stats as stats
import torch
import joblib

import utils  # we need this

######### Prediction Fns #########

"""
## Basic prediction function
"""


@torch.no_grad()
def basic_predict(model, x, device="cuda"):
    x = x.to(device)
    logits = model(x)
    return logits


# TODO: implement your defense(s) as a new prediction function
# Make sure it is compatible with the rest of the code in this file:
# - it needs to take model, x, device
# - it needs to return logits
# Note: if your predict function operates on probabilities/labels (instead of logits), that is fine provided you adjust the rest of the code.
# Put your code here

def _rotate_tensor(img, angle):
    try:
        return TF.rotate(img, float(angle), interpolation=InterpolationMode.BILINEAR, fill=0.0)
    except TypeError:
        # fallback: some versions require per-channel fill
        c = img.size(0)
        fill = (0.0,) if c == 1 else [0.0] * c
        return TF.rotate(img, float(angle), interpolation=InterpolationMode.BILINEAR, fill=fill)


@torch.no_grad()
def defended_predict(
    model,
    x,
    device="cuda",
    sigma=0.016,     # Randomized smoothing
    rotate_deg=5.00,     # rotation degree
    shift_px=2,     #shift pixal number
    T=1.35,          # Temperature
    out_noise=2e-4  # Output noise to break thresholds
):
    model.eval()
    x = x.to(device)

    # how many noisy forwards to average
    runs = 2
    
    acc_logits = None
    for _ in range(runs):
        xi = x

        # input Gaussian noise
        if sigma > 0.0:
            xi = xi + sigma * torch.randn_like(xi)

        # small random rotation in [-rotate_deg, +rotate_deg]
        if rotate_deg and rotate_deg > 0.0:
            B = xi.size(0)
            angles = (torch.rand(B, device=xi.device) * 2.0 - 1.0) * float(rotate_deg)
            imgs = []
            for i in range(B):
                try:
                    imgs.append(
                        TF.rotate(
                            xi[i],
                            float(angles[i].item()),
                            interpolation=InterpolationMode.BILINEAR,
                            fill=0.0
                        )
                    )
                except TypeError:
                    # fallback for torchvision versions requiring per-channel 'fill'
                    c = xi.size(1)
                    fill = (0.0,) if c == 1 else [0.0] * c
                    imgs.append(
                        TF.rotate(
                            xi[i],
                            float(angles[i].item()),
                            interpolation=InterpolationMode.BILINEAR,
                            fill=fill
                        )
                    )
            xi = torch.stack(imgs, dim=0)

        # tiny horizontal shift in [-shift_px, +shift_px]
        if shift_px and shift_px > 0:
            B = xi.size(0)
            shifts = torch.randint(low=-shift_px, high=shift_px + 1, size=(B,), device=xi.device)
            xi = torch.stack(
                [torch.roll(xi[i], shifts=int(shifts[i].item()), dims=2)
                 for i in range(B)], 
                dim=0
            )

        # forward pass
        logits = model(xi)

        # temperature scaling with a small jitter to destabilize per-sample loss
        if T and T != 1.0:
            jitter = 0.05  # jitter; set to 0.05~0.10 as needed
            T_eff = T * (1.0 + jitter * (2.0 * torch.rand(1, device=logits.device).item() - 1.0))
            logits = logits / T_eff

        # tiny output noise (set to 0.0 for AE evaluation if you want higher adv acc)
        if out_noise and out_noise > 0.0:
            logits = logits + out_noise * torch.randn_like(logits)

        acc_logits = logits if acc_logits is None else (acc_logits + logits)

    return acc_logits / float(runs)


# TODO [optional] implement new MIA attacks.
# Put your code here

"""
NOTE: Several of the added MIAs require access to information like the true labels of the input x or the training loss distribution.
      To keep the function signatures consistent with the pipeline, these variables are assumed to be globally available, rather than added parameters.
      The variable names are consistent with part1.py code. Copy these functions into part1.py directly and it should work.
      If a NameError is raised, the function prints an error message and returns all zeros.
      
"""

######### Membership Inference Attacks (MIAs) #########

"""
## A very simple confidence threshold-based MIA
"""


@torch.no_grad()
def simple_conf_threshold_mia(predict_fn, x, thresh=0.999, device="cuda"):
    pred_y = predict_fn(x, device).cpu()
    pred_y_probas = torch.softmax(pred_y, dim=1).numpy()
    pred_y_conf = np.max(pred_y_probas, axis=-1)
    return (pred_y_conf > thresh).astype(int)


"""
## A very simple logit threshold-based MIA
"""


@torch.no_grad()
def simple_logits_threshold_mia(predict_fn, x, thresh=9, device="cuda"):
    pred_y = predict_fn(x, device).cpu().numpy()
    pred_y_max_logit = np.max(pred_y, axis=-1)
    return (pred_y_max_logit > thresh).astype(int)


# TODO [optional] implement new MIA attacks.
# Put your code here
"""
    Simple loss threshold-based MIA using cross-entropy loss.
    Assumes access to true labels (in_y, out_y) globally.
"""


@torch.no_grad()
def loss_thresh_mia(predict_fn, x, thresh=0.001, device="cuda"):
    try:
        true_labels = torch.cat([in_y, out_y], 0).cpu().detach()
    except NameError:
        print("[ERROR: Loss-threshold MIA] in_y, out_y are not defined globally. This attack requires true labels to be globally available.")
        return np.zeros(x.shape[0])
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    # Calculate loss from logits and true labels
    pred_y = predict_fn(x, device).cpu()
    losses = criterion(pred_y, true_labels).cpu().numpy()
    return (losses < thresh).astype(int)


"""
    Loss threshold-based MIA using probability to lay within training loss distribution.
    Assumes access to true labels  globally.
    Assumes knowledge of the loss distributions of training and validation data.
    
    Without modifying other parts of the code, this calculates the loss from the dataset manually.
    Makes global variables for training and validation losses globally to avoid recomputation. Useful for multiple function calls.
    
    Taken from hw3
"""


@torch.no_grad()
def loss_prob_mia(predict_fn, x, thresh=0.402, device="cuda"):
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    global train_losses, val_losses
    try:
        train_losses = train_losses
        val_losses = val_losses
        true_labels = torch.cat([in_y, out_y], 0).cpu()
    except NameError:
        # Losses are not cached globally. Computing losses from train and val data."
        try:
            model_train_loader = train_loader
            model_val_loader = val_loader
            true_labels = torch.cat([in_y, out_y], 0).cpu()
        except NameError:
            print("[ERROR: Loss-threshold MIA] train, val, and membership data are not defined globally. This attack requires these to be globally available.")
            return np.zeros(x.shape[0])

        # Find loss from training and val data
        train_losses = []
        val_losses = []
        for batch_x, batch_y in model_train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred_y = predict_fn(batch_x, device)
            losses = criterion(pred_y, batch_y).cpu().numpy()
            train_losses.extend(losses)
        for batch_x, batch_y in model_val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred_y = predict_fn(batch_x, device)
            losses = criterion(pred_y, batch_y).cpu().numpy()
            val_losses.extend(losses)

        train_losses = np.array(train_losses)
        val_losses = np.array(val_losses)

    # Find mean and std of losses
    mean_train_loss = np.mean(train_losses)
    std_train_loss = np.std(train_losses)
    mean_val_loss = np.mean(val_losses)
    std_val_loss = np.std(val_losses)

    # Calculate loss from logits and true labels
    pred_y = predict_fn(x, device).cpu()
    losses = criterion(pred_y, true_labels).cpu().numpy()

    # Calculate probability of being in training set based on loss and normal distribution
    probs = stats.norm.cdf(losses, loc=mean_train_loss, scale=std_train_loss)
    return (probs < thresh).astype(int)


"""
    Predict IN if predicted label is same as true label.
    Assumes access to true labels (in_y, out_y) globally.

"""


@torch.no_grad()
def gap_attack_mia(predict_fn, x, device="cuda"):
    try:
        true_labels = torch.cat([in_y, out_y], 0).cpu(
        ).detach().numpy().reshape((-1, 1))
    except NameError:
        print("[ERROR: Gap Attack] in_y, out_y are not defined globally. This attack requires true labels to be globally available.")
        return np.zeros(x.shape[0])
    pred_y = predict_fn(x, device).cpu()
    pred_y_probas = torch.softmax(pred_y, dim=1).numpy()
    pred_labels = np.argmax(pred_y_probas, axis=1)
    
    # Predict IN if model predicted correctly
    return (pred_labels == true_labels.reshape(-1)).astype(int)


"""
    Compare outputs before and after translation and rotation. Predict IN if label does not change.
    This is label only and does not require precise logits, only the max label.
"""

"""
    Shokri attack from hw3
    Requires generation of attack models beforehand and saving to ./shokri_attack/attack_model_class_{i}.pkl
    Assumes access to true labels (in_y, out_y) globally.
"""


@torch.no_grad()
def augmentation_mia(predict_fn, x, shifts=7, angle=9.9, device="cuda"):
    pred_y = predict_fn(x, device).cpu()
    pred_y_probas = torch.softmax(pred_y, dim=1).numpy()
    pred_labels = np.argmax(pred_y_probas, axis=1)
    # Translation and rotation
    x_rot = torch.stack([_rotate_tensor(img, angle) for img in x])
    x_aug = torch.roll(x_rot, shifts=shifts, dims=2)

    pred_aug = predict_fn(x_aug, device).cpu()
    pred_aug_labels = torch.softmax(pred_aug, dim=1).argmax(dim=1).numpy()
    
    # Predict IN if label does not change after augmentation
    return (pred_labels == pred_aug_labels).astype(int)


@torch.no_grad()
def shokri_mia(predict_fn, x, device="cuda"):
    # Load attack models
    out_dir = './shokri_attack'
    attack_models = []
    for idx in range(10):
        model_fp = os.path.join(out_dir, f'attack_model_class_{idx}.pkl')
        if not os.path.exists(model_fp):
            print(
                f"[ERROR: Shokri MIA] Attack model file {model_fp} does not exist. Please generate the attack models first using genshokri.py.")
            return np.zeros(x.shape[0])
        am = joblib.load(model_fp)
        attack_models.append(am)

    # One-hot encode true labels
    try:
        true_labels = torch.cat([in_y, out_y], 0).cpu().detach()
        y_targets = torch.nn.functional.one_hot(
            true_labels, num_classes=10).numpy()
    except NameError:
        print("[ERROR: Shokri MIA] in_y, out_y are not defined globally. This attack requires true labels to be globally available.")
        return np.zeros(x.shape[0])

    num_classes = y_targets.shape[1]
    assert len(attack_models) == num_classes
    y_targets_labels = np.argmax(y_targets, axis=-1)

    # Obtain prediction vectors
    in_or_out_pred = np.zeros((x.shape[0],))
    pred_y = predict_fn(x, device).cpu()
    pv = torch.softmax(pred_y, dim=1).numpy()
    assert pv.shape[0] == y_targets_labels.shape[0]

    # Predict using class-specific attack models
    for i in range(pv.shape[0]):
        
        label = int(y_targets_labels[i])
        assert 0 <= label < num_classes

        am = attack_models[label]

        pred = am.predict(pv[i, :].reshape(1, -1))  
        in_or_out_pred[i] = float(pred[0])
    return in_or_out_pred





######### Adversarial Examples #########


# TODO [optional] implement new adversarial examples attacks.
# Put your code here
# Note: you should have your code save the data to file so it can be loaded and evaluated in Main() (see below).


def load_and_grab(fp, name, num_batches=4, batch_size=256, shuffle=True):
    loader = utils.make_loader(
        fp, f"{name}_x", f"{name}_y", batch_size=batch_size, shuffle=shuffle)
    utils.check_loader(loader)

    return utils.grab_from_loader(loader, num_batches=num_batches)


def load_advex(fp):
    data = np.load(fp)
    return data['adv_x'], data['benign_x'], data['benign_y']

######### Main() #########


if __name__ == "__main__":

    # Let's check our software versions
    print('### Python version: ' + __import__('sys').version)
    print('### NumPy version: ' + np.__version__)
    print('### Pytorch version: ' + torch.__version__)
    print('------------')

    # deterministic seed for reproducibility
    seed = 42

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Device: {device} ---")
    print("-------------------")

    # keep track of time
    st = time.time()

    # load the data
    print('\n------------ Loading Data & Model ----------')

    train_loader = utils.make_loader(
        './data/train.npz', 'train_x', 'train_y', batch_size=256, shuffle=False)
    utils.check_loader(train_loader)

    # val loader
    val_loader = utils.make_loader(
        './data/valtest.npz', 'val_x', 'val_y', batch_size=512, shuffle=False)
    utils.check_loader(val_loader)

    # create the model object
    model_fp = './target_model.pt'
    assert os.path.exists(model_fp)  # model must exist

    model, fg = utils.load_model(model_fp, device=device)
    assert fg == "0CCE0F932C863D6648E0", f"Modified model file {model_fp}!"

    st_after_model = time.time()

    # let's evaluate the raw model on the train and val data
    train_acc = utils.eval_model(model, train_loader, device=device)
    val_acc = utils.eval_model(model, val_loader, device=device)
    print(
        f"[Raw model] Train accuracy: {train_acc:.4f} ; Val accuracy: {val_acc:.4f}.")

    # let's wrap the model prediction function so it could be replaced to implement a defense
    # Turn this to True to evaluate your defense (turn it back to False to see the undefended model).
    defense_enabled = True
    if defense_enabled:
        def predict_fn(x, dev): return defended_predict(model, x, device=dev)
    else:
        # predict_fn points to undefended model
        def predict_fn(x, dev): return basic_predict(model, x, device=dev)

    # now let's evaluate the model with this prediction function wrapper
    train_acc = utils.eval_wrapper(predict_fn, train_loader, device=device)
    val_acc = utils.eval_wrapper(predict_fn, val_loader, device=device)

    print(
        f"[Model] Train accuracy: {train_acc:.4f} ; Val accuracy: {val_acc:.4f}.")

    # evaluating the privacy of the model wrt membership inference
    # load the data
    in_x, in_y = load_and_grab('./data/members.npz', 'members', num_batches=2)
    out_x, out_y = load_and_grab(
        './data/nonmembers.npz', 'nonmembers', num_batches=2)

    mia_eval_x = torch.cat([in_x, out_x], 0)
    mia_eval_y = torch.cat([torch.ones_like(in_y), torch.zeros_like(out_y)], 0)
    mia_eval_y = mia_eval_y.cpu().detach().numpy().reshape((-1, 1))

    assert mia_eval_x.shape[0] == mia_eval_y.shape[0]

    # so we can add new attack functions as needed
    print('\n------------ Privacy Attacks ----------')
    mia_attack_fns = []
    mia_attack_fns.append(
        ('Simple Conf threshold MIA', simple_conf_threshold_mia))
    mia_attack_fns.append(('Simple Logits threshold MIA',
                          simple_logits_threshold_mia))
    mia_attack_fns.append(('Loss threshold MIA', loss_thresh_mia))
    mia_attack_fns.append(('Loss prob MIA', loss_prob_mia))
    mia_attack_fns.append(('Gap Attack MIA', gap_attack_mia))
    mia_attack_fns.append(('Augmentation MIA', augmentation_mia))
    mia_attack_fns.append(('Shokri et al. MIA',shokri_mia))
  

    for i, tup in enumerate(mia_attack_fns):
        attack_str, attack_fn = tup

        in_out_preds = attack_fn(
            predict_fn, mia_eval_x, device=device).reshape((-1, 1))
        assert in_out_preds.shape == mia_eval_y.shape, 'Invalid attack output format'

        cm = confusion_matrix(mia_eval_y, in_out_preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        attack_acc = float(np.trace(cm)) / float(np.sum(cm))
        attack_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        attack_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        attack_adv = attack_tpr - attack_fpr
        attack_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        attack_recall = attack_tpr
        attack_f1 = tp / (tp + 0.5*(fp + fn)) if (tp + 0.5*(fp + fn)) > 0 else 0.0

        print(f"{attack_str} --- Attack acc: {100*attack_acc:.2f}%; advantage: {attack_adv:.3f}; precision: {attack_precision:.3f}; recall: {attack_recall:.3f}; f1: {attack_f1:.3f}.")

    # evaluating the robustness of the model wrt adversarial examples
    print('\n------------ Adversarial Examples ----------')
    advexp_fps = []

    advexp_fps.append(('Attack0', 'advexp0.npz', '519D7F5E79C3600B366A'))
    advexp_fps.append(('Attack FGSM', 'advexp_fgsm.npz', None))
    advexp_fps.append(('Attack PGD', 'advexp_pgd.npz', None))
    advexp_fps.append(('Attack CW', 'advexp_cw.npz', None))

    for i, tup in enumerate(advexp_fps):
        attack_str, attack_fp, attack_hash = tup

        assert os.path.exists(attack_fp), f"Attack file {attack_fp} not found."
        _, fg = utils.memv_filehash(attack_fp)
        if attack_hash is not None:
            assert fg == attack_hash, f"Modified attack file {attack_fp}."

        # load the attack data
        adv_x, benign_x, benign_y = load_advex(attack_fp)
        benign_y = benign_y.flatten()

        benign_pred_y = predict_fn(
            torch.from_numpy(benign_x), device).cpu().numpy()
        benign_pred_y = np.argmax(benign_pred_y, axis=-1).astype(int)
        benign_acc = np.mean(benign_y == benign_pred_y)

        adv_pred_y = predict_fn(torch.from_numpy(adv_x), device).cpu().numpy()
        adv_pred_y = np.argmax(adv_pred_y, axis=-1).astype(int)
        adv_acc = np.mean(benign_y == adv_pred_y)

        print(
            f"{attack_str} [{fg}] --- Benign acc: {100*benign_acc:.2f}%; adversarial acc: {100*adv_acc:.2f}%")
    print('------------\n')

    et = time.time()
    total_sec = et - st
    loading_sec = st_after_model - st

    print(
        f"Elapsed time -- total: {total_sec:.1f} seconds (data & model loading: {loading_sec:.1f} seconds).")

    sys.exit(0)
