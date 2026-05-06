#!/usr/bin/env python3
# gen_adv.py
# Generate adversarial example files for the project (FGSM, PGD, CW-L2)
# Place next to part1.py and utils.py and run: python gen_adv.py

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import utils
from part1 import load_and_grab

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)
model_fp = './target_model.pt'
model, fg = utils.load_model(model_fp, device=device)

# Grab a small set of benign examples 
benign_x, benign_y = load_and_grab('./data/valtest.npz', 'val', num_batches=4, batch_size=256, shuffle=False)
# benign_x is a torch tensor normalized already (N, C, H, W)
benign_x = benign_x.to(device).float()
benign_y = benign_y.to(device).long().flatten()

print("Benign sample shape:", benign_x.shape, "labels:", benign_y.shape)

loss_fn = nn.CrossEntropyLoss()

########################
# FGSM (L-inf)
########################
def fgsm_attack(model, x, y, eps=8/255.0, device='cuda'):
    mean, std = utils.CIFAR10_CHANNEL_STATS
    # convert to tensors on device
    mean = mean.to(device)
    std = std.to(device)
    # compute per-channel eps (normalized)
    eps_tensor = (eps / std).view(1,3,1,1)

    x_adv = x.clone().detach().requires_grad_(True)
    outputs = model(x_adv)
    loss = loss_fn(outputs, y)
    model.zero_grad()
    loss.backward()
    grad = x_adv.grad.data
    x_adv = x_adv + eps_tensor * grad.sign()
    # pixel range pre-normalization is [0,1]; normalized range:
    x_min = ((0.0 - mean) / std)
    x_max = ((1.0 - mean) / std)
    x_adv = torch.max(torch.min(x_adv, x_max), x_min).detach()
    return x_adv

########################
# PGD (L-inf)
########################
def pgd_attack(model, x, y, eps=8/255.0, alpha=2/255.0, steps=20, device='cuda'):
    mean, std = utils.CIFAR10_CHANNEL_STATS
    mean = mean.to(device)
    std = std.to(device)
    eps_tensor = (eps / std).view(1,3,1,1)
    alpha_tensor = (alpha / std).view(1,3,1,1)

    x_orig = x.clone().detach()
    x_adv = x_orig.clone().detach() + torch.zeros_like(x_orig).uniform_(-1e-6, 1e-6)
    x_adv.requires_grad = True

    for i in range(steps):
        outputs = model(x_adv)
        loss = loss_fn(outputs, y)
        model.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha_tensor * x_adv.grad.sign()
            # project onto L-inf ball around x_orig
            x_adv = torch.max(torch.min(x_adv, x_orig + eps_tensor), x_orig - eps_tensor)
            # clamp to valid range in normalized domain
            x_min = ((0.0 - mean) / std)
            x_max = ((1.0 - mean) / std)
            x_adv = torch.max(torch.min(x_adv, x_max), x_min)
            x_adv.requires_grad_(True)
    return x_adv.detach()

########################
# CW-L2 (simple optimizer-based version)
########################
def cw_l2_attack(model, x, y, steps=200, c=1e-2, lr=1e-2, device='cuda'):
    mean, std = utils.CIFAR10_CHANNEL_STATS
    mean = mean.to(device)
    std = std.to(device)

    x_orig = x.clone().detach()
    batch_size = x_orig.size(0)

    # delta is in normalized space
    delta = torch.zeros_like(x_orig, requires_grad=True, device=device)
    opt = optim.Adam([delta], lr=lr)

    kappa = 0.0  # confidence margin
    x_min = ((0.0 - mean) / std)
    x_max = ((1.0 - mean) / std)

    for i in range(steps):
        adv = x_orig + delta
        # clamp adv to valid normalized range
        adv = torch.max(torch.min(adv, x_max), x_min)

        outputs = model(adv)  # shape [N, C]
        # true logits
        true_logits = outputs.gather(1, y.view(-1, 1)).squeeze(1)  # shape [N]

        # mask out the true class by setting it to a very large negative number
        masked = outputs.clone()
        # set true class logits to -1e9 so they are not selected as max
        masked[torch.arange(batch_size, device=device), y] = -1e9
        max_other_logits = masked.max(dim=1)[0]  # shape [N]

        # cw margin loss: encourage max_other - true + kappa > 0
        loss1 = torch.clamp(max_other_logits - true_logits + kappa, min=0.0).mean()

        # L2 term: squared L2 norm per-sample, then mean
        loss2 = (delta.view(delta.size(0), -1).pow(2).sum(dim=1)).mean()

        loss = loss2 + c * loss1

        opt.zero_grad()
        loss.backward()
        opt.step()

        # project adv back into valid range by adjusting delta
        with torch.no_grad():
            adv = x_orig + delta
            adv = torch.max(torch.min(adv, x_max), x_min)
            delta.data = adv - x_orig

    adv_final = torch.max(torch.min(x_orig + delta.detach(), x_max), x_min)
    return adv_final.detach()


########################
# Generate and save files
########################
N = benign_x.shape[0]
print("Generating FGSM...")
t0 = time.time()
adv_fgsm = fgsm_attack(model, benign_x, benign_y, eps=8/255.0, device=device)
print("Done FGSM in", time.time() - t0)

print("Generating PGD...")
t0 = time.time()
adv_pgd = pgd_attack(model, benign_x, benign_y, eps=8/255.0, alpha=2/255.0, steps=20, device=device)
print("Done PGD in", time.time() - t0)

print("Generating CW-L2 (this may take a while)...")
t0 = time.time()
adv_cw = cw_l2_attack(model, benign_x, benign_y, steps=50, c=1e-2, lr=1e-2, device=device)
print("Done CW in", time.time() - t0)

def to_np(x):
    import torch, numpy as np
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

np.savez_compressed('advexp_fgsm.npz',
    adv_x=to_np(adv_fgsm),
    benign_x=to_np(benign_x),
    benign_y=to_np(benign_y)
)
np.savez_compressed('advexp_pgd.npz',
    adv_x=to_np(adv_pgd),
    benign_x=to_np(benign_x),
    benign_y=to_np(benign_y)
)
np.savez_compressed('advexp_cw.npz',
    adv_x=to_np(adv_cw),
    benign_x=to_np(benign_x),
    benign_y=to_np(benign_y)
)

print("Saved advexp_fgsm.npz, advexp_pgd.npz, advexp_cw.npz")
