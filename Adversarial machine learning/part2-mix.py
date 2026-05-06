#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" CIS6261TML -- Project Option 1 -- part2-2.py

Mixed Adversarial Training: Control the ratio of clean vs adversarial examples.
This allows us to explore the tradeoff curve between accuracy, robustness, and privacy.

Usage:
    python part2-2.py --adv_ratio 0.25   # 25% adversarial, 75% clean
    python part2-2.py --adv_ratio 0.50   # 50% adversarial, 50% clean
    python part2-2.py --adv_ratio 0.75   # 75% adversarial, 25% clean
"""

import sys
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import scipy.stats as stats
from sklearn.metrics import confusion_matrix

import utils

# CIFAR-100 normalization constants
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_cifar100_loaders(batch_size=128):
    """Returns CIFAR-100 DataLoaders for train and test sets."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    return trainloader, testloader, trainset


def pgd_attack(model, images, labels, device, eps=8/255, alpha=2/255, iters=7):
    """PGD attack for adversarial training."""
    images = images.to(device)
    labels = labels.to(device)
    loss_fn = nn.CrossEntropyLoss()

    mean = torch.tensor(CIFAR100_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR100_STD, device=device).view(1, 3, 1, 1)

    lower_bound = (0 - mean) / std
    upper_bound = (1 - mean) / std

    eps_normalized = eps / std
    alpha_normalized = alpha / std

    original_images = images.clone().detach()
    adv_images = images.clone().detach()

    for i in range(iters):
        adv_images.requires_grad = True
        outputs = model(adv_images)

        model.zero_grad()
        loss = loss_fn(outputs, labels)
        loss.backward()

        adv_images = adv_images + alpha_normalized * adv_images.grad.sign()
        eta = torch.clamp(adv_images - original_images, min=-eps_normalized, max=eps_normalized)
        adv_images = original_images + eta
        adv_images = torch.clamp(adv_images, min=lower_bound, max=upper_bound).detach()

    return adv_images


def train_model_mixed(model, trainloader, testloader, device, epochs=20, adv_ratio=0.5):
    """
    Trains a model using MIXED adversarial training.

    Args:
        adv_ratio: Fraction of batch to train on adversarial examples (0.0 to 1.0)
                   0.0 = standard training (all clean)
                   1.0 = full adversarial training (all perturbed)
                   0.5 = 50% clean, 50% adversarial
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n=== Mixed Adversarial Training (adv_ratio={adv_ratio}) ===")
    print(f"Training: {int(adv_ratio*100)}% adversarial, {int((1-adv_ratio)*100)}% clean\n")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        st_epoch = time.time()

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            # Determine how many samples to make adversarial
            num_adv = int(batch_size * adv_ratio)

            if num_adv > 0 and num_adv < batch_size:
                # Mixed: some clean, some adversarial
                clean_inputs = inputs[:batch_size - num_adv]
                clean_targets = targets[:batch_size - num_adv]

                adv_inputs_orig = inputs[batch_size - num_adv:]
                adv_targets = targets[batch_size - num_adv:]
                adv_inputs = pgd_attack(model, adv_inputs_orig, adv_targets, device)

                # Concatenate clean and adversarial
                mixed_inputs = torch.cat([clean_inputs, adv_inputs], dim=0)
                mixed_targets = torch.cat([clean_targets, adv_targets], dim=0)
            elif num_adv == 0:
                # All clean (standard training)
                mixed_inputs = inputs
                mixed_targets = targets
            else:
                # All adversarial (full adversarial training)
                mixed_inputs = pgd_attack(model, inputs, targets, device)
                mixed_targets = targets

            optimizer.zero_grad()
            outputs = model(mixed_inputs)
            loss = criterion(outputs, mixed_targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += mixed_targets.size(0)
            correct += predicted.eq(mixed_targets).sum().item()

        et_epoch = time.time()
        epoch_time = et_epoch - st_epoch

        val_acc = utils.eval_model(model, testloader, device)

        print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_time:.1f}s | "
              f"Train Loss: {train_loss/(batch_idx+1):.3f} | "
              f"Train Acc: {100.*correct/total:.2f}% | "
              f"Val Acc: {100.*val_acc:.2f}%")

        scheduler.step()

    return model


######### Evaluation Functions #########

@torch.no_grad()
def evaluate_clean_accuracy(model, testloader, device):
    """Evaluate clean accuracy on test set."""
    model.eval()
    correct = 0
    total = 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return correct / total


def evaluate_adversarial_accuracy(model, testloader, device, eps=8/255, alpha=2/255, iters=7):
    """Evaluate accuracy under PGD attack."""
    model.eval()
    correct = 0
    total = 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        adv_images = pgd_attack(model, images, labels, device, eps=eps, alpha=alpha, iters=iters)
        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total


######### MIA Functions #########

@torch.no_grad()
def basic_predict(model, x, device="cuda"):
    model.eval()
    x = x.to(device)
    return model(x)


@torch.no_grad()
def simple_conf_threshold_mia(predict_fn, x, thresh=0.4, device="cuda"):
    pred_y = predict_fn(x, device).cpu()
    pred_y_probas = torch.softmax(pred_y, dim=1).numpy()
    pred_y_conf = np.max(pred_y_probas, axis=-1)
    return (pred_y_conf > thresh).astype(int)


@torch.no_grad()
def simple_logits_threshold_mia(predict_fn, x, thresh=3, device="cuda"):
    pred_y = predict_fn(x, device).cpu().numpy()
    pred_y_max_logit = np.max(pred_y, axis=-1)
    return (pred_y_max_logit > thresh).astype(int)


@torch.no_grad()
def loss_thresh_mia(predict_fn, x, true_labels, thresh=3.0, device="cuda"):
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    pred_y = predict_fn(x, device).cpu()
    losses = criterion(pred_y, true_labels.cpu()).numpy()
    return (losses < thresh).astype(int)


@torch.no_grad()
def gap_attack_mia(predict_fn, x, true_labels, device="cuda"):
    pred_y = predict_fn(x, device).cpu()
    pred_labels = torch.argmax(pred_y, dim=1).numpy()
    return (pred_labels == true_labels.cpu().numpy()).astype(int)


@torch.no_grad()
def augmentation_mia(predict_fn, x, shifts=7, angle=9.9, device="cuda"):
    pred_y = predict_fn(x, device).cpu()
    pred_labels = torch.argmax(pred_y, dim=1).numpy()

    x_rot = torch.stack([TF.rotate(img, angle=angle) for img in x])
    x_aug = torch.roll(x_rot, shifts=shifts, dims=3)

    pred_aug = predict_fn(x_aug, device).cpu()
    pred_aug_labels = torch.argmax(pred_aug, dim=1).numpy()

    return (pred_labels == pred_aug_labels).astype(int)


def get_mia_splits(num_members=512, num_nonmembers=512):
    """Create member/non-member splits from CIFAR-100."""
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=False, transform=transform_eval)
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=False, transform=transform_eval)

    member_x, member_y = [], []
    nonmember_x, nonmember_y = [], []

    for idx in range(num_members):
        img, label = trainset[idx]
        member_x.append(img)
        member_y.append(label)

    for idx in range(num_nonmembers):
        img, label = testset[idx]
        nonmember_x.append(img)
        nonmember_y.append(label)

    return (torch.stack(member_x), torch.tensor(member_y),
            torch.stack(nonmember_x), torch.tensor(nonmember_y))


def evaluate_mia(model, device):
    """Evaluate all MIA attacks and return results."""
    print("\n------------ Privacy Evaluation (MIA) ----------")

    in_x, in_y, out_x, out_y = get_mia_splits()
    mia_eval_x = torch.cat([in_x, out_x], 0)
    true_labels = torch.cat([in_y, out_y], 0)
    mia_eval_y = torch.cat([torch.ones(len(in_x)), torch.zeros(len(out_x))], 0)
    mia_eval_y = mia_eval_y.numpy().reshape((-1, 1))

    def predict_fn(x, dev):
        return basic_predict(model, x, device=dev)

    results = {}

    attacks = [
        ('Confidence Threshold', lambda: simple_conf_threshold_mia(predict_fn, mia_eval_x, device=device)),
        ('Logits Threshold', lambda: simple_logits_threshold_mia(predict_fn, mia_eval_x, device=device)),
        ('Loss Threshold', lambda: loss_thresh_mia(predict_fn, mia_eval_x, true_labels, device=device)),
        ('Gap Attack', lambda: gap_attack_mia(predict_fn, mia_eval_x, true_labels, device=device)),
        ('Augmentation', lambda: augmentation_mia(predict_fn, mia_eval_x, device=device)),
    ]

    for attack_name, attack_fn in attacks:
        preds = attack_fn().reshape((-1, 1))
        cm = confusion_matrix(mia_eval_y, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        acc = np.trace(cm) / np.sum(cm)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        adv = tpr - fpr

        print(f"{attack_name}: Acc={100*acc:.2f}%, Advantage={adv:.3f}")
        results[attack_name] = {'acc': acc, 'advantage': adv}

    return results


def main():
    parser = argparse.ArgumentParser(description='Mixed Adversarial Training')
    parser.add_argument('--adv_ratio', type=float, default=0.5,
                        help='Ratio of adversarial examples (0.0 to 1.0)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    print('### Mixed Adversarial Training ###')
    print(f'### Adversarial Ratio: {args.adv_ratio} ###')
    print('### Python version: ' + sys.version)
    print('### PyTorch version: ' + torch.__version__)
    print('------------')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Device: {device} ---")

    st = time.time()

    # Load data
    print('\n------------ Loading Data ----------')
    trainloader, testloader, trainset = get_cifar100_loaders()
    print("CIFAR-100 data loaded.")

    # Create model
    print('\n------------ Creating Model ----------')
    model = utils.get_resnet18_cifar(num_classes=100)
    print("ResNet-18 model created.")

    # Train with mixed adversarial training
    model = train_model_mixed(model, trainloader, testloader, device,
                              epochs=args.epochs, adv_ratio=args.adv_ratio)

    # Evaluate
    print('\n------------ Final Evaluation ----------')
    clean_acc = evaluate_clean_accuracy(model, testloader, device)
    print(f"Clean Accuracy: {100*clean_acc:.2f}%")

    adv_acc = evaluate_adversarial_accuracy(model, testloader, device, eps=8/255)
    print(f"Adversarial Accuracy (eps=8/255): {100*adv_acc:.2f}%")
    print(f"Robustness Retention: {100*adv_acc/clean_acc:.1f}%")

    # MIA evaluation
    mia_results = evaluate_mia(model, device)

    # Save model
    ratio_str = str(int(args.adv_ratio * 100))
    model_path = f'./part2_model_adv{ratio_str}.pt'
    torch.save({
        'model': model.state_dict(),
        'adv_ratio': args.adv_ratio,
        'clean_acc': clean_acc,
        'adv_acc': adv_acc,
        'mia_results': mia_results
    }, model_path)
    print(f"\nModel saved to {model_path}")

    # Save results summary
    results_path = f'./results_adv{ratio_str}.txt'
    with open(results_path, 'w') as f:
        f.write(f"=== Results for adv_ratio={args.adv_ratio} ===\n\n")
        f.write(f"Clean Accuracy: {100*clean_acc:.2f}%\n")
        f.write(f"Adversarial Accuracy (eps=8/255): {100*adv_acc:.2f}%\n")
        f.write(f"Robustness Retention: {100*adv_acc/clean_acc:.1f}%\n\n")
        f.write("MIA Results:\n")
        for attack, res in mia_results.items():
            f.write(f"  {attack}: Acc={100*res['acc']:.2f}%, Adv={res['advantage']:.3f}\n")
    print(f"Results saved to {results_path}")

    et = time.time()
    print(f"\nTotal time: {et-st:.1f} seconds ({(et-st)/60:.1f} minutes)")


if __name__ == "__main__":
    main()
