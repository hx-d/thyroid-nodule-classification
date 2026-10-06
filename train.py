# -*- coding: utf-8 -*-
"""Train a ResNet50 classifier on the TN5000 thyroid ultrasound dataset.

Two fine-tuning strategies are provided so the improvement can be reproduced:

  --finetune-mode head     freeze the whole backbone, train the classifier only
                           (this is the naive baseline: on this imbalanced
                           dataset it collapses to the majority class)
  --finetune-mode partial  freeze conv1/layer1/layer2, fine-tune layer3/layer4
                           and the classifier, with inverse-frequency class
                           weights in the loss (the improved setting)

Example
-------
    python train.py --data-root ./data/TN5000_forReview --finetune-mode partial
"""

import argparse
import json
import os
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import models, transforms

from dataset import ThyroidDataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def build_model(mode, num_classes=2):
    """ResNet50 with an ImageNet-pretrained backbone."""
    model = models.resnet50(weights='DEFAULT')
    if mode == 'head':
        for p in model.parameters():
            p.requires_grad = False
    elif mode == 'partial':
        for name, p in model.named_parameters():
            if name.startswith(('conv1', 'bn1', 'layer1', 'layer2')):
                p.requires_grad = False
    else:
        raise ValueError(f'unknown finetune mode: {mode}')

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_optimizer(model, mode, lr):
    if mode == 'head':
        return optim.Adam(model.fc.parameters(), lr=lr)
    return optim.Adam([
        {'params': model.layer3.parameters(), 'lr': lr},
        {'params': model.layer4.parameters(), 'lr': lr},
        {'params': model.fc.parameters(), 'lr': lr * 10},
    ])


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if not training:
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

    return total_loss / total, 100.0 * correct / total, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='TN5000 thyroid nodule classification')
    parser.add_argument('--data-root', type=str, default='./data/TN5000_forReview')
    parser.add_argument('--out-dir', type=str, default='./results')
    parser.add_argument('--finetune-mode', choices=['head', 'partial'], default='partial',
                        help='head = train classifier only; partial = fine-tune layer3/4')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--step-size', type=int, default=7)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device} | mode: {args.finetune_mode} | epochs: {args.epochs} | lr: {args.lr}')

    train_tf, eval_tf = build_transforms()
    train_ds = ThyroidDataset(args.data_root, 'train', train_tf)
    val_ds = ThyroidDataset(args.data_root, 'val', eval_tf)
    test_ds = ThyroidDataset(args.data_root, 'test', eval_tf)
    print(f'train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}')

    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model = build_model(args.finetune_mode).to(device)

    # inverse-frequency class weights to counter the benign:malignant ~ 3:7 skew
    counts = [0, 0]
    for _, labels in train_loader:
        for c in labels.tolist():
            counts[c] += 1
    total = sum(counts)
    weights = [total / (2 * counts[c]) for c in range(2)]
    print(f'class counts: {counts} -> loss weights: {[round(w, 3) for w in weights]}')
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))

    optimizer = build_optimizer(model, args.finetune_mode, args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    tag = 'head' if args.finetune_mode == 'head' else 'ft'
    best_path = os.path.join(args.out_dir, f'best_model_{tag}.pth')
    history, best_val_acc = [], 0.0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc, _, _ = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history.append({'epoch': epoch,
                        'train_loss': round(tr_loss, 4), 'train_acc': round(tr_acc, 4),
                        'val_loss': round(va_loss, 4), 'val_acc': round(va_acc, 4)})
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_path)
        print(f'Epoch {epoch:02d}/{args.epochs} | train_loss={tr_loss:.4f} train_acc={tr_acc:.2f}% '
              f'| val_loss={va_loss:.4f} val_acc={va_acc:.2f}% | best={best_val_acc:.2f}%')

    model.load_state_dict(torch.load(best_path, weights_only=True))
    te_loss, te_acc, preds, labels = run_epoch(model, test_loader, criterion, device)
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=['Benign(0)', 'Malignant(1)'], digits=4)

    print('\n===== TEST RESULT (best checkpoint by val acc) =====')
    print(f'Test Accuracy: {te_acc:.2f}% | Test Loss: {te_loss:.4f}')
    print(cm)
    print(report)

    result = {
        'finetune_mode': args.finetune_mode,
        'dataset': 'TN5000_forReview (train 3500 / val 500 / test 1000)',
        'epochs': args.epochs, 'batch_size': args.batch_size, 'lr': args.lr,
        'class_weights': [round(w, 4) for w in weights],
        'best_val_acc': round(best_val_acc, 2),
        'test_acc': round(te_acc, 2),
        'test_loss': round(te_loss, 4),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'history': history,
        'elapsed_min': round((time.time() - start) / 60, 1),
    }
    out_json = os.path.join(args.out_dir, f'results_{tag}.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'\nresults saved -> {out_json}')


if __name__ == '__main__':
    main()
