# -*- coding: utf-8 -*-
"""Evaluate a saved checkpoint on the test split and export figures.

Example
-------
    python evaluate.py --data-root ./data/TN5000_forReview \
                       --weights ./results/best_model_ft.pth \
                       --out-dir ./assets
"""

import argparse
import os
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                             roc_curve)
from torch.utils.data import DataLoader
from torchvision import models, transforms

from dataset import ThyroidDataset

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a thyroid classifier')
    parser.add_argument('--data-root', type=str, default='./data/TN5000_forReview')
    parser.add_argument('--weights', type=str, default='./results/best_model_ft.pth')
    parser.add_argument('--out-dir', type=str, default='./assets')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = ThyroidDataset(args.data_root, args.split, eval_tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model = model.to(device).eval()

    preds, labels, probs = [], [], []
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images.to(device))
            probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
            preds.extend(logits.argmax(1).cpu().numpy().tolist())
            labels.extend(targets.numpy().tolist())

    acc = 100.0 * float(np.mean(np.array(preds) == np.array(labels)))
    print(f'\n{args.split} set | images: {len(labels)} | accuracy: {acc:.2f}%')
    print(classification_report(labels, preds, target_names=['Benign(0)', 'Malignant(1)'], digits=4))

    # ---- confusion matrix ----
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(5.2, 4.4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title(f'Confusion Matrix ({args.split}, acc={acc:.2f}%)')
    plt.tight_layout()
    cm_path = os.path.join(args.out_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150); plt.close()

    # ---- ROC curve ----
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5.0, 4.4))
    plt.plot(fpr, tpr, color='#c0392b', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Malignant vs Benign)')
    plt.legend(loc='lower right'); plt.tight_layout()
    roc_path = os.path.join(args.out_dir, 'roc_curve.png')
    plt.savefig(roc_path, dpi=150); plt.close()

    print(f'AUC: {roc_auc:.4f}')
    print(f'saved -> {cm_path}')
    print(f'saved -> {roc_path}')


if __name__ == '__main__':
    main()
