# -*- coding: utf-8 -*-
"""Generate training-curve and baseline-vs-improved figures from result JSONs.

Example
-------
    python plot_results.py --results-dir ./results --out-dir ./assets
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['axes.unicode_minus'] = False


def parse_args():
    parser = argparse.ArgumentParser(description='Plot training results')
    parser.add_argument('--results-dir', type=str, default='./results')
    parser.add_argument('--out-dir', type=str, default='./assets')
    return parser.parse_args()


def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def plot_curves(runs, out_dir):
    """Loss / accuracy curves. ``runs`` is a list of (label, result_dict)."""
    # loss
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    for label, res in runs:
        epochs = [h['epoch'] for h in res['history']]
        plt.plot(epochs, [h['train_loss'] for h in res['history']], label=f'{label} train')
        plt.plot(epochs, [h['val_loss'] for h in res['history']], '--', label=f'{label} val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Curve')
    plt.legend(fontsize=8); plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    for label, res in runs:
        epochs = [h['epoch'] for h in res['history']]
        plt.plot(epochs, [h['train_acc'] for h in res['history']], label=f'{label} train')
        plt.plot(epochs, [h['val_acc'] for h in res['history']], '--', label=f'{label} val')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.title('Accuracy Curve')
    plt.legend(fontsize=8); plt.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'training_curves.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f'saved -> {path}')


def plot_comparison(runs, out_dir):
    labels = [label for label, _ in runs]
    val = [res['best_val_acc'] for _, res in runs]
    test = [res['test_acc'] for _, res in runs]

    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(6.4, 4.2))
    bars1 = plt.bar(x - width / 2, val, width, label='Best Val Acc', color='#7f8c8d')
    bars2 = plt.bar(x + width / 2, test, width, label='Test Acc', color='#c0392b')
    for bars in (bars1, bars2):
        for b in bars:
            plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                     f'{b.get_height():.1f}', ha='center', fontsize=9)
    plt.xticks(x, labels)
    plt.ylabel('Accuracy (%)'); plt.ylim(0, 105)
    plt.title('Baseline vs Improved Fine-tuning')
    plt.legend(fontsize=9); plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, 'baseline_vs_improved.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f'saved -> {path}')


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    candidates = [('Baseline (head only)', 'results_head.json'),
                  ('Improved (partial FT)', 'results_ft.json')]
    runs = []
    for label, fname in candidates:
        path = os.path.join(args.results_dir, fname)
        if os.path.isfile(path):
            runs.append((label, load(path)))
        else:
            print(f'[skip] {path} not found')

    if not runs:
        raise SystemExit('no result json found, run train.py first')

    plot_curves(runs, args.out_dir)
    if len(runs) > 1:
        plot_comparison(runs, args.out_dir)


if __name__ == '__main__':
    main()
