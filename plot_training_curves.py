"""Plot training loss curves from tensorboard event files.

Usage:
    pip install tbparse
    python plot_training_curves.py \
        --log_dir experiments/20260430_dualsft/logs/train_controlnet \
        --output figures/training_curves.pdf
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_scalars(log_dir):
    try:
        from tbparse import SummaryReader
    except ImportError:
        raise SystemExit("Please `pip install tbparse` first.")
    reader = SummaryReader(log_dir, extra_columns={'dir_name'})
    df = reader.scalars
    print(f"Available tags: {sorted(df['tag'].unique())}")
    return df


def smooth(y, weight=0.6):
    """EMA smoothing."""
    s = []
    last = y[0]
    for v in y:
        last = last * weight + v * (1 - weight)
        s.append(last)
    return np.array(s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log_dir',
                   default='experiments/20260430_dualsft/logs/train_controlnet')
    p.add_argument('--output', default='figures/training_curves.pdf')
    p.add_argument('--smooth', type=float, default=0.7,
                   help="EMA smoothing weight (0=raw, 0.9=heavy)")
    args = p.parse_args()

    df = load_scalars(args.log_dir)

    panels = [
        ('train/loss_total', 'Total loss'),
        ('train/loss_mse',   'MSE loss'),
        ('train/loss_wav',   'Wavelet (freq) loss'),
        ('train/grad_norm',  'Gradient norm'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, (tag, ylabel) in enumerate(panels):
        sub = df[df['tag'] == tag].sort_values('step')
        if sub.empty:
            axes[i].set_title(f'{ylabel}\n(no data)')
            axes[i].axis('off')
            continue
        x = sub['step'].values
        y = sub['value'].values
        axes[i].plot(x, y, color='lightgray', alpha=0.5, label='raw')
        axes[i].plot(x, smooth(y, args.smooth), color='C3', linewidth=2,
                     label=f'EMA({args.smooth})')
        axes[i].set_xlabel('Step', fontsize=12)
        axes[i].set_ylabel(ylabel, fontsize=12)
        axes[i].set_title(ylabel, fontsize=13)
        axes[i].grid(True, linestyle='--', alpha=0.4)
        axes[i].legend(fontsize=10)
        if 'loss' in tag:
            axes[i].set_yscale('log')

    plt.tight_layout()
    out_dir = os.path.dirname(args.output) or '.'
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    png_path = os.path.splitext(args.output)[0] + '.png'
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {args.output}")
    print(f"Saved: {png_path}")


if __name__ == '__main__':
    main()
