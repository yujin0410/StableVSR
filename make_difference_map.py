"""Difference map |GT - method| for one frame, all methods.

Shows where each method deviates from GT. Brighter heat = larger error.

Usage:
    python make_difference_map.py --seq 000 --frame 50 \
        --output figures/diff_map.pdf
"""

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


PATHS = {
    'BasicVSR++': '/mnt/HDD_raid1/yjcho/BasicVSR_PlusPlus/reds/{seq}/{idx:08d}.png',
    'StableVSR':  '/mnt/HDD_raid1/yjcho/stablevsr_reds/{seq}/{idx:08d}.png',
    'Ours':       '/mnt/HDD_raid1/yjcho/20260430/reds/{seq}/{idx:08d}.png',
}
GT_TMPL = '/mnt/HDD_raid1/yjcho/data/REDS/test/gt/{seq}/{idx:08d}.png'


def load_y(path):
    img = np.array(Image.open(path).convert('RGB')).astype(np.float32) / 255.0
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def align(a, b):
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seq', default='000')
    p.add_argument('--frame', type=int, default=50)
    p.add_argument('--output', default='figures/diff_map.pdf')
    args = p.parse_args()

    gt_path = GT_TMPL.format(seq=args.seq, idx=args.frame)
    gt = load_y(gt_path)

    diffs = {}
    for name, tmpl in PATHS.items():
        path = tmpl.format(seq=args.seq, idx=args.frame)
        if not os.path.isfile(path):
            print(f"Missing: {path}")
            continue
        pr = load_y(path)
        gt_a, pr_a = align(gt, pr)
        diffs[name] = np.abs(gt_a - pr_a)

    # global vmax (so colorscales are comparable)
    vmax = max(d.max() for d in diffs.values())

    fig, axes = plt.subplots(1, len(diffs) + 1, figsize=(4 * (len(diffs) + 1), 4))

    axes[0].imshow(gt, cmap='gray')
    axes[0].set_title('GT', fontsize=13)
    axes[0].axis('off')

    for i, (name, d) in enumerate(diffs.items(), start=1):
        im = axes[i].imshow(d, cmap='magma', vmin=0, vmax=vmax)
        mae = d.mean()
        axes[i].set_title(f'|GT - {name}|  (MAE={mae:.4f})', fontsize=12)
        axes[i].axis('off')

    plt.tight_layout(pad=0.5, w_pad=0.3)
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
