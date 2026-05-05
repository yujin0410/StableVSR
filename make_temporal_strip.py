"""Temporal sequential strip: 5 methods x N consecutive frames.

Each row is one method, each column is a consecutive frame's zoom patch.
Designed to show temporal consistency / flicker.

Usage:
    python make_temporal_strip.py --seq 000 --start 50 --num 5 \
        --crop 600 350 100 100 --output figures/temporal_strip.pdf
"""

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


PATHS = {
    'LR':         '/mnt/HDD_raid1/yjcho/data/REDS/test/bicubic/{seq}/{idx:08d}.png',
    'BasicVSR++': '/mnt/HDD_raid1/yjcho/BasicVSR_PlusPlus/reds/{seq}/{idx:08d}.png',
    'StableVSR':  '/mnt/HDD_raid1/yjcho/stablevsr_reds/{seq}/{idx:08d}.png',
    'Ours':       '/mnt/HDD_raid1/yjcho/20260430/reds/{seq}/{idx:08d}.png',
    'GT':         '/mnt/HDD_raid1/yjcho/data/REDS/test/gt/{seq}/{idx:08d}.png',
}


def load_image(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return np.array(Image.open(path).convert('RGB')) / 255.0


def load_lr_upscaled(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    lr = Image.open(path).convert('RGB')
    lr = lr.resize((lr.width * 4, lr.height * 4), Image.BICUBIC)
    return np.array(lr) / 255.0


def make_strip(seq, start, num, crop_xywh, output_path,
               fontsize=14, dpi=300):
    x, y, w, h = crop_xywh
    methods = list(PATHS.keys())
    fig, axes = plt.subplots(len(methods), num,
                             figsize=(2.5 * num, 2.5 * len(methods)))

    for r, name in enumerate(methods):
        for c in range(num):
            idx = start + c
            path = PATHS[name].format(seq=seq, idx=idx)
            img = load_lr_upscaled(path) if name == 'LR' else load_image(path)
            H, W = img.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W, x + w), min(H, y + h)
            patch = img[y1:y2, x1:x2]
            axes[r, c].imshow(patch)
            axes[r, c].axis('off')
            if r == 0:
                axes[r, c].set_title(f"t={idx}", fontsize=fontsize)
            if c == 0:
                axes[r, c].text(-0.1, 0.5, name, transform=axes[r, c].transAxes,
                                fontsize=fontsize, va='center', ha='right',
                                rotation=0, fontweight='bold')

    plt.tight_layout(pad=0.5, h_pad=0.2, w_pad=0.2)
    out_dir = os.path.dirname(output_path) or '.'
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    png_path = os.path.splitext(output_path)[0] + '.png'
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    print(f"Saved: {png_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seq', default='000')
    p.add_argument('--start', type=int, default=50,
                   help="starting frame index")
    p.add_argument('--num', type=int, default=5,
                   help="number of consecutive frames")
    p.add_argument('--crop', type=int, nargs=4, default=[600, 350, 100, 100],
                   metavar=('X', 'Y', 'W', 'H'))
    p.add_argument('--output', default='figures/temporal_strip.pdf')
    args = p.parse_args()

    make_strip(args.seq, args.start, args.num, tuple(args.crop), args.output)
