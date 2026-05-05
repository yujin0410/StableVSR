"""Temporal profile (x-t plot) for VSR consistency visualization.

For each method, take a single horizontal row at y=row across all
frames, stack them vertically: x-axis is pixel column, y-axis is time.
Temporally consistent methods produce smooth horizontal bands;
flickering methods produce noisy vertical streaks.

Usage:
    python make_temporal_profile.py --seq 000 --row 360 \
        --x_start 400 --x_width 480 \
        --output figures/temporal_profile.pdf
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


def build_profile(seq, name, tmpl, num_frames, row, x_start, x_width):
    profile = np.zeros((num_frames, x_width, 3), dtype=np.float32)
    for t in range(num_frames):
        path = tmpl.format(seq=seq, idx=t)
        img = load_lr_upscaled(path) if name == 'LR' else load_image(path)
        # clamp row/x to image bounds
        H, W = img.shape[:2]
        r = min(row, H - 1)
        x1 = max(0, x_start)
        x2 = min(W, x1 + x_width)
        line = img[r, x1:x2]
        profile[t, :line.shape[0]] = line
    return profile


def make_figure(seq, num_frames, row, x_start, x_width, output_path,
                fontsize=14, dpi=300):
    methods = list(PATHS.keys())
    fig, axes = plt.subplots(1, len(methods),
                             figsize=(3 * len(methods), 4))

    for c, name in enumerate(methods):
        prof = build_profile(seq, name, PATHS[name], num_frames,
                             row, x_start, x_width)
        axes[c].imshow(prof, aspect='auto')
        axes[c].set_title(name, fontsize=fontsize)
        axes[c].set_xlabel('x', fontsize=fontsize - 2)
        if c == 0:
            axes[c].set_ylabel('frame t', fontsize=fontsize - 2)
        else:
            axes[c].set_yticks([])

    plt.tight_layout(pad=0.5, w_pad=0.3)
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
    p.add_argument('--num_frames', type=int, default=100,
                   help="number of frames to stack")
    p.add_argument('--row', type=int, default=360,
                   help="image row (y) to extract; default = center of 720")
    p.add_argument('--x_start', type=int, default=400)
    p.add_argument('--x_width', type=int, default=480)
    p.add_argument('--output', default='figures/temporal_profile.pdf')
    args = p.parse_args()

    make_figure(args.seq, args.num_frames, args.row,
                args.x_start, args.x_width, args.output)
