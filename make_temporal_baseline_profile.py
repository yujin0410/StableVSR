"""Temporal x-t profile: GT vs StableVSR baseline vs Ours.

For one row of pixels, stack across all frames vertically:
- x-axis: pixel column
- y-axis: time (frame index)
A temporally consistent method produces smooth horizontal bands; a
flickering method produces vertical streaks.

Usage:
    python make_temporal_baseline_profile.py --seq 000 \
        --row 360 --x_start 400 --x_width 480 \
        --output figures/temporal_profile_baseline.pdf
"""

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


PATHS = {
    'GT':                   '/mnt/HDD_raid1/yjcho/data/REDS/test/gt/{seq}/{idx:08d}.png',
    'StableVSR (baseline)': '/mnt/HDD_raid1/yjcho/stablevsr_reds/{seq}/{idx:08d}.png',
    'Ours':                 '/mnt/HDD_raid1/yjcho/20260430/reds/{seq}/{idx:08d}.png',
}


def load_image(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return np.array(Image.open(path).convert('RGB')) / 255.0


def build_profile(seq, tmpl, num_frames, row, x_start, x_width):
    profile = np.zeros((num_frames, x_width, 3), dtype=np.float32)
    for t in range(num_frames):
        path = tmpl.format(seq=seq, idx=t)
        img = load_image(path)
        H, W = img.shape[:2]
        r = min(row, H - 1)
        x1 = max(0, x_start)
        x2 = min(W, x1 + x_width)
        line = img[r, x1:x2]
        profile[t, :line.shape[0]] = line
    return profile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seq', default='000')
    p.add_argument('--num_frames', type=int, default=100)
    p.add_argument('--row', type=int, default=360)
    p.add_argument('--x_start', type=int, default=400)
    p.add_argument('--x_width', type=int, default=480)
    p.add_argument('--output', default='figures/temporal_profile_baseline.pdf')
    args = p.parse_args()

    methods = list(PATHS.keys())
    fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 5))

    for c, name in enumerate(methods):
        prof = build_profile(args.seq, PATHS[name], args.num_frames,
                             args.row, args.x_start, args.x_width)
        axes[c].imshow(prof, aspect='auto')
        axes[c].set_title(name, fontsize=14)
        axes[c].set_xlabel('x (pixel column)', fontsize=12)
        if c == 0:
            axes[c].set_ylabel('frame index t', fontsize=12)
        else:
            axes[c].set_yticks([])

    plt.suptitle(
        f'Temporal x-t profile (REDS4 seq {args.seq}, row y={args.row})\n'
        'smooth horizontal bands = consistent; vertical streaks = flicker',
        fontsize=12, y=1.02)
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
