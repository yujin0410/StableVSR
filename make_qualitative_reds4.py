"""Qualitative comparison figure for VSR results.

Layout: 2 rows x 5 columns
  Top row:    full frame with red crop rectangle
  Bottom row: zoomed-in crop patch

Usage:
    python make_qualitative_reds4.py \
        --seq 000 --frame 50 \
        --crop 600 350 120 120 \
        --output figures/qualitative_reds4.pdf
"""

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


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


def make_figure(seq, frame_idx, crop_xywh, output_path,
                box_lw=3, fontsize=16, dpi=300, png_only=False):
    x, y, w, h = crop_xywh
    n = len(PATHS)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 5))

    for col, (name, tmpl) in enumerate(PATHS.items()):
        path = tmpl.format(seq=seq, idx=frame_idx)
        img = load_lr_upscaled(path) if name == 'LR' else load_image(path)

        # full image with rectangle
        axes[0, col].imshow(img)
        axes[0, col].add_patch(Rectangle(
            (x, y), w, h, linewidth=box_lw, edgecolor='red', facecolor='none'
        ))
        axes[0, col].set_title(name, fontsize=fontsize)
        axes[0, col].axis('off')

        # zoom
        H, W = img.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        patch = img[y1:y2, x1:x2]
        axes[1, col].imshow(patch)
        axes[1, col].axis('off')

    plt.tight_layout(pad=0.5, h_pad=0.3, w_pad=0.3)
    out_dir = os.path.dirname(output_path) or '.'
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.splitext(output_path)[0] + '.png'
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {png_path}")
    if not png_only:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seq', default='000', help="REDS4 sequence (000/011/015/020)")
    p.add_argument('--frame', type=int, default=50, help="frame index (0-99)")
    p.add_argument('--crop', type=int, nargs=4, default=[600, 350, 120, 120],
                   metavar=('X', 'Y', 'W', 'H'),
                   help="crop region in HR (1280x720) coordinates")
    p.add_argument('--output', default='figures/qualitative_reds4.pdf')
    p.add_argument('--png_only', action='store_true',
                   help="skip PDF (much faster for batch generation)")
    args = p.parse_args()

    make_figure(
        seq=args.seq,
        frame_idx=args.frame,
        crop_xywh=tuple(args.crop),
        output_path=args.output,
        png_only=args.png_only,
    )
