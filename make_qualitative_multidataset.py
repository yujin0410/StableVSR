"""Qualitative comparison figure for VSR results across multiple datasets.

Layout: 2 rows x 5 columns
  Top row:    full frame with red crop rectangle
  Bottom row: zoomed-in crop patch
Methods (left to right): LR, BasicVSR++, StableVSR, Ours, GT

Usage:
    python make_qualitative_multidataset.py --dataset vid4 --seq calendar \
        --frame 20 --crop 600 350 120 120 \
        --output figures/qual_vid4_calendar.pdf
"""

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# Per-dataset path templates and frame naming
DATASETS = {
    'reds': {
        'LR':         '/mnt/HDD_raid1/yjcho/data/REDS/test/bicubic/{seq}/{idx:08d}.png',
        'BasicVSR++': '/mnt/HDD_raid1/yjcho/BasicVSR_PlusPlus/reds/{seq}/{idx:08d}.png',
        'StableVSR':  '/mnt/HDD_raid1/yjcho/stablevsr_reds/{seq}/{idx:08d}.png',
        'Ours':       '/mnt/HDD_raid1/yjcho/20260430/reds/{seq}/{idx:08d}.png',
        'GT':         '/mnt/HDD_raid1/yjcho/data/REDS/test/gt/{seq}/{idx:08d}.png',
    },
    'vid4': {
        'LR':         '/mnt/HDD_raid1/yjcho/data/Vid4/BIx4/{seq}/{idx:08d}.png',
        'BasicVSR++': '/mnt/HDD_raid1/yjcho/BasicVSR_PlusPlus/vid4/{seq}/{idx:08d}.png',
        'StableVSR':  '/mnt/HDD_raid1/yjcho/stablevsr_vid4/{seq}/{idx:08d}.png',
        'Ours':       '/mnt/HDD_raid1/yjcho/20260430/vid4/{seq}/{idx:08d}.png',
        'GT':         '/mnt/HDD_raid1/yjcho/data/Vid4/GT/{seq}/{idx:08d}.png',
    },
    'udm10': {
        'LR':         '/mnt/HDD_raid1/yjcho/data/UDM10/BIx4/{seq}/{idx:04d}.png',
        'BasicVSR++': '/mnt/HDD_raid1/yjcho/BasicVSR_PlusPlus/udm10/{seq}/{idx:04d}.png',
        'StableVSR':  '/mnt/HDD_raid1/yjcho/stablevsr_udm10/{seq}/{idx:04d}.png',
        'Ours':       '/mnt/HDD_raid1/yjcho/20260430/udm10/{seq}/{idx:04d}.png',
        'GT':         '/mnt/HDD_raid1/yjcho/data/UDM10/GT/{seq}/{idx:04d}.png',
    },
    'spmcs': {
        # SPMCS uses 4-digit frame naming and per-sequence variable starting index
        'LR':         '/mnt/HDD_raid1/yjcho/data/SPMCS_BIx4/{seq}/{idx:04d}.png',
        'BasicVSR++': '/mnt/HDD_raid1/yjcho/BasicVSR_PlusPlus/spmcs/{seq}/{idx:04d}.png',
        'StableVSR':  '/mnt/HDD_raid1/yjcho/stablevsr_spmc/{seq}/{idx:04d}.png',
        'Ours':       '/mnt/HDD_raid1/yjcho/20260430/spmcs/{seq}/{idx:04d}.png',
        'GT':         '/mnt/HDD_raid1/yjcho/data/SPMCS_GT/{seq}/{idx:04d}.png',
    },
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


def make_figure(dataset, seq, frame_idx, crop_xywh, output_path,
                box_lw=3, fontsize=16, dpi=300, png_only=False):
    paths = DATASETS[dataset]
    x, y, w, h = crop_xywh
    n = len(paths)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 5))

    for col, (name, tmpl) in enumerate(paths.items()):
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
    p.add_argument('--dataset', required=True,
                   choices=list(DATASETS.keys()),
                   help="Dataset to visualize")
    p.add_argument('--seq', required=True,
                   help="Sequence name (e.g. 'calendar' for vid4, '000' for reds)")
    p.add_argument('--frame', type=int, default=10,
                   help="Frame index (use the actual frame number)")
    p.add_argument('--crop', type=int, nargs=4, default=[600, 350, 120, 120],
                   metavar=('X', 'Y', 'W', 'H'),
                   help="crop region in HR coordinates")
    p.add_argument('--output', default=None,
                   help="output path; default = figures/qual_{dataset}_{seq}.pdf")
    p.add_argument('--png_only', action='store_true')
    args = p.parse_args()

    if args.output is None:
        args.output = f'figures/qual_{args.dataset}_{args.seq}.pdf'

    make_figure(
        dataset=args.dataset,
        seq=args.seq,
        frame_idx=args.frame,
        crop_xywh=tuple(args.crop),
        output_path=args.output,
        png_only=args.png_only,
    )
