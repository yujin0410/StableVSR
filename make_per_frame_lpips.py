"""Per-frame LPIPS over a REDS4 sequence (line plot).

Computes LPIPS at every frame for each method against GT, plots all
methods on one graph. Lower curves and smaller temporal variation are
better.

Usage:
    python make_per_frame_lpips.py --seq 000 \
        --output figures/per_frame_lpips_000.pdf
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, CenterCrop
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
import matplotlib.pyplot as plt


PATHS = {
    'BasicVSR++': '/mnt/HDD_raid1/yjcho/BasicVSR_PlusPlus/reds/{seq}/{idx:08d}.png',
    'StableVSR':  '/mnt/HDD_raid1/yjcho/stablevsr_reds/{seq}/{idx:08d}.png',
    'Ours':       '/mnt/HDD_raid1/yjcho/20260430/reds/{seq}/{idx:08d}.png',
}
GT_TMPL = '/mnt/HDD_raid1/yjcho/data/REDS/test/gt/{seq}/{idx:08d}.png'

COLORS = {'BasicVSR++': '#1f77b4', 'StableVSR': '#2ca02c', 'Ours': '#d62728'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seq', default='000')
    p.add_argument('--num_frames', type=int, default=100)
    p.add_argument('--output', default='figures/per_frame_lpips.pdf')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    device = args.device
    lpips = LPIPS(normalize=True).to(device)
    tt = ToTensor()

    series = {m: [] for m in PATHS}
    for t in range(args.num_frames):
        gt_path = GT_TMPL.format(seq=args.seq, idx=t)
        if not os.path.isfile(gt_path):
            continue
        gt = tt(Image.open(gt_path).convert('RGB')).unsqueeze(0).to(device)

        for m, tmpl in PATHS.items():
            pred_path = tmpl.format(seq=args.seq, idx=t)
            if not os.path.isfile(pred_path):
                series[m].append(np.nan)
                continue
            pr = tt(Image.open(pred_path).convert('RGB')).unsqueeze(0).to(device)
            if gt.shape[-2:] != pr.shape[-2:]:
                h = min(gt.shape[-2], pr.shape[-2])
                w = min(gt.shape[-1], pr.shape[-1])
                gt_c = CenterCrop((h, w))(gt)
                pr = CenterCrop((h, w))(pr)
            else:
                gt_c = gt
            with torch.no_grad():
                series[m].append(lpips(gt_c, pr).item())

    # plot
    plt.figure(figsize=(10, 5))
    for m, vals in series.items():
        v = np.array(vals)
        plt.plot(np.arange(len(v)), v, label=f'{m} (mean={np.nanmean(v):.3f})',
                 color=COLORS[m], linewidth=2)
    plt.xlabel('Frame index', fontsize=14)
    plt.ylabel('LPIPS (lower = better)', fontsize=14)
    plt.title(f'Per-frame LPIPS on REDS4 sequence {args.seq}', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.4)

    out_dir = os.path.dirname(args.output) or '.'
    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    png_path = os.path.splitext(args.output)[0] + '.png'
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {args.output}")
    print(f"Saved: {png_path}")


if __name__ == '__main__':
    main()
