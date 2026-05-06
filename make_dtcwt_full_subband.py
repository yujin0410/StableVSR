"""Visualize all 24 DT-CWT subbands (4 levels x 6 directions) for one frame.

Shows magnitude of each directional subband to inspect the spatial
content captured at each frequency level. This helps verify the
HIGH (j=1,2) vs LOW (j=3,4) band partitioning visually.

Usage:
    python make_dtcwt_full_subband.py \
        --input /mnt/HDD_raid1/yjcho/data/REDS/test/gt/000/00000050.png \
        --output figures/dtcwt_full_subband.pdf
"""

import os
import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from util.frequency_utils import DTCWTForward


# DT-CWT 6 directional orientations (approximate)
DIR_LABELS = ['+15°', '-15°', '+45°', '-45°', '+75°', '-75°']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True,
                   help="Path to one image (HR or LR; will be visualized as-is)")
    p.add_argument('--output', default='figures/dtcwt_full_subband.pdf')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    # --- Load image ---
    img = Image.open(args.input).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(args.device)

    # --- DT-CWT forward ---
    dtcwt = DTCWTForward(J=4, biort='near_sym_a', qshift='qshift_a').to(args.device)
    dtcwt.requires_grad_(False)
    with torch.no_grad():
        yl, yh = dtcwt(t)
    # yh: list of 4, each [B=1, C=3, D=6, H_j, W_j, 2]
    # yl: [B=1, C=3, H/16, W/16] (real)

    # --- Layout: 6 rows x 7 cols
    #     row 0: input image + lowpass
    #     row 1: direction column headers
    #     rows 2..5: j=1..4 subbands (4 levels)
    fig = plt.figure(figsize=(20, 13))
    # Top: input image (small, top-left)
    ax_img = plt.subplot2grid((6, 7), (0, 0), colspan=2, rowspan=1)
    ax_img.imshow(arr)
    ax_img.set_title('Input Image', fontsize=12)
    ax_img.axis('off')

    # Lowpass on top-right
    ax_lp = plt.subplot2grid((6, 7), (0, 5), colspan=2, rowspan=1)
    lp_vis = yl[0].mean(dim=0).cpu().numpy()  # avg over 3 colors
    ax_lp.imshow(lp_vis, cmap='gray')
    ax_lp.set_title(f'Lowpass (LL, j=4 size, real-valued)', fontsize=11)
    ax_lp.axis('off')

    # Direction column headers (row 1)
    for d in range(6):
        ax_h = plt.subplot2grid((6, 7), (1, d), rowspan=1)
        ax_h.text(0.5, 0.3, f'd={d+1}\n{DIR_LABELS[d]}',
                  ha='center', va='center', fontsize=12, fontweight='bold',
                  transform=ax_h.transAxes)
        ax_h.axis('off')
    ax_band = plt.subplot2grid((6, 7), (1, 6), rowspan=1)
    ax_band.text(0.5, 0.5, 'Band', ha='center', va='center',
                 fontsize=12, fontweight='bold', transform=ax_band.transAxes)
    ax_band.axis('off')

    # 4 levels x 6 directions
    bands = ['HIGH', 'HIGH', 'LOW', 'LOW']
    band_colors = ['#3B82F6', '#3B82F6', '#F97316', '#F97316']

    for j in range(4):
        for d in range(6):
            ax = plt.subplot2grid((6, 7), (j + 2, d), rowspan=1)
            sub = yh[j][0, :, d]   # [3, H_j, W_j, 2]
            mag = torch.sqrt(sub[..., 0] ** 2 + sub[..., 1] ** 2 + 1e-8)  # [3, H, W]
            mag = mag.mean(dim=0).cpu().numpy()  # avg over 3 colors → [H, W]
            # log magnitude for better dynamic range
            mag_log = np.log10(mag + 1e-6)
            vmin, vmax = np.percentile(mag_log, [5, 99])
            ax.imshow(mag_log, cmap='magma', vmin=vmin, vmax=vmax)
            ax.axis('off')
            if d == 0:
                H_j = mag_log.shape[0]
                W_j = mag_log.shape[1]
                ax.text(-0.15, 0.5,
                        f'j={j+1}\n{H_j}×{W_j}',
                        ha='right', va='center',
                        fontsize=11, fontweight='bold',
                        color=band_colors[j],
                        transform=ax.transAxes)

        # Band label on the right
        ax_b = plt.subplot2grid((6, 7), (j + 2, 6), rowspan=1)
        ax_b.text(0.05, 0.5, bands[j],
                  ha='left', va='center',
                  fontsize=14, fontweight='bold',
                  color=band_colors[j],
                  transform=ax_b.transAxes)
        ax_b.axis('off')

    plt.suptitle(
        'DT-CWT (J=4) directional subbands — magnitude (log-scale, color-averaged)\n'
        'HIGH band = {j=1, 2} (textural details);  LOW band = {j=3, 4} (structural content)',
        fontsize=13, y=0.99)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
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
