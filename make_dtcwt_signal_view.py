"""1D signal-style visualization of DT-CWT subband magnitudes.

Three views in one figure:
  1. Total energy per subband (bar chart, 24 bars = 4 levels x 6 dirs)
  2. Histogram of magnitude distribution per level
  3. 1D cross-section: middle horizontal row of each level (avg over directions)

Usage:
    python make_dtcwt_signal_view.py \
        --input /mnt/HDD_raid1/yjcho/data/REDS/test/gt/000/00000024.png \
        --output figures/dtcwt_signal_view.pdf
"""

import os
import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from util.frequency_utils import DTCWTForward


DIR_LABELS = ['+15°', '-15°', '+45°', '-45°', '+75°', '-75°']
LEVEL_COLORS = ['#1E40AF', '#3B82F6', '#F97316', '#EA580C']  # blue, blue, orange, orange
BAND_LABELS = ['HIGH', 'HIGH', 'LOW', 'LOW']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', default='figures/dtcwt_signal_view.pdf')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    img = Image.open(args.input).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(args.device)

    dtcwt = DTCWTForward(J=4, biort='near_sym_a', qshift='qshift_a').to(args.device)
    dtcwt.requires_grad_(False)
    with torch.no_grad():
        yl, yh = dtcwt(t)

    # Compute magnitudes per (level, direction): [3, H_j, W_j]
    mags = []  # [j][d] = (H_j, W_j) avg over color
    for j in range(4):
        per_dir = []
        for d in range(6):
            sub = yh[j][0, :, d]  # [3, H, W, 2]
            mag = torch.sqrt(sub[..., 0] ** 2 + sub[..., 1] ** 2 + 1e-8).mean(dim=0)
            per_dir.append(mag.cpu().numpy())
        mags.append(per_dir)

    # === Layout: 3 panels (vertical) ===
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # ----- Panel 1: Bar chart of total energy per subband -----
    ax1 = axes[0]
    energies = []
    labels = []
    colors = []
    for j in range(4):
        for d in range(6):
            E = (mags[j][d] ** 2).sum()
            energies.append(E)
            labels.append(f'j={j+1}\n{DIR_LABELS[d]}')
            colors.append(LEVEL_COLORS[j])
    x = np.arange(len(energies))
    ax1.bar(x, energies, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel('Total Energy (Σ M²)', fontsize=11)
    ax1.set_title('(a) Total energy per subband (4 levels × 6 directions)', fontsize=12)
    ax1.set_yscale('log')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.4)
    # Band group annotations
    for j in range(4):
        x_mid = j * 6 + 2.5
        ax1.text(x_mid, ax1.get_ylim()[1] * 0.7,
                 f'{BAND_LABELS[j]} (j={j+1})',
                 ha='center', fontsize=10, fontweight='bold',
                 color=LEVEL_COLORS[j])

    # ----- Panel 2: Histogram of magnitudes per level -----
    ax2 = axes[1]
    for j in range(4):
        # Pool all 6 directions of this level
        all_mag = np.concatenate([mags[j][d].flatten() for d in range(6)])
        # log scale histogram
        log_mag = np.log10(all_mag + 1e-6)
        ax2.hist(log_mag, bins=60, alpha=0.5, label=f'j={j+1} ({BAND_LABELS[j]})',
                 color=LEVEL_COLORS[j], density=True)
    ax2.set_xlabel('log₁₀(Magnitude)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('(b) Magnitude distribution per level', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.4)

    # ----- Panel 3: 1D cross-section (middle row of each level) -----
    ax3 = axes[2]
    for j in range(4):
        # Average magnitude over 6 directions, take middle row
        mag_avg = np.mean(mags[j], axis=0)  # [H, W]
        mid_row = mag_avg[mag_avg.shape[0] // 2, :]
        # normalize x to [0, 1] for fair comparison across levels
        x_norm = np.linspace(0, 1, len(mid_row))
        ax3.plot(x_norm, mid_row, color=LEVEL_COLORS[j], linewidth=1.5,
                 label=f'j={j+1} ({BAND_LABELS[j]})')
    ax3.set_xlabel('Normalized horizontal position (0=left, 1=right)', fontsize=11)
    ax3.set_ylabel('Magnitude (avg over directions)', fontsize=11)
    ax3.set_title('(c) 1D cross-section: middle horizontal row of each level',
                  fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.4)

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
