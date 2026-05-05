"""DT-CWT subband magnitude visualization.

For one chosen frame, decompose each method's output via 4-level DT-CWT
(matching the dual-SFT design), and visualize the magnitude maps of
each level. Layout: 5 methods (rows) x 4 levels (columns).

Usage:
    python make_dtcwt_subband.py --seq 000 --frame 50 \
        --output figures/dtcwt_subband.pdf
"""

import os
import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from util.frequency_utils import DTCWTForward


PATHS = {
    'LR (bicubic up)': '/mnt/HDD_raid1/yjcho/data/REDS/test/bicubic/{seq}/{idx:08d}.png',
    'BasicVSR++':      '/mnt/HDD_raid1/yjcho/BasicVSR_PlusPlus/reds/{seq}/{idx:08d}.png',
    'StableVSR':       '/mnt/HDD_raid1/yjcho/stablevsr_reds/{seq}/{idx:08d}.png',
    'Ours':            '/mnt/HDD_raid1/yjcho/20260430/reds/{seq}/{idx:08d}.png',
    'GT':              '/mnt/HDD_raid1/yjcho/data/REDS/test/gt/{seq}/{idx:08d}.png',
}


def load_tensor(path, upscale4=False, device='cpu'):
    img = Image.open(path).convert('RGB')
    if upscale4:
        img = img.resize((img.width * 4, img.height * 4), Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0
    # [B, C, H, W]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seq', default='000')
    p.add_argument('--frame', type=int, default=50)
    p.add_argument('--output', default='figures/dtcwt_subband.pdf')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    device = args.device
    dtcwt = DTCWTForward(J=4, biort='near_sym_a', qshift='qshift_a').to(device)
    dtcwt.requires_grad_(False)

    methods = list(PATHS.keys())
    n = len(methods)
    levels = 4  # j=1,2,3,4 (HIGH = 1,2; LOW = 3,4)

    fig, axes = plt.subplots(n, levels, figsize=(3 * levels, 2.7 * n))

    for r, name in enumerate(methods):
        path = PATHS[name].format(seq=args.seq, idx=args.frame)
        if not os.path.isfile(path):
            print(f"Missing: {path}")
            for c in range(levels):
                axes[r, c].axis('off')
            continue
        upscale4 = (name == 'LR (bicubic up)')
        t = load_tensor(path, upscale4=upscale4, device=device)
        with torch.no_grad():
            yl, yh = dtcwt(t)
        # yh: list of 4 tensors, each [B, C, 6, H_j, W_j, 2]
        # magnitude: sqrt(real^2 + imag^2), average across direction (D=6) and color (C=3)
        for j in range(levels):
            mag = torch.sqrt(yh[j][..., 0] ** 2 + yh[j][..., 1] ** 2 + 1e-8)
            # mag: [B, C, D, H_j, W_j] -> mean over C and D
            mag_2d = mag.mean(dim=(1, 2))[0].cpu().numpy()  # [H_j, W_j]
            # log scale + normalize for display
            mag_log = np.log10(mag_2d + 1e-6)
            vmin, vmax = np.percentile(mag_log, [5, 99])
            axes[r, j].imshow(mag_log, cmap='magma', vmin=vmin, vmax=vmax)
            axes[r, j].axis('off')
            band = 'HIGH' if j < 2 else 'LOW'
            if r == 0:
                axes[r, j].set_title(f'j={j+1} ({band})', fontsize=12)
            if j == 0:
                axes[r, j].text(-0.08, 0.5, name, transform=axes[r, j].transAxes,
                                fontsize=12, va='center', ha='right',
                                fontweight='bold')

    plt.tight_layout(pad=0.5, h_pad=0.2, w_pad=0.2)
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
