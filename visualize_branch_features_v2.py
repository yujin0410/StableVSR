"""Improved feature flow visualization showing multiple channels and attention map.

Layout:
  Row 1: 6 input channels heatmap
  Row 2: 6 channels after DWConv
  Row 3: 6 channels after SiLU
  Row 4: 6 channels after Spatial Attention + attention map
  Row 5: 6 channels after 1×1 Conv (different output channels)
  Bottom: Channel statistics (mean, std) per stage

Usage:
    python visualize_branch_features_v2.py \
        --input /path/to/image.png \
        --ckpt experiments/.../sft_adapter.bin \
        --branch magnitude --level 1 \
        --output figures/feature_flow_v2.pdf
"""

import os
import sys
import argparse
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from util.frequency_utils import DTCWTForward
from util.sft_utils import FrequencyConditioningEncoder


N_CHANNELS_TO_SHOW = 6   # show 6 channels per stage


def heat(t_2d):
    """Convert tensor to log-magnitude numpy for display."""
    a = np.log10(np.abs(t_2d.cpu().float().numpy()) + 1e-6)
    return a


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--branch', choices=['magnitude', 'phase'], default='magnitude')
    p.add_argument('--level', type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument('--output', default='figures/feature_flow_v2.pdf')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    print(f"[*] input: {args.input}")
    print(f"[*] ckpt:  {args.ckpt}")

    if not os.path.isfile(args.input):
        print(f"[!] Input not found: {args.input}")
        sys.exit(1)
    if not os.path.isfile(args.ckpt):
        print(f"[!] Ckpt not found: {args.ckpt}")
        sys.exit(1)

    device = args.device
    enc = FrequencyConditioningEncoder(
        in_channels=18, mid_channels=64,
        sft_out_channels_high=256, sft_out_channels_low=256,
    )
    enc.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    enc = enc.to(device).eval()
    proc = getattr(enc, f'proc_j{args.level}')

    # DT-CWT
    img = Image.open(args.input).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    dtcwt = DTCWTForward(J=4, biort='near_sym_a', qshift='qshift_a').to(device)
    with torch.no_grad():
        yl, yh = dtcwt(t)

    yh_j = yh[args.level - 1]
    B, C, D, H, W, _ = yh_j.shape
    real = yh_j[..., 0]
    imag = yh_j[..., 1]
    M = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
    sin_phi = imag / (M + 1e-8)
    cos_phi = real / (M + 1e-8)
    M_flat = M.reshape(B, C * D, H, W)
    sin_flat = sin_phi.reshape(B, C * D, H, W)
    cos_flat = cos_phi.reshape(B, C * D, H, W)

    if args.branch == 'magnitude':
        x_in = M_flat
        dw, sa, pw = proc.M_dw, proc.M_sa, proc.M_pw
        title = 'Magnitude branch'
    else:
        x_in = torch.cat([sin_flat, cos_flat], dim=1)
        dw, sa, pw = proc.phase_dw, proc.phase_sa, proc.phase_pw
        title = 'Phase branch'

    with torch.no_grad():
        x0 = x_in
        x1 = dw(x0)
        x2 = F.silu(x1)
        avg = x2.mean(dim=1, keepdim=True)
        max_p = x2.max(dim=1, keepdim=True)[0]
        attn_input = torch.cat([avg, max_p], dim=1)
        attn_map = torch.sigmoid(sa.conv(attn_input))
        x3 = x2 * attn_map
        x4 = pw(x3)

    n_in = x0.shape[1]
    n_out = x4.shape[1]
    show_n = min(N_CHANNELS_TO_SHOW, n_in)
    # spread channels evenly across all
    in_indices = np.linspace(0, n_in - 1, show_n, dtype=int)
    out_indices = np.linspace(0, n_out - 1, show_n, dtype=int)

    print(f"[*] Showing {show_n} channels at each stage")
    print(f"[*] Input channels: {in_indices.tolist()}")
    print(f"[*] Output channels (1×1 Conv): {out_indices.tolist()}")

    # Stage info
    stages = [
        ('Input',        x0, in_indices),
        ('DWConv 3×3',   x1, in_indices),
        ('SiLU',         x2, in_indices),
        ('SA (attn applied)', x3, in_indices),
        ('1×1 Conv',     x4, out_indices),
    ]
    n_rows = len(stages)
    n_cols = show_n + 1   # +1 for attention map column on row 4 / stats

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.2 * n_rows))

    for r, (name, feat, indices) in enumerate(stages):
        for c, ch in enumerate(indices):
            ax = axes[r, c]
            im = heat(feat[0, ch])
            vmin, vmax = np.percentile(im, [5, 99])
            ax.imshow(im, cmap='magma', vmin=vmin, vmax=vmax)
            ax.axis('off')
            if r == 0:
                ax.set_title(f'ch {ch}', fontsize=9)
            if c == 0:
                # Stage label on left
                stats = feat[0].cpu().float().numpy()
                ax.text(-0.15, 0.5,
                        f'{name}\n[{feat.shape[1]} ch]\n'
                        f'μ={stats.mean():.2f}\n'
                        f'σ={stats.std():.2f}',
                        transform=ax.transAxes,
                        ha='right', va='center', fontsize=9,
                        fontweight='bold' if r in (0, 4) else 'normal')

        # Last column for SA row: attention map
        ax_last = axes[r, -1]
        if r == 3:
            attn_vis = attn_map[0, 0].cpu().float().numpy()
            ax_last.imshow(attn_vis, cmap='hot', vmin=0, vmax=1)
            ax_last.set_title('attention map\n[1 ch, range 0~1]', fontsize=9)
            ax_last.axis('off')
        else:
            ax_last.axis('off')

    plt.suptitle(f'{title} — level j={args.level} — multi-channel feature flow',
                 fontsize=13, y=0.995)
    plt.tight_layout()

    out_dir = os.path.dirname(args.output) or '.'
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    png = os.path.splitext(args.output)[0] + '.png'
    plt.savefig(png, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {args.output}")
    print(f"[+] Saved: {png}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n[!] ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
