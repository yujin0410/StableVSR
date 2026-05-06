"""Simplified version for visualizing intermediate features.

Usage:
    python visualize_branch_features_simple.py \
        --input /path/to/image.png \
        --ckpt experiments/.../sft_adapter.bin \
        --branch magnitude --level 1 \
        --output figures/feature_flow.pdf
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--branch', choices=['magnitude', 'phase'], default='magnitude')
    p.add_argument('--level', type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument('--channel', type=int, default=0)
    p.add_argument('--output', default='figures/feature_flow.pdf')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    print(f"[*] input: {args.input}")
    print(f"[*] ckpt:  {args.ckpt}")
    print(f"[*] branch: {args.branch}, level: {args.level}")

    # Sanity checks
    if not os.path.isfile(args.input):
        print(f"[!] Input file not found: {args.input}")
        sys.exit(1)
    if not os.path.isfile(args.ckpt):
        print(f"[!] Checkpoint not found: {args.ckpt}")
        sys.exit(1)

    device = args.device

    # Build and load encoder
    print("[*] Building encoder...")
    enc = FrequencyConditioningEncoder(
        in_channels=18, mid_channels=64,
        sft_out_channels_high=256, sft_out_channels_low=256,
    )
    print("[*] Loading state_dict...")
    sd = torch.load(args.ckpt, map_location='cpu')
    try:
        enc.load_state_dict(sd)
    except Exception as e:
        print(f"[!] state_dict load error: {e}")
        sys.exit(1)
    enc = enc.to(device).eval()
    print(f"[*] Encoder loaded.")

    proc = getattr(enc, f'proc_j{args.level}')

    # DT-CWT
    print("[*] Computing DT-CWT...")
    img = Image.open(args.input).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    dtcwt = DTCWTForward(J=4, biort='near_sym_a', qshift='qshift_a').to(device)
    with torch.no_grad():
        yl, yh = dtcwt(t)

    yh_j = yh[args.level - 1]
    B, C, D, H, W, _ = yh_j.shape
    print(f"[*] yh[{args.level - 1}] shape: {tuple(yh_j.shape)}")

    real = yh_j[..., 0]
    imag = yh_j[..., 1]
    M = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
    sin_phi = imag / (M + 1e-8)
    cos_phi = real / (M + 1e-8)

    M_flat = M.reshape(B, C * D, H, W)
    sin_flat = sin_phi.reshape(B, C * D, H, W)
    cos_flat = cos_phi.reshape(B, C * D, H, W)

    # Run through branch
    if args.branch == 'magnitude':
        x_in = M_flat
        dw, sa, pw = proc.M_dw, proc.M_sa, proc.M_pw
        title = 'Magnitude branch'
    else:
        x_in = torch.cat([sin_flat, cos_flat], dim=1)
        dw, sa, pw = proc.phase_dw, proc.phase_sa, proc.phase_pw
        title = 'Phase branch'

    print(f"[*] Running {args.branch} branch forward pass...")
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

    print(f"[*] Stage shapes:")
    for nm, t in [('x0 input', x0), ('x1 dwconv', x1), ('x2 silu', x2),
                  ('x3 sa-out', x3), ('x4 pwconv', x4), ('attn', attn_map)]:
        print(f"    {nm:<12} {tuple(t.shape)}")

    # Pick channel
    ch = min(args.channel, x0.shape[1] - 1)

    def heat(t_2d):
        a = np.log10(np.abs(t_2d.cpu().float().numpy()) + 1e-6)
        return a

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    stages = [
        (axes[0], 'Input',         heat(x0[0, ch]),  f'{x0.shape[1]} ch'),
        (axes[1], 'After DWConv',  heat(x1[0, ch]),  f'{x1.shape[1]} ch'),
        (axes[2], 'After SiLU',    heat(x2[0, ch]),  f'{x2.shape[1]} ch'),
        (axes[3], 'After SA',      heat(x3[0, ch]),  f'{x3.shape[1]} ch'),
        (axes[4], 'After 1×1Conv', heat(x4[0, 0]),   f'{x4.shape[1]} ch (out_ch=0)'),
    ]
    for ax, name, im, ch_str in stages:
        vmin, vmax = np.percentile(im, [5, 99])
        ax.imshow(im, cmap='magma', vmin=vmin, vmax=vmax)
        ax.set_title(f'{name}\n{ch_str}', fontsize=10)
        ax.axis('off')

    plt.suptitle(f'{title} — level j={args.level}', fontsize=12, y=1.02)
    plt.tight_layout()

    out_dir = os.path.dirname(args.output) or '.'
    os.makedirs(out_dir, exist_ok=True)
    print(f"[*] Saving to {args.output} ...")
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
