"""Visualize intermediate features through magnitude/phase branches.

Loads the trained model, passes a real DT-CWT subband through
PerLevelProcessor, and shows feature maps at each stage:
  Input → DWConv → SiLU → SpatialAttention → 1×1 Conv

This helps reviewers see how data transforms through the architecture
beyond a static block diagram.

Usage:
    python visualize_branch_features.py \
        --input /mnt/HDD_raid1/yjcho/data/REDS/test/gt/000/00000024.png \
        --ckpt experiments/20260430_dualsft/checkpoint-20000/sft_adapter.bin \
        --branch magnitude \
        --level 1 \
        --output figures/feature_flow_mag_j1.pdf
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from util.frequency_utils import DTCWTForward
from util.sft_utils import FrequencyConditioningEncoder, PerLevelProcessor


def to_heatmap(t):
    """Convert a [H, W] tensor to log-magnitude numpy for display."""
    arr = t.detach().cpu().float().numpy()
    arr = np.abs(arr)
    return np.log10(arr + 1e-6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--ckpt', required=True,
                   help="Path to sft_adapter.bin")
    p.add_argument('--branch', choices=['magnitude', 'phase'],
                   default='magnitude')
    p.add_argument('--level', type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument('--channel', type=int, default=0,
                   help="Which channel to visualize (0~17 for mag, 0~35 for phase)")
    p.add_argument('--output', default='figures/feature_flow.pdf')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    device = args.device

    # Load encoder
    enc = FrequencyConditioningEncoder(
        in_channels=18, mid_channels=64,
        sft_out_channels_high=256, sft_out_channels_low=256,
    )
    enc.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    enc = enc.to(device)
    enc.eval()

    # Get the level's processor
    proc: PerLevelProcessor = getattr(enc, f'proc_j{args.level}')

    # Load image and DT-CWT
    img = Image.open(args.input).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    dtcwt = DTCWTForward(J=4, biort='near_sym_a', qshift='qshift_a').to(device)
    with torch.no_grad():
        yl, yh = dtcwt(t)

    # Pick the j-th level
    yh_j = yh[args.level - 1]   # [B, 3, 6, H_j, W_j, 2]
    B, C, D, H, W, _ = yh_j.shape

    real = yh_j[..., 0]
    imag = yh_j[..., 1]
    M = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
    sin_phi = imag / (M + 1e-8)
    cos_phi = real / (M + 1e-8)

    M_flat = M.reshape(B, C * D, H, W)              # [B, 18, H, W]
    sin_flat = sin_phi.reshape(B, C * D, H, W)      # [B, 18, H, W]
    cos_flat = cos_phi.reshape(B, C * D, H, W)      # [B, 18, H, W]

    # Run through the chosen branch step by step
    if args.branch == 'magnitude':
        x_in = M_flat
        dw, sa, pw = proc.M_dw, proc.M_sa, proc.M_pw
        title_prefix = 'Magnitude branch'
    else:
        x_in = torch.cat([sin_flat, cos_flat], dim=1)   # [B, 36, H, W]
        dw, sa, pw = proc.phase_dw, proc.phase_sa, proc.phase_pw
        title_prefix = 'Phase branch'

    with torch.no_grad():
        # Stage 1: Input
        x0 = x_in
        # Stage 2: DWConv
        x1 = dw(x0)
        # Stage 3: SiLU
        x2 = F.silu(x1)
        # Stage 4: SpatialAttention
        # need attention map separately for visualization
        avg = x2.mean(dim=1, keepdim=True)
        max_p = x2.max(dim=1, keepdim=True)[0]
        attn_input = torch.cat([avg, max_p], dim=1)
        attn_map = torch.sigmoid(sa.conv(attn_input))   # [B, 1, H, W]
        x3 = x2 * attn_map                               # weighted
        # Stage 5: 1×1 Conv
        x4 = pw(x3)

    # === Plot ===
    n_in_channels = x_in.shape[1]
    n_out_channels = x4.shape[1]
    ch = min(args.channel, n_in_channels - 1)
    out_ch = 0   # show first output channel

    stages = [
        ('Input', x0[0, ch], f'{x0.shape[1]} ch'),
        ('After DWConv 3×3', x1[0, ch], f'{x1.shape[1]} ch'),
        ('After SiLU', x2[0, ch], f'{x2.shape[1]} ch'),
        ('After Spatial Attn', x3[0, ch], f'{x3.shape[1]} ch'),
        ('After 1×1 Conv', x4[0, out_ch], f'{x4.shape[1]} ch (out_ch=0)'),
    ]

    fig, axes = plt.subplots(2, len(stages), figsize=(4 * len(stages), 6))

    for i, (name, feat, channel_str) in enumerate(stages):
        # Top row: feature heatmap
        im = to_heatmap(feat)
        vmin, vmax = np.percentile(im, [5, 99])
        axes[0, i].imshow(im, cmap='magma', vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f'{name}\n{channel_str}\n[H={feat.shape[0]}, W={feat.shape[1]}]',
                             fontsize=10)
        axes[0, i].axis('off')

        # Bottom row: stage label / additional info
        if i == 3:
            # Show attention map
            im_attn = attn_map[0, 0].cpu().numpy()
            axes[1, i].imshow(im_attn, cmap='hot', vmin=0, vmax=1)
            axes[1, i].set_title('Attention map [B, 1, H, W]', fontsize=10)
            axes[1, i].axis('off')
        else:
            axes[1, i].axis('off')
            axes[1, i].text(0.5, 0.5,
                            f'channel {ch if i < 4 else out_ch}',
                            ha='center', va='center', fontsize=12,
                            transform=axes[1, i].transAxes)

    plt.suptitle(
        f'{title_prefix} (level j={args.level}) — feature flow visualization\n'
        'Heatmap shows log-magnitude of the visualized channel.',
        fontsize=12, y=1.02)
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
