"""Print parameter counts and channel configuration for the trained dual-SFT model.

Usage:
    python print_dualsft_info.py <checkpoint_dir>
    # e.g. python print_dualsft_info.py experiments/20260430_dualsft/checkpoint-20000
"""

import os
import sys
import torch
from diffusers import UNet2DConditionModel, ControlNetModel, AutoencoderKL
from util.sft_utils import (
    FrequencyConditioningEncoder, UNetWithDualSFT,
    PerLevelProcessor, SFTHead, SpatialAttention,
)


def fmt(n):
    return f"{n:,}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python print_dualsft_info.py <checkpoint_dir>")
        sys.exit(1)
    ckpt_dir = sys.argv[1]
    sft_path = os.path.join(ckpt_dir, "sft_adapter.bin")
    if not os.path.isfile(sft_path):
        print(f"Not found: {sft_path}")
        sys.exit(1)

    model_id = "claudiom4sir/StableVSR"

    print("=== Loading UNet to read block_out_channels ===")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    boc = unet.config.block_out_channels
    print(f"UNet block_out_channels = {list(boc)}")
    inject = boc[1]
    per_level = inject // 2
    print(f"Inject channels = {inject}")
    print(f"Per-level (HIGH=LOW) = {per_level}")
    print()

    print("=== Building FrequencyConditioningEncoder with same config ===")
    enc = FrequencyConditioningEncoder(
        in_channels=18, mid_channels=64,
        sft_out_channels_high=per_level,
        sft_out_channels_low=per_level,
        high_target=None, low_target=None,
    )
    enc.load_state_dict(torch.load(sft_path, map_location="cpu"))
    enc_total = sum(p.numel() for p in enc.parameters())
    enc_train = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    print(f"FrequencyConditioningEncoder: total={fmt(enc_total)}, trainable={fmt(enc_train)}")
    print(f"  -> {enc_train / 1e6:.3f}M trainable")
    print()

    print("=== Per-PerLevelProcessor breakdown (proc_j1) ===")
    proc = enc.proc_j1
    for name, mod in proc.named_children():
        n = sum(p.numel() for p in mod.parameters())
        print(f"  {name:15s}: {fmt(n):>12}")
    proc_total = sum(p.numel() for p in proc.parameters())
    print(f"  {'TOTAL':15s}: {fmt(proc_total):>12}  ({proc_total/1e6:.3f}M)")
    print(f"  x4 processors  : {fmt(proc_total * 4):>12}")
    print()

    print("=== Other components (frozen at training) ===")
    n_unet = sum(p.numel() for p in unet.parameters())
    print(f"UNet:        total={fmt(n_unet)}  ({n_unet/1e6:.1f}M)")

    try:
        cn_path = os.path.join(ckpt_dir, "controlnet")
        if os.path.isdir(cn_path):
            cn = ControlNetModel.from_pretrained(ckpt_dir, subfolder="controlnet")
        else:
            cn = ControlNetModel.from_pretrained(model_id, subfolder="controlnet")
        n_cn = sum(p.numel() for p in cn.parameters())
        print(f"ControlNet:  total={fmt(n_cn)}  ({n_cn/1e6:.1f}M)  [trainable]")
    except Exception as e:
        print(f"ControlNet load skipped: {e}")

    try:
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        n_vae = sum(p.numel() for p in vae.parameters())
        print(f"VAE:         total={fmt(n_vae)}  ({n_vae/1e6:.1f}M)")
    except Exception as e:
        print(f"VAE load skipped: {e}")


if __name__ == "__main__":
    main()
