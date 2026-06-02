"""Visualize LR / SR / GT for consecutive frames to show end-to-end
temporal stability of the final outputs (not just intermediate SFT
modulation parameters).

The previous trajectory plot + (γ, β) grid showed *internal*
modulation stability. Reviewers without ML background may struggle to
connect abstract heatmaps to actual SR quality. This script produces
the complementary *external* visualization: actual SR frames side by
side with LR input and GT, so the temporal stability is visible
directly in pixel space.

Outputs:
  - results_lr_sr_gt/grid_{clip}_f{start}.png : 3-column grid
        rows = consecutive frames
        cols = LR | SR(Ours) | GT
  - results_lr_sr_gt/diff_{clip}_f{start}.png : 2-column difference grid
        rows = consecutive frame pairs (n-1, n)
        cols = |LR_n - LR_{n-1}| | |SR_n - SR_{n-1}|
        (visualizes "how much each pipeline output changes between frames")

Usage
-----
    PYTHONPATH=. python scripts/visualize_lr_sr_gt.py \\
        --ckpt /home/yjcho/StableVSR/experiments/20260430_dualsft/checkpoint-20000/ \\
        --reds_lr_root /mnt/HDD_raid1/yjcho/20260430/reds/ \\
        --reds_gt_root /mnt/HDD_raid1/yjcho/20260430/reds/ \\
        --pretrained_model claudiom4sir/StableVSR \\
        --clip 011 \\
        --start_frame 0 --n_frames 8 \\
        --inject_high 256 --inject_low 256 \\
        --output_dir results_lr_sr_gt/

Notes
-----
- LR root layout tries: <root>/<clip>/, <root>/train_sharp_bicubic/X4/<clip>/,
  <root>/val_sharp_bicubic/X4/<clip>/, <root>/REDS4/<clip>/
- GT root layout tries: <root>/<clip>/, <root>/train_sharp/<clip>/,
  <root>/val_sharp/<clip>/, <root>/REDS4_HR/<clip>/
- Pipeline runs the full bidirectional sweep (multiple minutes per clip
  on A100). Restrict n_frames to keep runtime tractable.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from util.sft_utils import FrequencyConditioningEncoder, UNetWithDualSFT
from util.frequency_utils import DTCWTForward


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def find_lr_dir(reds_lr_root: Path, clip: str) -> Optional[Path]:
    for sub in ("", "train_sharp_bicubic/X4", "val_sharp_bicubic/X4", "REDS4"):
        p = reds_lr_root / sub / clip if sub else reds_lr_root / clip
        if p.is_dir():
            return p
    return None


def find_gt_dir(reds_gt_root: Path, clip: str) -> Optional[Path]:
    for sub in ("", "train_sharp", "val_sharp", "REDS4_HR", "REDS4_GT"):
        p = reds_gt_root / sub / clip if sub else reds_gt_root / clip
        if p.is_dir():
            return p
    return None


def load_consecutive(dir_path: Path, start: int, n: int, crop_lr: int) -> List[Image.Image]:
    """Load n consecutive PNG/JPG frames as PIL images. crop_lr is the LR center crop size."""
    paths = sorted(p for p in dir_path.iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    end = start + n
    if end > len(paths):
        print(f"[warn] requested frames {start}..{end-1} but only {len(paths)} available")
        end = len(paths)
    out = []
    for p in paths[start:end]:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        left, top = (w - crop_lr) // 2, (h - crop_lr) // 2
        img = img.crop((left, top, left + crop_lr, top + crop_lr))
        out.append(img)
    return out


def load_gt_consecutive(dir_path: Path, start: int, n: int, crop_hr: int) -> List[Image.Image]:
    """Same as load_consecutive but with HR crop size."""
    return load_consecutive(dir_path, start, n, crop_hr)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_encoder(ckpt_dir: Path, inject_high: int, inject_low: int, device):
    enc = FrequencyConditioningEncoder(
        in_channels=18, mid_channels=64,
        sft_out_channels_high=inject_high,
        sft_out_channels_low=inject_low,
    )
    state = torch.load(ckpt_dir / "sft_adapter.bin", map_location="cpu")
    miss, unexp = enc.load_state_dict(state, strict=False)
    print(f"[load] encoder: missing={len(miss)} unexpected={len(unexp)}")
    return enc.to(device).eval()


def build_pipeline(args, device, encoder, weight_dtype):
    """Reconstruct the full StableVSR pipeline with our trained components."""
    from diffusers import AutoencoderKL, ControlNetModel, UNet2DConditionModel
    from transformers import AutoTokenizer
    from pipeline.stablevsr_pipeline import StableVSRPipeline
    from scheduler.ddpm_scheduler import DDPMScheduler

    # text + vae + unet
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model, subfolder="tokenizer", use_fast=False,
    )
    # Reuse train.py's helper to pick the right text encoder class.
    from train import import_model_class_from_model_name_or_path
    text_cls = import_model_class_from_model_name_or_path(args.pretrained_model, None)
    text_encoder = text_cls.from_pretrained(
        args.pretrained_model, subfolder="text_encoder", torch_dtype=weight_dtype,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model, subfolder="vae", torch_dtype=weight_dtype,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model, subfolder="unet", torch_dtype=weight_dtype,
    )
    controlnet = ControlNetModel.from_pretrained(args.ckpt, subfolder="controlnet")
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")

    unet_with_sft = UNetWithDualSFT(unet)
    unet_with_sft.cond_encoder = encoder

    pipe = StableVSRPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        unet=unet, controlnet=controlnet, scheduler=scheduler,
        safety_checker=None, feature_extractor=None, requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe, unet_with_sft


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_pipeline(pipe, unet_with_sft, lr_pil_list, args, device):
    """Run the bidirectional pipeline on consecutive LR frames; return PIL list."""
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

    of_model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()
    dtcwt_model = DTCWTForward(J=4, biort="near_sym_a", qshift="qshift_a").to(device).eval()
    for q in of_model.parameters():
        q.requires_grad_(False)
    for q in dtcwt_model.parameters():
        q.requires_grad_(False)

    print(f"[infer] running pipeline on {len(lr_pil_list)} LR frames "
          f"({args.num_inference_steps} steps × bidirectional)")
    out = pipe(
        prompt="", images=lr_pil_list, guidance_scale=0,
        num_inference_steps=args.num_inference_steps,
        of_model=of_model, dtcwt_model=dtcwt_model,
        unet_with_sft=unet_with_sft,
        output_type="pil",
    ).images
    # out is a list-of-lists or list of PILs depending on pipeline version; flatten.
    if out and isinstance(out[0], list):
        out = [frame[0] for frame in out]
    return out


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_grid(lr_list, sr_list, gt_list, save_path: Path, title: str):
    """3-column grid: rows=frames, cols=LR | SR | GT."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[viz] matplotlib missing"); return

    n = min(len(lr_list), len(sr_list), len(gt_list))
    if n == 0:
        return
    fig, axs = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axs = [axs]
    for r in range(n):
        axs[r][0].imshow(lr_list[r]);  axs[r][0].axis("off")
        axs[r][1].imshow(sr_list[r]);  axs[r][1].axis("off")
        axs[r][2].imshow(gt_list[r]);  axs[r][2].axis("off")
        axs[r][0].text(-0.10, 0.5, f"frame {r}", transform=axs[r][0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=10)
        if r == 0:
            axs[0][0].set_title("LR (input)", fontsize=11)
            axs[0][1].set_title("SR (Ours)",  fontsize=11)
            axs[0][2].set_title("GT",         fontsize=11)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    print(f"[viz] saved {save_path}")


def save_diff_grid(lr_list, sr_list, save_path: Path, title: str):
    """Difference grid: rows=consecutive pairs, cols=|ΔLR| | |ΔSR|.

    Both maps are min-max normalised independently per column for fair
    visual comparison across rows.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    n_pairs = min(len(lr_list), len(sr_list)) - 1
    if n_pairs <= 0:
        return

    # Compute absolute differences as numpy arrays.
    def pil_to_np(p):  # [H, W, 3] in [0,1]
        return torch.from_numpy(__import__("numpy").array(p)).float().div(255).numpy()

    lr_diffs, sr_diffs = [], []
    for n in range(1, len(lr_list)):
        d_lr = abs(pil_to_np(lr_list[n]) - pil_to_np(lr_list[n - 1])).mean(axis=-1)
        d_sr = abs(pil_to_np(sr_list[n]) - pil_to_np(sr_list[n - 1])).mean(axis=-1)
        # Match LR resolution to SR for visual alignment.
        lr_diffs.append(d_lr)
        sr_diffs.append(d_sr)

    # Global per-column normalisation for fair comparison.
    lr_max = max(d.max() for d in lr_diffs) + 1e-12
    sr_max = max(d.max() for d in sr_diffs) + 1e-12

    fig, axs = plt.subplots(n_pairs, 2, figsize=(8, 4 * n_pairs))
    if n_pairs == 1:
        axs = [axs]
    for r in range(n_pairs):
        axs[r][0].imshow(lr_diffs[r] / lr_max, cmap="hot", vmin=0, vmax=1)
        axs[r][1].imshow(sr_diffs[r] / sr_max, cmap="hot", vmin=0, vmax=1)
        for c in (0, 1):
            axs[r][c].axis("off")
        axs[r][0].text(-0.10, 0.5, f"frames\n{r}-{r+1}", transform=axs[r][0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=9)
        if r == 0:
            axs[0][0].set_title("|ΔLR|", fontsize=11)
            axs[0][1].set_title("|ΔSR|", fontsize=11)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    print(f"[viz] saved {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--reds_lr_root", required=True, type=Path)
    p.add_argument("--reds_gt_root", required=True, type=Path,
                   help="HR ground-truth root (often same as LR root)")
    p.add_argument("--pretrained_model", required=True,
                   help="Base SD pipeline (e.g. claudiom4sir/StableVSR)")
    p.add_argument("--clip", default="011")
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--n_frames", type=int, default=8)
    p.add_argument("--crop_lr", type=int, default=128,
                   help="LR center-crop size (matches training pipeline)")
    p.add_argument("--inject_high", type=int, default=320)
    p.add_argument("--inject_low", type=int, default=320)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", type=Path, default=Path("results_lr_sr_gt"))
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    weight_dtype = torch.float16  # standard for inference; encoder stays fp32

    # ---- Resolve paths ----
    lr_dir = find_lr_dir(args.reds_lr_root, args.clip)
    gt_dir = find_gt_dir(args.reds_gt_root, args.clip)
    if lr_dir is None:
        sys.exit(f"LR dir for clip {args.clip} not found under {args.reds_lr_root}")
    if gt_dir is None:
        sys.exit(f"GT dir for clip {args.clip} not found under {args.reds_gt_root}")
    print(f"[paths] LR: {lr_dir}\n[paths] GT: {gt_dir}")

    # ---- Load consecutive LR + GT ----
    lr_pil = load_consecutive(lr_dir, args.start_frame, args.n_frames, args.crop_lr)
    crop_hr = args.crop_lr * 4
    gt_pil = load_gt_consecutive(gt_dir, args.start_frame, args.n_frames, crop_hr)
    print(f"[load] {len(lr_pil)} LR frames, {len(gt_pil)} GT frames")

    # ---- Build pipeline + encoder ----
    print("[init] building encoder + pipeline ...")
    encoder = load_encoder(args.ckpt, args.inject_high, args.inject_low, device)
    pipe, unet_with_sft = build_pipeline(args, device, encoder, weight_dtype)

    # ---- Run inference ----
    sr_pil = run_pipeline(pipe, unet_with_sft, lr_pil, args, device)
    print(f"[infer] got {len(sr_pil)} SR frames")

    # ---- Save grids ----
    title_main = (f"LR  /  SR (Ours)  /  GT  --  clip {args.clip}, "
                  f"frames {args.start_frame}..{args.start_frame + len(lr_pil) - 1}")
    save_grid(lr_pil, sr_pil, gt_pil,
              save_path=args.output_dir / f"grid_{args.clip}_f{args.start_frame}.png",
              title=title_main)

    title_diff = (f"|ΔLR| vs |ΔSR| between consecutive frames -- "
                  f"clip {args.clip}, frames {args.start_frame}..{args.start_frame + len(lr_pil) - 1}\n"
                  "(small = temporally stable; SR should look much darker than LR)")
    save_diff_grid(lr_pil, sr_pil,
                   save_path=args.output_dir / f"diff_{args.clip}_f{args.start_frame}.png",
                   title=title_diff)

    print(f"\n[done] outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
