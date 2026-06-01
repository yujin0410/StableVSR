"""Empirical verification of the Lipschitz-anchored conditioning theory.

Measures, on a trained dual-SFT checkpoint:

  Phase 1 (fast, ~10 min on REDS4, no inference required):
    - K'_input:   ||Δ(transform-domain feature)|| / ||ΔLR||  per transform
                  (DT-CWT magnitude, FFT magnitude, raw pixel)
    - K'_encoder: ||Δ(γ, β)|| / ||ΔLR||  via the trained encoder
                  (only DT-CWT mode; pixel/fft/dwt require their own
                  checkpoint with matching pixel_cond_model)
    - ΔLR is motion-compensated via RAFT (LR_n vs warp(LR_{n-1})).

  Phase 2 (slow, opt-in via --run_inference):
    - K'_e2e:    ||Δ(x̂_HR)|| / ||ΔLR||  via the StableVSR pipeline
                  for each LR pair (motion-compensated at HR resolution).

Outputs:
    JSON with per-pair records and aggregate summary statistics.
    PNG scatter plot if matplotlib is available.

Theory under test
-----------------
The composed-bound theory (Lipschitz + DT-CWT shift invariance +
conditional sampling) predicts:
    ||Δx̂||_HR ≤ L · K'_encoder · ||ΔLR||
If the data scatter lies under a linear envelope with finite slope and
the slope tracks K'_encoder across transforms, the theory is supported.
If not, the temporal gain comes from a different mechanism.

Usage
-----
    python scripts/verify_lipschitz.py \
        --ckpt /home/yjcho/StableVSR/experiments/20260430_dualsft/checkpoint-20000/ \
        --reds_lr_root /mnt/HDD_raid1/yjcho/20260430/reds/ \
        --clips 000 011 015 020 \
        --max_frames 30 \
        --output_dir results_lipschitz/

Add --run_inference for Phase 2 (requires the full pipeline; one-shot
StableVSR inference per LR pair, ~minutes per pair on an A100).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import statistics
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Allow `python scripts/verify_lipschitz.py` from the repo root by
# prepending the repo root to sys.path. Required because `scripts/` is
# not a package and Python only auto-adds the script's directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Repo modules — these live in the user's tree, not in this remote container.
from util.sft_utils import FrequencyConditioningEncoder
from util.frequency_utils import DTCWTForward
from util.flow_utils import get_flow, flow_warp
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_clip_dirs(reds_lr_root: Path, clip_names: List[str]) -> List[Path]:
    """Resolve clip dirs robustly against common REDS layouts.

    Tries, in order, for each clip name:
        <root>/<name>/
        <root>/train_sharp_bicubic/X4/<name>/
        <root>/val_sharp_bicubic/X4/<name>/
        <root>/REDS4/<name>/
    """
    candidates = []
    for name in clip_names:
        for sub in ("", "train_sharp_bicubic/X4", "val_sharp_bicubic/X4", "REDS4"):
            p = reds_lr_root / sub / name if sub else reds_lr_root / name
            if p.is_dir():
                candidates.append(p)
                break
        else:
            print(f"[warn] clip {name} not found under {reds_lr_root}", file=sys.stderr)
    return candidates


def load_clip(clip_dir: Path, max_frames: int, crop: int) -> torch.Tensor:
    """Load up to max_frames LR PNG frames as a [T, 3, H, W] tensor in [0, 1]."""
    frame_paths = sorted(
        p for p in clip_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )[:max_frames]
    to_tensor = transforms.ToTensor()
    frames = []
    for p in frame_paths:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        left, top = (w - crop) // 2, (h - crop) // 2
        img = img.crop((left, top, left + crop, top + crop))
        frames.append(to_tensor(img))
    return torch.stack(frames)


# ---------------------------------------------------------------------------
# Distance measurements
# ---------------------------------------------------------------------------

@torch.no_grad()
def motion_compensated_lr_diff(frame_n, frame_prev, of_model, device):
    """||LR_n - warp(LR_{n-1}, flow_{n-1 -> n})|| in [-1, 1] space.

    Returns (delta_norm, lr_n_norm, flow).  flow returned so the caller
    can reuse it at HR (×4) for end-to-end measurement.
    """
    a = (frame_n.to(device).float() * 2 - 1).unsqueeze(0)
    b = (frame_prev.to(device).float() * 2 - 1).unsqueeze(0)
    flow = get_flow(of_model, a, b)
    b_warp = flow_warp(b, flow)
    delta = (a - b_warp).norm().item()
    norm_a = a.norm().item()
    return delta, norm_a, flow


@torch.no_grad()
def dtcwt_magnitude_diff(lr_n, lr_prev, dtcwt_model, device):
    """||M[LR_n] - M[LR_prev]||_F summed over levels (Pillar B input check)."""
    a = (lr_n.to(device).float() * 2 - 1).unsqueeze(0)
    b = (lr_prev.to(device).float() * 2 - 1).unsqueeze(0)
    _, yh_a = dtcwt_model(a)
    _, yh_b = dtcwt_model(b)
    sq, norm_sq = 0.0, 0.0
    for ya, yb in zip(yh_a, yh_b):
        mag_a = torch.sqrt(ya[..., 0] ** 2 + ya[..., 1] ** 2 + 1e-8)
        mag_b = torch.sqrt(yb[..., 0] ** 2 + yb[..., 1] ** 2 + 1e-8)
        sq += (mag_a - mag_b).pow(2).sum().item()
        norm_sq += mag_a.pow(2).sum().item()
    return math.sqrt(sq), math.sqrt(norm_sq)


@torch.no_grad()
def fft_magnitude_diff(lr_n, lr_prev, device):
    """||M_FFT[LR_n] - M_FFT[LR_prev]||_F  (Pillar B counter-control)."""
    a = (lr_n.to(device).float() * 2 - 1).unsqueeze(0)
    b = (lr_prev.to(device).float() * 2 - 1).unsqueeze(0)
    fa, fb = torch.fft.fft2(a).abs(), torch.fft.fft2(b).abs()
    return (fa - fb).norm().item(), fa.norm().item()


@torch.no_grad()
def encoder_cond_diff(lr_n, lr_prev, encoder, dtcwt_model, device):
    """||Δ(γ_L, β_L, γ_H, β_H)||_F via the trained encoder."""
    a = (lr_n.to(device).float() * 2 - 1).unsqueeze(0)
    b = (lr_prev.to(device).float() * 2 - 1).unsqueeze(0)
    yl_a, yh_a = dtcwt_model(a)
    yl_b, yh_b = dtcwt_model(b)
    out_a = encoder(yh_a, yl_a)
    out_b = encoder(yh_b, yl_b)
    sq, norm_sq = 0.0, 0.0
    for k in ("low_gamma", "low_beta", "high_gamma", "high_beta"):
        sq += (out_a[k] - out_b[k]).pow(2).sum().item()
        norm_sq += out_a[k].pow(2).sum().item()
    return math.sqrt(sq), math.sqrt(norm_sq)


@torch.no_grad()
def hr_pair_diff(x_hat_n, x_hat_prev, flow_lr, of_model, device):
    """Motion-compensated ||x̂_n - warp(x̂_{n-1})|| in HR space.

    flow_lr (from LR) is upscaled ×4 and scale-aware to apply at HR.
    """
    # Recompute HR flow directly on HR predictions for accuracy.
    flow_hr = get_flow(of_model, x_hat_n.float(), x_hat_prev.float())
    warp_prev = flow_warp(x_hat_prev.float(), flow_hr)
    return (x_hat_n.float() - warp_prev).norm().item(), x_hat_n.float().norm().item()


# ---------------------------------------------------------------------------
# Encoder loading
# ---------------------------------------------------------------------------

def build_encoder(ckpt_dir: Path, inject_high: int, inject_low: int, device) -> FrequencyConditioningEncoder:
    """Reconstruct FrequencyConditioningEncoder and load sft_adapter.bin.

    Channel counts default to 320 (SD 2.x base, block_out_channels[1] // 2)
    but should be overridden via --inject_high / --inject_low if the
    checkpoint was trained on a base whose UNet block_out_channels[1]
    differs (e.g. claudiom4sir/StableVSR uses 256 -> 128).
    """
    enc = FrequencyConditioningEncoder(
        in_channels=18,
        mid_channels=64,
        sft_out_channels_high=inject_high,
        sft_out_channels_low=inject_low,
    )
    sft_path = ckpt_dir / "sft_adapter.bin"
    if not sft_path.is_file():
        raise FileNotFoundError(f"sft_adapter.bin not found in {ckpt_dir}")
    state = torch.load(sft_path, map_location="cpu")
    missing, unexpected = enc.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"[load]   first missing: {missing[:3]}")
        if unexpected:
            print(f"[load]   first unexpected: {unexpected[:3]}")
    return enc.to(device).eval()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def aggregate_K(records: List[Dict], delta_key: str, denom_key: str = "delta_lr_mc") -> Dict[str, float]:
    """Slope summary: median / mean / std of (delta / ΔLR)."""
    vals = [r[delta_key] / max(r[denom_key], 1e-9) for r in records]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "p10": sorted(vals)[int(0.10 * len(vals))],
        "p90": sorted(vals)[int(0.90 * len(vals))],
    }


def maybe_plot(records: List[Dict], summary: Dict, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not available, skipping plot")
        return
    if not records:
        return

    xs = [r["delta_lr_mc"] for r in records]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    series = [
        ("delta_cond_enc", "Encoder (γ, β)  [DT-CWT trained]"),
        ("delta_mag_dtcwt", "Input  M_DTCWT (no encoder)"),
        ("delta_mag_fft", "Input  M_FFT   (no encoder)"),
    ]
    for ax, (key, title) in zip(axes, series):
        ys = [r[key] for r in records]
        ax.scatter(xs, ys, s=10, alpha=0.5)
        # Linear envelope through the origin (Lipschitz upper bound)
        if xs and ys:
            slope = max(y / max(x, 1e-9) for x, y in zip(xs, ys))
            xx = sorted(xs)
            ax.plot(xx, [slope * v for v in xx], "r--", linewidth=1,
                    label=f"upper envelope (slope={slope:.2f})")
            # Median-slope reference
            med = statistics.median([y / max(x, 1e-9) for x, y in zip(xs, ys)])
            ax.plot(xx, [med * v for v in xx], "g--", linewidth=1,
                    label=f"median slope ({med:.2f})")
            ax.legend(fontsize=8)
        ax.set_xlabel("||ΔLR||  (motion-compensated)")
        ax.set_ylabel(f"||Δ{title.split()[0]}||")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle("Lipschitz-anchored conditioning verification", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[plot] saved scatter to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--reds_lr_root", required=True, type=Path)
    p.add_argument("--clips", nargs="+", default=["000", "011", "015", "020"])
    p.add_argument("--max_frames", type=int, default=30)
    p.add_argument("--crop", type=int, default=128, help="LR center-crop (pipeline uses 128)")
    p.add_argument("--inject_high", type=int, default=320,
                   help="per-level HIGH SFT channels (= UNet block_out_channels[1] // 2)")
    p.add_argument("--inject_low", type=int, default=320)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", type=Path, default=Path("results_lipschitz"))
    p.add_argument("--run_inference", action="store_true",
                   help="Phase 2: run full StableVSR inference and measure HR-space ||Δx̂||")
    p.add_argument("--pretrained_model", default="stabilityai/stable-diffusion-2-1",
                   help="Base SD path for Phase 2 pipeline (only used with --run_inference)")
    p.add_argument("--num_inference_steps", type=int, default=50)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ---- Build models ----
    print("[init] loading encoder + DT-CWT + RAFT ...")
    encoder = build_encoder(args.ckpt, args.inject_high, args.inject_low, device)
    dtcwt_model = DTCWTForward(J=4, biort="near_sym_a", qshift="qshift_a").to(device).eval()
    of_model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()
    for m in (encoder, dtcwt_model, of_model):
        for q in m.parameters():
            q.requires_grad_(False)

    # ---- Resolve clip paths ----
    clip_dirs = find_clip_dirs(args.reds_lr_root, args.clips)
    if not clip_dirs:
        sys.exit(f"No clips found under {args.reds_lr_root}")

    # ---- Phase 1: K' measurement ----
    records = []
    for clip_dir in clip_dirs:
        print(f"\n=== {clip_dir.name} ===")
        frames = load_clip(clip_dir, max_frames=args.max_frames, crop=args.crop)
        if len(frames) < 2:
            continue

        # Cache HR predictions if Phase 2 is enabled.
        hr_preds = None
        if args.run_inference:
            hr_preds = run_pipeline_on_clip(
                frames, args, device, encoder
            )

        for n in range(1, len(frames)):
            d_lr, n_lr, flow = motion_compensated_lr_diff(
                frames[n], frames[n - 1], of_model, device
            )
            d_mag_dtcwt, n_mag_dtcwt = dtcwt_magnitude_diff(
                frames[n], frames[n - 1], dtcwt_model, device
            )
            d_mag_fft, n_mag_fft = fft_magnitude_diff(
                frames[n], frames[n - 1], device
            )
            d_cond, n_cond = encoder_cond_diff(
                frames[n], frames[n - 1], encoder, dtcwt_model, device
            )

            rec = {
                "clip": clip_dir.name,
                "frame": n,
                "delta_lr_mc": d_lr,
                "norm_lr": n_lr,
                "delta_mag_dtcwt": d_mag_dtcwt,
                "norm_mag_dtcwt": n_mag_dtcwt,
                "delta_mag_fft": d_mag_fft,
                "norm_mag_fft": n_mag_fft,
                "delta_cond_enc": d_cond,
                "norm_cond": n_cond,
            }

            if hr_preds is not None and n < len(hr_preds):
                d_hr, n_hr = hr_pair_diff(
                    hr_preds[n], hr_preds[n - 1], flow, of_model, device
                )
                rec.update({"delta_x_hat_mc": d_hr, "norm_x_hat": n_hr})

            records.append(rec)

            print(
                f"  f{n:02d}: ΔLR={d_lr:8.3f}  Δ(γ,β)={d_cond:8.3f}  "
                f"ΔM_DTCWT={d_mag_dtcwt:8.3f}  ΔM_FFT={d_mag_fft:8.3f}"
                + (f"  Δx̂={rec.get('delta_x_hat_mc', float('nan')):.3f}"
                   if hr_preds is not None else "")
            )

    # ---- Aggregate ----
    summary = {
        "K_input_DTCWT_magnitude": aggregate_K(records, "delta_mag_dtcwt"),
        "K_input_FFT_magnitude":   aggregate_K(records, "delta_mag_fft"),
        "K_encoder_DTCWT_full":    aggregate_K(records, "delta_cond_enc"),
    }
    if records and "delta_x_hat_mc" in records[0]:
        summary["K_e2e_x_hat"] = aggregate_K(records, "delta_x_hat_mc")

    print("\n=== Summary (Lipschitz slope = Δ/ΔLR) ===")
    for k, v in summary.items():
        print(f"  {k}: median={v.get('median', float('nan')):.4e}  "
              f"mean={v.get('mean', float('nan')):.4e}  "
              f"p10={v.get('p10', float('nan')):.4e}  "
              f"p90={v.get('p90', float('nan')):.4e}  (n={v.get('n', 0)})")

    out_json = args.output_dir / "lipschitz_records.json"
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "records": records, "args": vars(args)},
                  f, indent=2, default=str)
    print(f"\n[save] {out_json}")

    maybe_plot(records, summary, args.output_dir / "lipschitz_scatter.png")


# ---------------------------------------------------------------------------
# Phase 2: full pipeline inference for end-to-end Lipschitz
# ---------------------------------------------------------------------------

def run_pipeline_on_clip(frames, args, device, encoder):
    """Run StableVSRPipeline on the entire clip in one call.

    Returns a list [T, 3, 4H, 4W] of HR predictions (CPU tensors).
    Cached per clip to amortize the dual-direction sweep cost.
    """
    print("[phase2] full pipeline inference (this is the slow path)")
    from diffusers import AutoencoderKL, ControlNetModel, UNet2DConditionModel
    from transformers import AutoTokenizer
    from pipeline.stablevsr_pipeline import StableVSRPipeline
    from scheduler.ddpm_scheduler import DDPMScheduler
    from util.sft_utils import UNetWithDualSFT

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model, subfolder="tokenizer", use_fast=False,
    )
    from train import import_model_class_from_model_name_or_path  # noqa
    text_cls = import_model_class_from_model_name_or_path(args.pretrained_model, None)
    text_encoder = text_cls.from_pretrained(args.pretrained_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(args.ckpt, subfolder="controlnet")
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")

    unet_with_sft = UNetWithDualSFT(unet)
    unet_with_sft.cond_encoder = encoder

    pipe = StableVSRPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        unet=unet, controlnet=controlnet, scheduler=scheduler,
        safety_checker=None, feature_extractor=None, requires_safety_checker=False,
    ).to(device)

    # The pipeline expects a list of PIL-like frames. We provide the
    # tensors converted into PIL since this matches log_validation.
    pil_frames = [transforms.ToPILImage()(f) for f in frames]

    # Hide tqdm
    pipe.set_progress_bar_config(disable=True)
    out = pipe(
        prompt="", images=pil_frames, guidance_scale=0,
        num_inference_steps=args.num_inference_steps,
        of_model=raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval(),
        dtcwt_model=DTCWTForward(J=4, biort="near_sym_a", qshift="qshift_a").to(device).eval(),
        unet_with_sft=unet_with_sft,
        output_type="pt",
    ).images
    return [t.squeeze(0).cpu() for t in out]


if __name__ == "__main__":
    main()
