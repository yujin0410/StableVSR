"""Magnitude-only vs Phase-only ablation on the trained checkpoint.

We've argued (from architecture) that magnitude is the dominant
contributor and phase is a secondary modifier. This script tests
that claim directly by intercepting the DT-CWT input to the encoder
and overriding one component at a time, without retraining.

Three modes:

  FULL          : real, imag from DT-CWT (current model)
  MAGNITUDE_ONLY: force phase = 0  (cos=1, sin=0)
                  i.e. real = sqrt(real^2 + imag^2),  imag = 0
                  Removes all oriented direction information.
  PHASE_ONLY    : force magnitude = 1
                  i.e. real = cos_phi,  imag = sin_phi
                  Removes the energy/strength signal.

Outputs per mode:

  Numerical
    - (gamma, beta) raw mean, |gamma-1| mean, |beta| mean per branch
    - Cross-frame stability: rel ||delta(gamma, beta)|| / ||(gamma, beta)||
      aggregated over consecutive REDS4 pairs. Tests whether removing
      one component breaks the temporal stability we measured at K=0.76.
    - Distance to FULL mode: ||(gamma, beta) - (gamma, beta)_FULL||
      relative to the FULL output's norm.

  Visual
    For one selected frame, save a grid PNG of three rows
        FULL | MAGNITUDE_ONLY | PHASE_ONLY
    with four columns
        gamma_L, beta_L, gamma_H, beta_H
    Each heatmap is per-column global min-max so rows are comparable.

Usage
-----
    PYTHONPATH=. python scripts/verify_mag_phase_ablation.py \\
        --ckpt /home/yjcho/StableVSR/experiments/20260430_dualsft/checkpoint-20000/ \\
        --reds_lr_root /mnt/HDD_raid1/yjcho/20260430/reds/ \\
        --clips 000 011 015 020 \\
        --viz_clip 011 --viz_frame_idx 5 \\
        --max_frames 30 \\
        --inject_high 256 --inject_low 256 \\
        --output_dir results_mag_phase_ablation/
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from util.sft_utils import FrequencyConditioningEncoder
from util.frequency_utils import DTCWTForward


def find_clip_dirs(reds_lr_root: Path, clip_names: List[str]) -> Dict[str, Path]:
    out = {}
    for name in clip_names:
        for sub in ("", "train_sharp_bicubic/X4", "val_sharp_bicubic/X4", "REDS4"):
            p = reds_lr_root / sub / name if sub else reds_lr_root / name
            if p.is_dir():
                out[name] = p
                break
        else:
            print(f"[warn] clip {name} not found")
    return out


def load_clip(clip_dir: Path, max_frames: int, crop: int) -> torch.Tensor:
    paths = sorted(p for p in clip_dir.iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg"})[:max_frames]
    to_t = transforms.ToTensor()
    frames = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        left, top = (w - crop) // 2, (h - crop) // 2
        img = img.crop((left, top, left + crop, top + crop))
        frames.append(to_t(img))
    return torch.stack(frames)


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


# ---------------------------------------------------------------------------
# DT-CWT input intervention
# ---------------------------------------------------------------------------

def make_modified_yh(yh: List[torch.Tensor], mode: str) -> List[torch.Tensor]:
    """Return a new yh with the (real, imag) components overridden per mode.

    yh[j] shape: [B, C, 6, H, W, 2]   last dim: (real, imag)
    """
    out = []
    for yh_j in yh:
        real = yh_j[..., 0]
        imag = yh_j[..., 1]
        eps = 1e-8
        if mode == "FULL":
            out.append(yh_j)
            continue
        elif mode == "MAGNITUDE_ONLY":
            # Phase forced to 0  ->  cos = 1, sin = 0
            M = torch.sqrt(real * real + imag * imag + eps)
            new_real = M
            new_imag = torch.zeros_like(M)
        elif mode == "PHASE_ONLY":
            # Magnitude forced to 1
            M = torch.sqrt(real * real + imag * imag + eps)
            new_real = real / (M + eps)
            new_imag = imag / (M + eps)
        else:
            raise ValueError(f"unknown mode {mode}")
        new_yh = torch.stack([new_real, new_imag], dim=-1)
        out.append(new_yh)
    return out


# ---------------------------------------------------------------------------
# Numerical: distribution stats per mode + cross-frame stability
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_mode_per_clip(clip_frames, encoder, dtcwt_model, device, mode: str):
    """Return list of cond dicts (one per frame) for the given mode."""
    cached = []
    for k in range(clip_frames.shape[0]):
        lr = clip_frames[k:k + 1].to(device).float() * 2 - 1
        yl, yh = dtcwt_model(lr)
        yh_mod = make_modified_yh(yh, mode)
        out = encoder(yh_mod, yl)
        cached.append({kk: vv.detach() for kk, vv in out.items()})
    return cached


def distribution_stats(cached_outs: List[Dict]) -> Dict:
    """|gamma-1| / |beta| stats across frames per branch."""
    stats = {}
    for branch, gkey, bkey in [
        ("LOW",  "low_gamma",  "low_beta"),
        ("HIGH", "high_gamma", "high_beta"),
    ]:
        g_dev, b_abs = [], []
        g_mean, b_mean = [], []
        for c in cached_outs:
            g = c[gkey].float()
            b = c[bkey].float()
            g_dev.append((g - 1).abs().mean().item())
            b_abs.append(b.abs().mean().item())
            g_mean.append(g.mean().item())
            b_mean.append(b.mean().item())
        stats[f"{branch}_|gamma-1|_mean"] = statistics.mean(g_dev)
        stats[f"{branch}_|beta|_mean"] = statistics.mean(b_abs)
        stats[f"{branch}_gamma_raw_mean"] = statistics.mean(g_mean)
        stats[f"{branch}_beta_raw_mean"] = statistics.mean(b_mean)
    return stats


def cross_frame_stability(cached_outs: List[Dict]) -> Dict:
    """Relative change of (gamma, beta) between adjacent frames."""
    rec = {"rel_gamma_LOW": [], "rel_beta_LOW": [],
           "rel_gamma_HIGH": [], "rel_beta_HIGH": []}
    for n in range(1, len(cached_outs)):
        for branch, gkey, bkey in [
            ("LOW",  "low_gamma",  "low_beta"),
            ("HIGH", "high_gamma", "high_beta"),
        ]:
            g_n, g_p = cached_outs[n][gkey].float(), cached_outs[n - 1][gkey].float()
            b_n, b_p = cached_outs[n][bkey].float(), cached_outs[n - 1][bkey].float()
            rec[f"rel_gamma_{branch}"].append(
                ((g_n - g_p).norm() / g_n.norm().clamp(min=1e-12)).item()
            )
            rec[f"rel_beta_{branch}"].append(
                ((b_n - b_p).norm() / b_n.norm().clamp(min=1e-12)).item()
            )
    return {k: {"median": statistics.median(v),
                "mean":   statistics.mean(v),
                "n":      len(v)}
            for k, v in rec.items()}


def distance_from_full(cached_outs_mode: List[Dict],
                       cached_outs_full: List[Dict]) -> Dict:
    """How far does this mode's (gamma, beta) deviate from FULL?"""
    rec = {}
    for branch, gkey, bkey in [
        ("LOW",  "low_gamma",  "low_beta"),
        ("HIGH", "high_gamma", "high_beta"),
    ]:
        vals_g, vals_b = [], []
        for c_m, c_f in zip(cached_outs_mode, cached_outs_full):
            g_m, g_f = c_m[gkey].float(), c_f[gkey].float()
            b_m, b_f = c_m[bkey].float(), c_f[bkey].float()
            vals_g.append(((g_m - g_f).norm() / g_f.norm().clamp(min=1e-12)).item())
            vals_b.append(((b_m - b_f).norm() / b_f.norm().clamp(min=1e-12)).item())
        rec[f"{branch}_gamma_dist_from_FULL_median"] = statistics.median(vals_g)
        rec[f"{branch}_beta_dist_from_FULL_median"] = statistics.median(vals_b)
    return rec


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_mode_grid(clip_frames, encoder, dtcwt_model, device, save_path: Path,
                   frame_idx: int):
    """Grid: 3 rows (FULL, MAG_ONLY, PHASE_ONLY) x 4 cols (γ_L, β_L, γ_H, β_H)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    lr = clip_frames[frame_idx:frame_idx + 1].to(device).float() * 2 - 1

    yl, yh = dtcwt_model(lr)
    outs = {}
    for mode in ("FULL", "MAGNITUDE_ONLY", "PHASE_ONLY"):
        yh_mod = make_modified_yh(yh, mode)
        outs[mode] = {k: v.detach().cpu() for k, v in encoder(yh_mod, yl).items()}

    def map2d(t):
        return t.float().mean(dim=1).squeeze(0).cpu()

    modes = ["FULL", "MAGNITUDE_ONLY", "PHASE_ONLY"]
    cols = ["low_gamma", "low_beta", "high_gamma", "high_beta"]
    col_titles = ["γ_LOW", "β_LOW", "γ_HIGH", "β_HIGH"]

    # Per-column min-max across the three modes for fair comparison.
    col_ranges = {}
    for c in cols:
        stack = torch.stack([map2d(outs[m][c]) for m in modes])
        col_ranges[c] = (stack.min().item(), stack.max().item())

    fig, axs = plt.subplots(4, 4, figsize=(13, 13))
    # Top row: original LR + LR magnitude reference + repeat for sym layout.
    lr_pil = ((lr.squeeze(0) + 1) / 2).clamp(0, 1).permute(1, 2, 0).cpu()
    axs[0][0].imshow(lr_pil.numpy())
    axs[0][0].set_title(f"LR input\n(frame {frame_idx})")
    axs[0][0].axis("off")
    for c in (1, 2, 3):
        axs[0][c].axis("off")

    for r, mode in enumerate(modes, start=1):
        for c, (cname, ctitle) in enumerate(zip(cols, col_titles)):
            ax = axs[r][c]
            lo, hi = col_ranges[cname]
            ax.imshow(map2d(outs[mode][cname]).numpy(), cmap="viridis",
                      vmin=lo, vmax=hi)
            ax.axis("off")
            if c == 0:
                ax.text(-0.10, 0.5, mode, transform=ax.transAxes,
                        rotation=90, va="center", ha="right", fontsize=10)
            if r == 1:
                ax.set_title(ctitle, fontsize=11)

    fig.suptitle("Magnitude-only vs Phase-only vs Full SFT outputs\n"
                 "(per-column normalised so rows are visually comparable)",
                 fontsize=12)
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
    p.add_argument("--clips", nargs="+", default=["000", "011", "015", "020"])
    p.add_argument("--max_frames", type=int, default=30)
    p.add_argument("--crop", type=int, default=128)
    p.add_argument("--viz_clip", default="011")
    p.add_argument("--viz_frame_idx", type=int, default=5)
    p.add_argument("--inject_high", type=int, default=320)
    p.add_argument("--inject_low", type=int, default=320)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", type=Path, default=Path("results_mag_phase_ablation"))
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("[init] loading encoder + DT-CWT ...")
    encoder = load_encoder(args.ckpt, args.inject_high, args.inject_low, device)
    dtcwt_model = DTCWTForward(J=4, biort="near_sym_a", qshift="qshift_a").to(device).eval()
    for q in dtcwt_model.parameters():
        q.requires_grad_(False)

    clip_dirs = find_clip_dirs(args.reds_lr_root, args.clips)
    if not clip_dirs:
        sys.exit("No clips found.")

    modes = ["FULL", "MAGNITUDE_ONLY", "PHASE_ONLY"]
    aggregate: Dict[str, Dict] = {m: {"distribution": {}, "stability": {},
                                       "dist_from_FULL": {}} for m in modes}
    raw_per_clip = {}
    for clip_name, clip_dir in clip_dirs.items():
        print(f"\n=== {clip_name} ===")
        frames = load_clip(clip_dir, args.max_frames, args.crop)
        cached_full = run_mode_per_clip(frames, encoder, dtcwt_model, device, "FULL")
        cached_mag  = run_mode_per_clip(frames, encoder, dtcwt_model, device, "MAGNITUDE_ONLY")
        cached_pha  = run_mode_per_clip(frames, encoder, dtcwt_model, device, "PHASE_ONLY")

        raw_per_clip[clip_name] = {
            "FULL_distribution":           distribution_stats(cached_full),
            "MAGNITUDE_ONLY_distribution": distribution_stats(cached_mag),
            "PHASE_ONLY_distribution":     distribution_stats(cached_pha),
            "FULL_stability":           cross_frame_stability(cached_full),
            "MAGNITUDE_ONLY_stability": cross_frame_stability(cached_mag),
            "PHASE_ONLY_stability":     cross_frame_stability(cached_pha),
            "MAGNITUDE_ONLY_dist_from_FULL": distance_from_full(cached_mag, cached_full),
            "PHASE_ONLY_dist_from_FULL":     distance_from_full(cached_pha, cached_full),
        }

        if clip_name == args.viz_clip:
            save_mode_grid(
                frames, encoder, dtcwt_model, device,
                save_path=args.output_dir / f"grid_modes_{clip_name}_f{args.viz_frame_idx}.png",
                frame_idx=args.viz_frame_idx,
            )

    # Aggregate across clips.
    def median_across_clips(field_path):
        vals = []
        for clip in raw_per_clip.values():
            d = clip
            for k in field_path:
                d = d.get(k, {})
                if not isinstance(d, dict):
                    break
            if isinstance(d, dict) and "median" in d:
                vals.append(d["median"])
            elif isinstance(d, (int, float)):
                vals.append(d)
        return statistics.median(vals) if vals else float("nan")

    print("\n=== Per-mode summary (median across REDS4 clips) ===")
    print(f"\n  Distribution stats (mean |gamma-1| / |beta|):")
    for mode in modes:
        print(f"\n  [{mode}]")
        for branch in ("LOW", "HIGH"):
            for sub in (f"|gamma-1|_mean", f"|beta|_mean"):
                key = f"{branch}_{sub}"
                vals = [raw_per_clip[c][f"{mode}_distribution"].get(key)
                        for c in raw_per_clip]
                vals = [v for v in vals if v is not None]
                if vals:
                    print(f"    {branch:<5s} {sub:<22s}  median={statistics.median(vals):.4f}")

    print(f"\n  Cross-frame stability (rel ||delta(gamma,beta)|| / ||(gamma,beta)||):")
    for mode in modes:
        print(f"\n  [{mode}]")
        for kk in ("rel_gamma_LOW", "rel_beta_LOW", "rel_gamma_HIGH", "rel_beta_HIGH"):
            vals = [raw_per_clip[c][f"{mode}_stability"][kk]["median"]
                    for c in raw_per_clip]
            print(f"    {kk:<18s}  median={statistics.median(vals):.4f}")

    print(f"\n  Distance from FULL (||mode - FULL|| / ||FULL||):")
    for mode in ("MAGNITUDE_ONLY", "PHASE_ONLY"):
        print(f"\n  [{mode}]")
        for kk in ("LOW_gamma_dist_from_FULL_median", "LOW_beta_dist_from_FULL_median",
                   "HIGH_gamma_dist_from_FULL_median", "HIGH_beta_dist_from_FULL_median"):
            vals = [raw_per_clip[c][f"{mode}_dist_from_FULL"][kk]
                    for c in raw_per_clip]
            print(f"    {kk:<40s}  median={statistics.median(vals):.4f}")

    out_json = args.output_dir / "mag_phase_ablation.json"
    with open(out_json, "w") as f:
        json.dump(raw_per_clip, f, indent=2, default=str)
    print(f"\n[done] {out_json}")


if __name__ == "__main__":
    main()
