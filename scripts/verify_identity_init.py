"""Verify the 'identity initialization stays near identity' claim.

The Lipschitz-anchored conditioning narrative argues that identity init
(gamma_init=1, beta_init=0 in SFTHead) keeps the trained gamma close to
1 and beta close to 0, which in turn bounds the encoder's Lipschitz
constant. This script tests that claim directly on the trained
checkpoint, without retraining.

Two measurements:

  M1. Distribution of trained (gamma, beta) on real LR inputs.
      For each LR frame, forward through the encoder and report:
        |gamma - 1| mean / max / std    (deviation from identity scale)
        |beta|       mean / max / std    (deviation from identity shift)
      Aggregated over (B, C, H, W) per LOW/HIGH group.
      Expected if identity-init worked: small deviations
      (|gamma - 1| mean << 1, |beta| mean << 1).

  M2. Counterfactual: replace trained (gamma, beta) with forced
      identity (gamma=1, beta=0) and measure how much downstream
      U-Net features change.
      If forced identity ~ trained -> training kept gamma close to 1
      and the *added* deviation is small.
      If forced identity != trained -> learned deviation does
      contribute, but its magnitude tells us how far from identity.

Together these answer "did identity init really stay near identity?"

Usage
-----
    PYTHONPATH=. python scripts/verify_identity_init.py \\
        --ckpt /home/yjcho/StableVSR/experiments/20260430_dualsft/checkpoint-20000/ \\
        --reds_lr_root /mnt/HDD_raid1/yjcho/20260430/reds/ \\
        --clips 000 011 015 020 \\
        --max_frames 30 \\
        --inject_high 256 --inject_low 256 \\
        --output_dir results_identity_init/
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from util.sft_utils import FrequencyConditioningEncoder
from util.frequency_utils import DTCWTForward


# ---------------------------------------------------------------------------
# Helpers (shared with other scripts)
# ---------------------------------------------------------------------------

def find_clip_dirs(reds_lr_root: Path, clip_names: List[str]) -> List[Path]:
    cands = []
    for name in clip_names:
        for sub in ("", "train_sharp_bicubic/X4", "val_sharp_bicubic/X4", "REDS4"):
            p = reds_lr_root / sub / name if sub else reds_lr_root / name
            if p.is_dir():
                cands.append(p)
                break
        else:
            print(f"[warn] clip {name} not found")
    return cands


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
# M1. Distribution of trained (gamma, beta)
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_identity_deviation(frames, encoder, dtcwt_model, device) -> Dict:
    """Aggregate |gamma - 1| and |beta| statistics across frames."""
    print("\n=== M1: trained (gamma, beta) deviation from identity ===")

    stats = {
        "low_gamma":  {"abs_dev": [], "max_dev": [], "raw_mean": [], "raw_std": []},
        "low_beta":   {"abs": [], "max": [], "raw_mean": [], "raw_std": []},
        "high_gamma": {"abs_dev": [], "max_dev": [], "raw_mean": [], "raw_std": []},
        "high_beta":  {"abs": [], "max": [], "raw_mean": [], "raw_std": []},
    }

    for k in range(frames.shape[0]):
        lr = frames[k:k + 1].to(device).float() * 2 - 1
        yl, yh = dtcwt_model(lr)
        out = encoder(yh, yl)

        for gkey, bkey in [("low_gamma", "low_beta"), ("high_gamma", "high_beta")]:
            g = out[gkey].float()
            b = out[bkey].float()
            dev_g = (g - 1).abs()
            abs_b = b.abs()
            stats[gkey]["abs_dev"].append(dev_g.mean().item())
            stats[gkey]["max_dev"].append(dev_g.max().item())
            stats[gkey]["raw_mean"].append(g.mean().item())
            stats[gkey]["raw_std"].append(g.std().item())
            stats[bkey]["abs"].append(abs_b.mean().item())
            stats[bkey]["max"].append(abs_b.max().item())
            stats[bkey]["raw_mean"].append(b.mean().item())
            stats[bkey]["raw_std"].append(b.std().item())

    # Aggregate.
    summary = {}
    for name in ("low_gamma", "high_gamma"):
        v = stats[name]
        summary[name] = {
            "|gamma - 1| mean across frames": statistics.mean(v["abs_dev"]),
            "|gamma - 1| max  across frames": max(v["max_dev"]),
            "gamma raw mean":  statistics.mean(v["raw_mean"]),
            "gamma raw std":   statistics.mean(v["raw_std"]),
            "n": len(v["abs_dev"]),
        }
    for name in ("low_beta", "high_beta"):
        v = stats[name]
        summary[name] = {
            "|beta| mean across frames": statistics.mean(v["abs"]),
            "|beta| max  across frames": max(v["max"]),
            "beta raw mean":  statistics.mean(v["raw_mean"]),
            "beta raw std":   statistics.mean(v["raw_std"]),
            "n": len(v["abs"]),
        }

    print("\n  Identity deviation summary:")
    for k, v in summary.items():
        print(f"\n  [{k}]")
        for kk, vv in v.items():
            print(f"    {kk:<35s}  {vv:+.6f}" if isinstance(vv, float) else
                  f"    {kk:<35s}  {vv}")
    return summary


# ---------------------------------------------------------------------------
# M2. Counterfactual: how much does forcing identity change things?
# ---------------------------------------------------------------------------

@torch.no_grad()
def counterfactual_identity(frames, encoder, dtcwt_model, device) -> Dict:
    """Force gamma=1, beta=0 and compare with trained output magnitudes."""
    print("\n=== M2: trained gamma vs forced identity (gamma=1, beta=0) ===")

    rel_diff_g_low, rel_diff_b_low = [], []
    rel_diff_g_high, rel_diff_b_high = [], []

    for k in range(frames.shape[0]):
        lr = frames[k:k + 1].to(device).float() * 2 - 1
        yl, yh = dtcwt_model(lr)
        out = encoder(yh, yl)

        # Forced identity tensors of the same shape.
        for (gkey, bkey, rel_g, rel_b) in [
            ("low_gamma",  "low_beta",  rel_diff_g_low,  rel_diff_b_low),
            ("high_gamma", "high_beta", rel_diff_g_high, rel_diff_b_high),
        ]:
            g_trained = out[gkey].float()
            b_trained = out[bkey].float()
            g_identity = torch.ones_like(g_trained)
            b_identity = torch.zeros_like(b_trained)
            # Relative L2 difference (signal-magnitude normalised).
            denom_g = g_trained.norm().item() + 1e-12
            denom_b = b_trained.norm().item() + 1e-12
            rel_g.append((g_trained - g_identity).norm().item() / denom_g)
            rel_b.append((b_trained - b_identity).norm().item() / denom_b)

    summary = {
        "low_gamma  ||trained - identity|| / ||trained||": {
            "mean":   statistics.mean(rel_diff_g_low),
            "median": statistics.median(rel_diff_g_low),
            "max":    max(rel_diff_g_low),
        },
        "low_beta   ||trained - identity|| / ||trained||": {
            "mean":   statistics.mean(rel_diff_b_low),
            "median": statistics.median(rel_diff_b_low),
            "max":    max(rel_diff_b_low),
        },
        "high_gamma ||trained - identity|| / ||trained||": {
            "mean":   statistics.mean(rel_diff_g_high),
            "median": statistics.median(rel_diff_g_high),
            "max":    max(rel_diff_g_high),
        },
        "high_beta  ||trained - identity|| / ||trained||": {
            "mean":   statistics.mean(rel_diff_b_high),
            "median": statistics.median(rel_diff_b_high),
            "max":    max(rel_diff_b_high),
        },
    }

    print("\n  Relative distance of trained (gamma, beta) from identity:")
    for k, v in summary.items():
        print(f"    {k}")
        for kk, vv in v.items():
            print(f"      {kk:<8s}  {vv:.6f}")
    return summary


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
    p.add_argument("--inject_high", type=int, default=320)
    p.add_argument("--inject_low", type=int, default=320)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", type=Path, default=Path("results_identity_init"))
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("[init] loading encoder + DT-CWT ...")
    encoder = load_encoder(args.ckpt, args.inject_high, args.inject_low, device)
    dtcwt_model = DTCWTForward(J=4, biort="near_sym_a", qshift="qshift_a").to(device).eval()
    for q in dtcwt_model.parameters():
        q.requires_grad_(False)

    clip_dirs = find_clip_dirs(args.reds_lr_root, args.clips)
    all_frames = []
    for cd in clip_dirs:
        all_frames.append(load_clip(cd, args.max_frames, args.crop))
    if not all_frames:
        sys.exit("No clips found.")
    frames = torch.cat(all_frames, dim=0)
    print(f"[init] {len(frames)} frames from {len(clip_dirs)} clips")

    results = {
        "M1_identity_deviation": measure_identity_deviation(frames, encoder, dtcwt_model, device),
        "M2_counterfactual_identity": counterfactual_identity(frames, encoder, dtcwt_model, device),
    }

    out_json = args.output_dir / "identity_init_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[done] {out_json}")


if __name__ == "__main__":
    main()
