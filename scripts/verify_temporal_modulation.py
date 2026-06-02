"""Show that SFT produces temporally stable (gamma, beta) across
consecutive frames -- the direct visual + numerical temporal evidence.

We have ablation (w/o SFT: tLPIPS 25.18 vs Full 15.25), K=0.76 Lipschitz,
and stage-wise visualisations of single-frame behaviour. What's missing
for the defense is a side-by-side visualisation of (gamma_n, beta_n)
across consecutive frames, plus a per-pair quantitative trajectory of
relative change.

Outputs

  Numerical
    For every adjacent frame pair (n, n-1), report
        ||delta_gamma|| / ||gamma_n||      (relative change of gamma)
        ||delta_beta||  / ||beta_n||
        ||delta_LR||    / ||LR_n||         (relative change of LR)
    Aggregated per LOW/HIGH branch. Small ratios = stable across time.

  Visual
    For one clip, save a single grid PNG of T consecutive frames:
        row k = frame k
        columns = LR, gamma_LOW, beta_LOW, gamma_HIGH, beta_HIGH
    Each heatmap is min-max normalised globally per column so adjacent
    rows are visually comparable. If SFT is temporally stable the
    columns look like soft animations of the same pattern.

  Trajectory plot
    For each clip, x = frame index, y = ||delta(gamma,beta)|| / ||(gamma,beta)||
    Plotted alongside the ||delta_LR|| / ||LR|| reference. Curves stay
    flat and low if SFT preserves frame-to-frame stability.

Usage
-----
    PYTHONPATH=. python scripts/verify_temporal_modulation.py \\
        --ckpt /home/yjcho/StableVSR/experiments/20260430_dualsft/checkpoint-20000/ \\
        --reds_lr_root /mnt/HDD_raid1/yjcho/20260430/reds/ \\
        --clips 000 011 015 020 \\
        --viz_clip 011 --viz_n_frames 8 \\
        --max_frames 30 \\
        --inject_high 256 --inject_low 256 \\
        --output_dir results_temporal_modulation/
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
# Numerical: per-pair relative change per branch
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_pair_changes(clip_frames, encoder, dtcwt_model, device) -> List[Dict]:
    """For each consecutive frame pair n / n-1, return relative changes."""
    records = []
    cached_outs = []
    for k in range(clip_frames.shape[0]):
        lr = clip_frames[k:k + 1].to(device).float() * 2 - 1
        yl, yh = dtcwt_model(lr)
        out = encoder(yh, yl)
        cached_outs.append({k_: v.detach() for k_, v in out.items()})

    for n in range(1, len(cached_outs)):
        rec: Dict = {"frame": n}
        # LR relative change (raw, no flow-comp -- we test SFT stability
        # against raw LR similarity, the Lipschitz claim doesn't need flow).
        lr_n = (clip_frames[n].to(device).float() * 2 - 1).flatten()
        lr_p = (clip_frames[n - 1].to(device).float() * 2 - 1).flatten()
        rec["rel_LR"] = ((lr_n - lr_p).norm() / lr_n.norm().clamp(min=1e-12)).item()

        for branch, gkey, bkey in [
            ("LOW",  "low_gamma",  "low_beta"),
            ("HIGH", "high_gamma", "high_beta"),
        ]:
            g_n, g_p = cached_outs[n][gkey].float(), cached_outs[n - 1][gkey].float()
            b_n, b_p = cached_outs[n][bkey].float(), cached_outs[n - 1][bkey].float()
            rec[f"rel_gamma_{branch}"] = (
                (g_n - g_p).norm() / g_n.norm().clamp(min=1e-12)
            ).item()
            rec[f"rel_beta_{branch}"] = (
                (b_n - b_p).norm() / b_n.norm().clamp(min=1e-12)
            ).item()
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Visualisation: T frames x 5 columns grid
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_consecutive_grid(clip_frames, encoder, dtcwt_model, device, save_path: Path,
                          n_frames: int):
    """Grid of T rows x 5 cols: LR | gamma_L | beta_L | gamma_H | beta_H."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[viz] matplotlib missing")
        return

    n_frames = min(n_frames, clip_frames.shape[0])
    outs = []
    for k in range(n_frames):
        lr = clip_frames[k:k + 1].to(device).float() * 2 - 1
        yl, yh = dtcwt_model(lr)
        out = encoder(yh, yl)
        outs.append({k_: v.detach().cpu() for k_, v in out.items()})

    # Build per-column tensor stacks so we can normalise per column.
    def to_2d_channel_mean(t):
        return t.float().mean(dim=1).squeeze(0)

    cols = {
        "LR":      [(clip_frames[k] * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu()
                    for k in range(n_frames)],
        "gamma_L": [to_2d_channel_mean(outs[k]["low_gamma"])  for k in range(n_frames)],
        "beta_L":  [to_2d_channel_mean(outs[k]["low_beta"])   for k in range(n_frames)],
        "gamma_H": [to_2d_channel_mean(outs[k]["high_gamma"]) for k in range(n_frames)],
        "beta_H":  [to_2d_channel_mean(outs[k]["high_beta"])  for k in range(n_frames)],
    }

    # Global min/max per column for fair comparison across rows.
    col_ranges = {}
    for name, lst in cols.items():
        if name == "LR":
            continue
        stack = torch.stack(lst)
        col_ranges[name] = (stack.min().item(), stack.max().item())

    fig, axs = plt.subplots(n_frames, 5, figsize=(15, n_frames * 2.5))
    if n_frames == 1:
        axs = [axs]

    col_titles = list(cols.keys())
    for r in range(n_frames):
        for c, name in enumerate(col_titles):
            ax = axs[r][c]
            if name == "LR":
                ax.imshow(cols[name][r].numpy())
            else:
                lo, hi = col_ranges[name]
                ax.imshow(cols[name][r].numpy(), cmap="viridis", vmin=lo, vmax=hi)
            ax.axis("off")
            if r == 0:
                ax.set_title(name, fontsize=10)
            if c == 0:
                ax.text(-0.1, 0.5, f"frame {r}", transform=ax.transAxes,
                        rotation=90, va="center", ha="right", fontsize=9)

    fig.suptitle("Consecutive frames vs SFT (gamma, beta) -- "
                 "rows should look near-identical if SFT is temporally stable",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    print(f"[viz] saved {save_path}")


# ---------------------------------------------------------------------------
# Trajectory plot per clip
# ---------------------------------------------------------------------------

def save_trajectory_plot(records_per_clip: Dict[str, List[Dict]], save_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not records_per_clip:
        return

    n_clips = len(records_per_clip)
    fig, axs = plt.subplots(n_clips, 1, figsize=(10, 3 * n_clips), sharex=True)
    if n_clips == 1:
        axs = [axs]

    for ax, (clip, recs) in zip(axs, records_per_clip.items()):
        xs = [r["frame"] for r in recs]
        ax.plot(xs, [r["rel_LR"] for r in recs], "k-",  label="rel ΔLR (input)", linewidth=1)
        ax.plot(xs, [r["rel_gamma_LOW"] for r in recs],  label="rel Δγ_LOW",  alpha=0.85)
        ax.plot(xs, [r["rel_beta_LOW"]  for r in recs],  label="rel Δβ_LOW",  alpha=0.85)
        ax.plot(xs, [r["rel_gamma_HIGH"] for r in recs], label="rel Δγ_HIGH", alpha=0.85)
        ax.plot(xs, [r["rel_beta_HIGH"]  for r in recs], label="rel Δβ_HIGH", alpha=0.85)
        ax.set_title(f"clip {clip}", fontsize=10)
        ax.set_ylabel("relative change")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=5, loc="upper right")
    axs[-1].set_xlabel("frame index n (vs n-1)")
    fig.suptitle("Per-frame relative change of SFT (γ, β) and input LR "
                 "(small + flat = temporal stability)", fontsize=11)
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
    p.add_argument("--viz_clip", default="011",
                   help="Which clip to render the consecutive-frame grid for")
    p.add_argument("--viz_n_frames", type=int, default=8)
    p.add_argument("--crop", type=int, default=128)
    p.add_argument("--inject_high", type=int, default=320)
    p.add_argument("--inject_low", type=int, default=320)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", type=Path, default=Path("results_temporal_modulation"))
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

    per_clip_records: Dict[str, List[Dict]] = {}
    for clip_name, clip_dir in clip_dirs.items():
        print(f"\n=== {clip_name} ===")
        frames = load_clip(clip_dir, args.max_frames, args.crop)
        recs = compute_pair_changes(frames, encoder, dtcwt_model, device)
        per_clip_records[clip_name] = recs

        if clip_name == args.viz_clip:
            save_consecutive_grid(
                frames, encoder, dtcwt_model, device,
                save_path=args.output_dir / f"grid_{clip_name}.png",
                n_frames=args.viz_n_frames,
            )

    # Aggregate stats.
    print("\n=== Per-branch relative change (aggregated) ===")
    fields = ["rel_LR", "rel_gamma_LOW", "rel_beta_LOW",
              "rel_gamma_HIGH", "rel_beta_HIGH"]
    summary = {}
    all_recs = [r for recs in per_clip_records.values() for r in recs]
    for f in fields:
        vals = [r[f] for r in all_recs]
        summary[f] = {
            "mean":   statistics.mean(vals),
            "median": statistics.median(vals),
            "p10":    sorted(vals)[int(0.10 * len(vals))],
            "p90":    sorted(vals)[int(0.90 * len(vals))],
            "n":      len(vals),
        }
        print(f"  {f:<18s}  median={summary[f]['median']:.4f}  "
              f"mean={summary[f]['mean']:.4f}  "
              f"p10={summary[f]['p10']:.4f}  p90={summary[f]['p90']:.4f}  "
              f"(n={summary[f]['n']})")

    save_trajectory_plot(per_clip_records,
                          save_path=args.output_dir / "trajectories.png")

    out_json = args.output_dir / "temporal_modulation_records.json"
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "per_clip": per_clip_records},
                  f, indent=2, default=str)
    print(f"\n[done] {out_json}")


if __name__ == "__main__":
    main()
