"""Verify that each compression / amplification stage in
PerLevelProcessor performs *meaningful* signal processing rather than
arbitrary transformation.

The per-layer Lipschitz decomposition showed that PerLevelProcessor has
a 'compression then amplification' structure:
    DWConv -> SiLU -> CBAM SA -> PWConv  (contracts to ~0.16x)
    SFTHead conv1 -> SiLU -> conv2       (amplifies 3-8.7x)
This script asks: are the compressions preserving meaningful structure,
and is the amplification producing structured (γ, β) instead of noise?

Numerical metrics per stage (captured via forward hooks):

  - signal_energy        ||x||_2 / sqrt(N)      RMS amplitude
  - mean / std            x.mean(), x.std()
  - effective_rank        sum(s_i / s_max), s_i singular values
                          (information richness across channels)
  - spatial_entropy       Shannon entropy of normalised pixel histogram
  - corr_with_input       spatial Pearson r between this stage's
                          channel-mean map and the original LR
                          magnitude map (after appropriate resize).
                          Tracks 'did this stage preserve LR spatial
                          structure?'

Visualisation:

  For one representative LR frame, the script saves a side-by-side
  PNG showing the channel-mean spatial map of every captured stage
  (and the input LR for reference). Each map is min-max normalised
  per panel so structure is comparable.

Usage
-----
    PYTHONPATH=. python scripts/verify_stagewise_meaning.py \\
        --ckpt /home/yjcho/StableVSR/experiments/20260430_dualsft/checkpoint-20000/ \\
        --reds_lr_root /mnt/HDD_raid1/yjcho/20260430/reds/ \\
        --clips 000 011 015 020 \\
        --max_frames 20 \\
        --inject_high 256 --inject_low 256 \\
        --output_dir results_stagewise/
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from util.sft_utils import FrequencyConditioningEncoder
from util.frequency_utils import DTCWTForward


# ---------------------------------------------------------------------------
# Shared helpers
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
# Metrics
# ---------------------------------------------------------------------------

def _signal_energy(t: torch.Tensor) -> float:
    """RMS amplitude over all elements."""
    return (t.float().pow(2).mean().sqrt()).item()


def _effective_rank(t: torch.Tensor) -> float:
    """Effective rank: sum_i (s_i / s_max), s = SVs of channel-flattened map."""
    t = t.float()
    # Flatten to [C, H*W] (collapse batch on dim 0).
    if t.ndim == 4:
        t = t.reshape(t.shape[1], -1)
    elif t.ndim == 3:
        t = t.reshape(t.shape[0], -1)
    else:
        return float("nan")
    if t.shape[0] < 2 or t.shape[1] < 2:
        return 1.0
    s = torch.linalg.svdvals(t)
    if s[0] < 1e-12:
        return 0.0
    return (s / s[0]).sum().item()


def _spatial_entropy(t: torch.Tensor, bins: int = 32) -> float:
    """Shannon entropy of channel-mean spatial map, after min-max norm."""
    if t.ndim == 4:
        m = t.float().mean(dim=1).flatten()
    elif t.ndim == 3:
        m = t.float().mean(dim=0).flatten()
    else:
        return float("nan")
    if m.numel() == 0:
        return 0.0
    m_lo, m_hi = m.min(), m.max()
    if (m_hi - m_lo) < 1e-12:
        return 0.0
    m_norm = (m - m_lo) / (m_hi - m_lo)
    hist = torch.histc(m_norm, bins=bins, min=0.0, max=1.0)
    p = hist / hist.sum().clamp(min=1e-12)
    p = p.clamp(min=1e-12)
    return float(-(p * p.log()).sum().item())


def _spatial_corr_with_input(t: torch.Tensor, lr_map: torch.Tensor) -> float:
    """Pearson r between channel-mean spatial map of t and the LR mag map.

    Both maps are resized to a common spatial size first.
    """
    if t.ndim == 4:
        a = t.float().mean(dim=1)
    elif t.ndim == 3:
        a = t.float().mean(dim=0).unsqueeze(0)
    else:
        return float("nan")

    # a: [B, H, W],  lr_map: [B, H_lr, W_lr]
    target_h = min(a.shape[-2], lr_map.shape[-2])
    target_w = min(a.shape[-1], lr_map.shape[-1])
    a_r = F.interpolate(a.unsqueeze(1), size=(target_h, target_w),
                        mode="bilinear", align_corners=False).flatten()
    b_r = F.interpolate(lr_map.unsqueeze(1), size=(target_h, target_w),
                        mode="bilinear", align_corners=False).flatten()
    ah = a_r - a_r.mean()
    bh = b_r - b_r.mean()
    denom = ah.norm() * bh.norm() + 1e-12
    return float(((ah * bh).sum() / denom).item())


def stage_metrics(t: torch.Tensor, lr_map: torch.Tensor) -> Dict:
    return {
        "energy":        _signal_energy(t),
        "mean":          float(t.float().mean().item()),
        "std":           float(t.float().std().item()),
        "eff_rank":      _effective_rank(t),
        "spatial_entropy": _spatial_entropy(t),
        "corr_with_LR":  _spatial_corr_with_input(t, lr_map),
    }


# ---------------------------------------------------------------------------
# Hooked forward through one PerLevelProcessor
# ---------------------------------------------------------------------------

@torch.no_grad()
def capture_stages(proc, yh_j) -> Dict[str, torch.Tensor]:
    """Replay PerLevelProcessor.forward and capture intermediates.

    Returns dict {stage_name: tensor} for the magnitude pathway and the
    SFTHead pathway. The phase pathway is captured by symmetry but
    omitted from the visualisation to keep panels readable.
    """
    B, C, D, H, W, _ = yh_j.shape
    real = yh_j[..., 0]
    imag = yh_j[..., 1]
    M = torch.sqrt(real * real + imag * imag + 1e-8)
    sin_phi = imag / (M + 1e-8)
    cos_phi = real / (M + 1e-8)

    M_flat = M.reshape(B, C * D, H, W)
    sin_flat = sin_phi.reshape(B, C * D, H, W)
    cos_flat = cos_phi.reshape(B, C * D, H, W)

    if proc.target_size is not None and M_flat.shape[-2:] != tuple(proc.target_size):
        M_flat = F.interpolate(M_flat, size=proc.target_size,
                               mode='bilinear', align_corners=False)
        sin_flat = F.interpolate(sin_flat, size=proc.target_size,
                                 mode='bilinear', align_corners=False)
        cos_flat = F.interpolate(cos_flat, size=proc.target_size,
                                 mode='bilinear', align_corners=False)

    # Magnitude pathway, stage by stage.
    s0_input = M_flat                                # input to processor
    s1_dwconv = proc.M_dw(M_flat)                    # after DWConv
    s2_silu = F.silu(s1_dwconv)                      # after SiLU
    s3_cbam = proc.M_sa(s2_silu)                     # after CBAM SA
    s4_pwconv = proc.M_pw(s3_cbam)                   # final e_M

    # Phase pathway (for the polar recombination only).
    phase_in = torch.cat([sin_flat, cos_flat], dim=1)
    e_phase = F.silu(proc.phase_dw(phase_in))
    e_phase = proc.phase_sa(e_phase)
    e_phase = proc.phase_pw(e_phase)
    e_sin, e_cos = e_phase.chunk(2, dim=1)
    e_real = s4_pwconv * e_cos
    e_imag = s4_pwconv * e_sin
    combined = torch.cat([s4_pwconv, e_real, e_imag], dim=1)

    # SFTHead pathway for gamma.
    gh_conv1 = proc.sft_gamma.conv1(combined)
    gh_act = F.silu(gh_conv1)
    gh_conv2 = proc.sft_gamma.conv2(gh_act)          # final gamma

    # SFTHead pathway for beta.
    bh_conv1 = proc.sft_beta.conv1(combined)
    bh_act = F.silu(bh_conv1)
    bh_conv2 = proc.sft_beta.conv2(bh_act)           # final beta

    return {
        "0_input(M)":    s0_input,
        "1_DWConv":      s1_dwconv,
        "2_SiLU":        s2_silu,
        "3_CBAM_SA":     s3_cbam,
        "4_PWConv(e_M)": s4_pwconv,
        "5_SFThead_in":  combined,
        "6_gamma_conv1": gh_conv1,
        "7_gamma_SiLU":  gh_act,
        "8_gamma_OUT":   gh_conv2,
        "6_beta_conv1":  bh_conv1,
        "7_beta_SiLU":   bh_act,
        "8_beta_OUT":    bh_conv2,
    }


# ---------------------------------------------------------------------------
# Aggregation across frames
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_all_frames(frames, encoder, dtcwt_model, device, proc_name="proc_j2") -> Dict:
    """Aggregate stage metrics across many frames for the chosen processor."""
    print(f"\n=== Stage-wise metrics for {proc_name} ===")
    proc = getattr(encoder, proc_name)

    agg: Dict[str, Dict[str, List[float]]] = {}
    n_done = 0
    for k in range(frames.shape[0]):
        lr = frames[k:k + 1].to(device).float() * 2 - 1
        yl, yh = dtcwt_model(lr)
        j_index = int(proc_name.split("_j")[1]) - 1
        yh_j = yh[j_index]

        # Build a 2D LR magnitude map for correlation reference.
        m_input = (torch.sqrt(yh_j[..., 0] ** 2 + yh_j[..., 1] ** 2 + 1e-8)
                   .sum(dim=(1, 2)))  # [B, H, W]

        stages = capture_stages(proc, yh_j)
        for sname, t in stages.items():
            m = stage_metrics(t, m_input)
            for kk, vv in m.items():
                agg.setdefault(sname, {}).setdefault(kk, []).append(vv)
        n_done += 1

    # Summarise.
    summary = {}
    for sname, mm in agg.items():
        summary[sname] = {kk: {"mean": statistics.mean(vv),
                               "median": statistics.median(vv),
                               "std": statistics.stdev(vv) if len(vv) > 1 else 0.0,
                               "n": len(vv)}
                          for kk, vv in mm.items()}

    print(f"\n  Stage              energy   eff_rank  sp_entropy  corr(LR)  n")
    print(  "  ------------------------------------------------------------------")
    for sname, mm in summary.items():
        print(f"  {sname:<18s}  "
              f"{mm['energy']['mean']:7.3f}  "
              f"{mm['eff_rank']['mean']:7.2f}   "
              f"{mm['spatial_entropy']['mean']:7.3f}    "
              f"{mm['corr_with_LR']['mean']:+7.3f}   {mm['energy']['n']}")
    return summary


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_stage_visualisation(encoder, dtcwt_model, device, lr_frame, save_path: Path,
                              proc_name="proc_j2"):
    """Save a grid PNG of channel-mean spatial maps at every stage."""
    proc = getattr(encoder, proc_name)
    lr = lr_frame.unsqueeze(0).to(device).float() * 2 - 1
    yl, yh = dtcwt_model(lr)
    j_index = int(proc_name.split("_j")[1]) - 1
    yh_j = yh[j_index]
    stages = capture_stages(proc, yh_j)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[viz] matplotlib not available, skipping visualization")
        return

    # Add the raw LR for reference.
    lr_panel = ((lr.squeeze(0) + 1) / 2).permute(1, 2, 0).clamp(0, 1).cpu().numpy()

    stage_names = list(stages.keys())
    n_panels = len(stage_names) + 1
    n_cols = 4
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.5))
    axs = axs.flatten() if n_panels > 1 else [axs]

    axs[0].imshow(lr_panel)
    axs[0].set_title("LR (input)")
    axs[0].axis("off")

    for i, sname in enumerate(stage_names):
        t = stages[sname]
        if t.ndim == 4:
            m = t.float().mean(dim=1).squeeze(0).cpu().numpy()
        else:
            m = t.float().mean(dim=0).cpu().numpy()
        axs[i + 1].imshow(m, cmap="hot")
        # Brief shape annotation.
        shape_str = " × ".join(str(s) for s in t.shape[1:])
        axs[i + 1].set_title(f"{sname}\n[{shape_str}]", fontsize=9)
        axs[i + 1].axis("off")

    # Hide leftover axes.
    for j in range(n_panels, len(axs)):
        axs[j].axis("off")

    fig.suptitle(f"PerLevelProcessor stage-by-stage feature maps "
                 f"({proc_name})", fontsize=12)
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
    p.add_argument("--max_frames", type=int, default=20)
    p.add_argument("--crop", type=int, default=128)
    p.add_argument("--inject_high", type=int, default=320)
    p.add_argument("--inject_low", type=int, default=320)
    p.add_argument("--device", default="cuda")
    p.add_argument("--processors", nargs="+",
                   default=["proc_j1", "proc_j2", "proc_j3", "proc_j4"])
    p.add_argument("--viz_frame_idx", type=int, default=0,
                   help="Which loaded frame to visualise per processor")
    p.add_argument("--output_dir", type=Path, default=Path("results_stagewise"))
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

    all_results = {}
    for proc_name in args.processors:
        all_results[proc_name] = measure_all_frames(
            frames, encoder, dtcwt_model, device, proc_name=proc_name
        )
        save_stage_visualisation(
            encoder, dtcwt_model, device,
            frames[args.viz_frame_idx],
            save_path=args.output_dir / f"stagewise_{proc_name}.png",
            proc_name=proc_name,
        )

    out_json = args.output_dir / "stagewise_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[done] {out_json}")


if __name__ == "__main__":
    main()
