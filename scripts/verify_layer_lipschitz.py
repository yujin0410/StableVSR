"""Decompose PerLevelProcessor's Lipschitz contribution layer by layer.

After Phase 1 (overall K_encoder = 0.76) and the identity-init check
(LOW gamma near identity, HIGH gamma at 0.78), the remaining question
is: which layers inside PerLevelProcessor actually bound the function?

Two measurements:

  W1. Weight spectral norms.
      For each Conv2d in PerLevelProcessor (+ SFTHead conv1/conv2),
      compute the largest singular value of the (n_out, n_in*kH*kW)
      flattened weight matrix. This is the layer's Lipschitz upper
      bound (assuming unit stride, ignoring padding effects).
      Combined with SiLU (~1.1) and CBAM sigmoid (≤1), the product
      is a theoretical Lipschitz upper bound on PerLevelProcessor.

  L1. Empirical per-layer Lipschitz on real REDS frame pairs.
      Register forward hooks on every Conv2d / SiLU / SpatialAttention
      output. For each consecutive frame pair, measure
        K_layer := ||Δh_out|| / ||Δh_in||
      averaged over frame pairs. Identifies where the smoothness comes
      from in the actual learned pipeline.

Together W1 and L1 answer "is the K=0.76 bound a property of the
architecture (theoretical) and / or the learned weights (empirical)?"

Usage
-----
    PYTHONPATH=. python scripts/verify_layer_lipschitz.py \\
        --ckpt /home/yjcho/StableVSR/experiments/20260430_dualsft/checkpoint-20000/ \\
        --reds_lr_root /mnt/HDD_raid1/yjcho/20260430/reds/ \\
        --clips 000 011 015 020 \\
        --max_frames 30 \\
        --inject_high 256 --inject_low 256 \\
        --output_dir results_layer_lipschitz/
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
# Shared helpers (same as other scripts)
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
# W1. Weight spectral norms
# ---------------------------------------------------------------------------

@torch.no_grad()
def w1_spectral_norms(encoder) -> Dict:
    """Spectral norm of every Conv2d weight inside the encoder.

    Returns a flat list of (path, n_in, n_out, kH, kW, spectral_norm).
    Also reports the per-PerLevelProcessor product as an upper bound
    on that processor's Lipschitz constant.
    """
    print("\n=== W1: weight spectral norms (per-layer Lipschitz upper bounds) ===")

    per_proc_products: Dict[str, float] = {}
    rows = []
    for name, mod in encoder.named_modules():
        if not isinstance(mod, nn.Conv2d):
            continue
        W = mod.weight.detach().float()
        # Flatten to (n_out, n_in * kH * kW) and take its largest SV.
        Wf = W.reshape(W.shape[0], -1)
        sv = torch.linalg.svdvals(Wf)
        s_max = sv[0].item()
        rows.append({
            "path": name,
            "shape": tuple(W.shape),
            "groups": mod.groups,
            "spectral_norm": s_max,
            "frobenius_norm": W.norm().item(),
        })
        # Identify which PerLevelProcessor this conv belongs to.
        proc = name.split(".")[0]    # e.g. "proc_j1"
        per_proc_products.setdefault(proc, 1.0)
        per_proc_products[proc] *= s_max

    # Heuristic upper-bound product including activation Lipschitz constants.
    # PerLevelProcessor pipeline per pathway:
    #   DWConv -> SiLU (1.1) -> SpatialAttention (sigmoid <= 1) -> PWConv
    # Plus SFTHead: Conv1 -> SiLU (1.1) -> Conv2.
    # Many SVs already small; this product is intentionally loose, not tight.
    silu_lip = 1.1
    cbam_sig_lip = 1.0    # sigmoid gating <= 1 multiplicatively
    structural_factor = silu_lip * cbam_sig_lip * silu_lip
    for k in per_proc_products:
        per_proc_products[k] *= structural_factor

    print(f"\n  Per-conv spectral norms ({len(rows)} convs):")
    for r in rows:
        print(f"    {r['path']:<35s} shape={r['shape']} groups={r['groups']:<2d}  "
              f"sigma_max={r['spectral_norm']:.4f}  ||W||_F={r['frobenius_norm']:.4f}")

    print(f"\n  Per-processor approximate Lipschitz upper bound (product):")
    for k, v in per_proc_products.items():
        print(f"    {k:<10s}  upper-bound K ≤ {v:.3f}")

    return {
        "per_layer": rows,
        "per_processor_upper_bound": per_proc_products,
    }


# ---------------------------------------------------------------------------
# L1. Empirical per-layer Lipschitz on real frame pairs
# ---------------------------------------------------------------------------

class LayerLipschitzProbe:
    """Hook every Conv2d / SiLU / SpatialAttention output in the encoder."""
    def __init__(self, encoder):
        self.handles = []
        self.in_outs: Dict[str, torch.Tensor] = {}
        for name, mod in encoder.named_modules():
            if isinstance(mod, (nn.Conv2d,)):
                self.handles.append(mod.register_forward_hook(self._mk(name + "[Conv]")))
            elif isinstance(mod, nn.SiLU):
                self.handles.append(mod.register_forward_hook(self._mk(name + "[SiLU]")))
            elif hasattr(mod, "conv") and "SpatialAttention" in type(mod).__name__:
                # CBAM SpatialAttention captured as a whole
                self.handles.append(mod.register_forward_hook(self._mk(name + "[CBAM_SA]")))

    def _mk(self, key):
        def hook(_m, inp, out):
            x_in = inp[0] if isinstance(inp, tuple) else inp
            self.in_outs[key] = (x_in.detach(), out.detach())
        return hook

    def reset(self):
        self.in_outs.clear()

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def l1_empirical_per_layer(frames, encoder, dtcwt_model, device) -> Dict:
    """Measure ||Δoutput|| / ||Δinput|| per layer on consecutive frames."""
    print("\n=== L1: empirical per-layer Lipschitz (Δout / Δin per frame pair) ===")

    probe = LayerLipschitzProbe(encoder)
    layer_K: Dict[str, List[float]] = {}
    layer_order: List[str] = []

    # Forward each frame, cache (in, out) per layer.
    cached = []
    for k in range(frames.shape[0]):
        lr = frames[k:k + 1].to(device).float() * 2 - 1
        yl, yh = dtcwt_model(lr)
        probe.reset()
        _ = encoder(yh, yl)
        # Detach-copy is enough; tensors are already detached in hook.
        cached.append({k_: (v[0].clone(), v[1].clone()) for k_, v in probe.in_outs.items()})
    probe.remove()

    # Compare frame n vs n-1.
    for n in range(1, len(cached)):
        for key in cached[n]:
            if key not in cached[n - 1]:
                continue
            in_n, out_n = cached[n][key]
            in_p, out_p = cached[n - 1][key]
            din = (in_n.float() - in_p.float()).norm().item()
            dout = (out_n.float() - out_p.float()).norm().item()
            if din < 1e-9:
                continue
            K = dout / din
            layer_K.setdefault(key, []).append(K)
            if key not in layer_order:
                layer_order.append(key)

    # Aggregate.
    summary = {}
    for key in layer_order:
        v = layer_K[key]
        if not v:
            continue
        summary[key] = {
            "median": statistics.median(v),
            "mean":   statistics.mean(v),
            "p10":    sorted(v)[int(0.10 * len(v))],
            "p90":    sorted(v)[int(0.90 * len(v))],
            "n":      len(v),
        }

    print("\n  Empirical per-layer Lipschitz on real LR pairs:")
    print(f"    {'Layer':<60s}  {'median':>8s}  {'p10':>8s}  {'p90':>8s}")
    for key in layer_order:
        v = summary.get(key)
        if v is None:
            continue
        marker = "  <-- amplifies" if v["median"] > 1.0 else "  <-- contracts"
        print(f"    {key:<60s}  {v['median']:>8.4f}  {v['p10']:>8.4f}  "
              f"{v['p90']:>8.4f}{marker}")
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
    p.add_argument("--output_dir", type=Path, default=Path("results_layer_lipschitz"))
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("[init] loading encoder + DT-CWT ...")
    encoder = load_encoder(args.ckpt, args.inject_high, args.inject_low, device)
    dtcwt_model = DTCWTForward(J=4, biort="near_sym_a", qshift="qshift_a").to(device).eval()
    for q in dtcwt_model.parameters():
        q.requires_grad_(False)

    results = {}

    # W1: weight-only, no data needed.
    results["W1_spectral_norms"] = w1_spectral_norms(encoder)

    # L1: needs real LR frames.
    clip_dirs = find_clip_dirs(args.reds_lr_root, args.clips)
    all_frames = []
    for cd in clip_dirs:
        all_frames.append(load_clip(cd, args.max_frames, args.crop))
    if not all_frames:
        sys.exit("No clips found.")
    frames = torch.cat(all_frames, dim=0)
    print(f"\n[init] {len(frames)} frames from {len(clip_dirs)} clips")

    results["L1_empirical_per_layer"] = l1_empirical_per_layer(
        frames, encoder, dtcwt_model, device
    )

    out_json = args.output_dir / "layer_lipschitz_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[done] {out_json}")


if __name__ == "__main__":
    main()
