"""Empirical verification of the *role* SFT plays inside the U-Net.

Tests three specific claims about Dual-SFT's contribution that go
beyond the Phase-1 Lipschitz measurement:

  E1. VAE loses high-frequency content (the problem SFT addresses).
      For each LR frame, run the round trip
          x_HR_bicubic ─VAE.encode─ z ─VAE.decode─ x_HR_recon
      and measure per-level relative DT-CWT magnitude loss
          ||M_j[x_HR_bicubic] - M_j[x_HR_recon]|| / ||M_j[x_HR_bicubic]||
      Expected: monotone decreasing in j (finest j=1 worst, j=4 best).
      If true, this defines the *informational gap* that SFT later fills.

  E2. SFT carries that high-frequency information back into U-Net.
      For each LR frame, compute
          H_LR(x, y) := sum_d M_{j=1,d}(x, y)   (DT-CWT high-freq energy map)
          H_SFT(x, y) := mean_c |gamma_H(c, x, y) - 1| + |beta_H(c, x, y)|
      and report the Pearson spatial correlation r(H_LR, H_SFT).
      Expected: r > 0 (well above a shuffled-baseline control), meaning
      regions where LR has more oriented high-freq energy are exactly
      where the SFT modulation deviates most from identity.

  E3. SFT influence propagates through U-Net depth (not localized).
      Sample noise z_t at a fixed timestep, run U-Net twice:
          h_with    = features when current_low/high = real (gamma, beta)
          h_without = features when current_low/high = (1, 0) identity
      Hook every down_blocks[i], mid_block, up_blocks[i] output and
      compute the relative deviation
          delta_i = ||h_with_i - h_without_i|| / ||h_with_i||
      Plot delta_i vs layer index.
      Expected pattern: 0 at down[0] (before LOW injection), spike at
      down[1] (LOW injection), persists through bottleneck and all
      up_blocks (deep propagation), bigger spike at up[1] (HIGH
      injection), 0 at up[3] only if HIGH effect already absorbed.

Outputs
-------
    results_sft_role/
        e1_vae_highfreq_loss.json
        e2_spatial_correlation.json
        e2_spatial_panels.png        (LR | M_DTCWT_j1 | |gamma_H - 1|)
        e3_propagation.json
        e3_propagation_depth.png     (bar chart per layer)

Usage
-----
    PYTHONPATH=. python scripts/verify_sft_role.py \\
        --ckpt          /home/yjcho/StableVSR/experiments/20260430_dualsft/checkpoint-20000/ \\
        --reds_lr_root  /mnt/HDD_raid1/yjcho/20260430/reds/ \\
        --pretrained_model claudiom4sir/StableVSR \\
        --clips         000 011 015 020 \\
        --max_frames    20 \\
        --inject_high   256 --inject_low 256 \\
        --output_dir    results_sft_role/

Pass --skip_e1, --skip_e2, --skip_e3 to selectively disable experiments.
E3 is the most defense-relevant; if time-constrained, run with
--skip_e1 --skip_e2.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

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
# Shared utilities
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
# E1. VAE high-frequency loss
# ---------------------------------------------------------------------------

@torch.no_grad()
def e1_vae_highfreq_loss(frames, vae, dtcwt_model, device) -> Dict:
    """Round-trip LR through VAE and quantify per-level magnitude loss.

    Runs at HR (4x bicubic upscaled) which is the VAE's native scale.
    Returns mean relative L2 loss per DT-CWT level.
    """
    print("\n=== E1: VAE high-frequency loss ===")
    vae_dtype = next(vae.parameters()).dtype
    per_level = {1: [], 2: [], 3: [], 4: []}
    for k in range(frames.shape[0]):
        lr = frames[k:k + 1].to(device).float() * 2 - 1
        hr = F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False)
        z = vae.encode(hr.to(vae_dtype)).latent_dist.sample() * vae.config.scaling_factor
        hr_recon = vae.decode(z / vae.config.scaling_factor).sample.float()
        _, yh_orig = dtcwt_model(hr)
        _, yh_recon = dtcwt_model(hr_recon)
        for j, (ya, yb) in enumerate(zip(yh_orig, yh_recon)):
            Ma = torch.sqrt(ya[..., 0] ** 2 + ya[..., 1] ** 2 + 1e-8)
            Mb = torch.sqrt(yb[..., 0] ** 2 + yb[..., 1] ** 2 + 1e-8)
            rel_loss = ((Ma - Mb).norm() / Ma.norm()).item()
            per_level[j + 1].append(rel_loss)
    summary = {
        f"j={j} (relative magnitude loss)": {
            "mean": statistics.mean(v),
            "median": statistics.median(v),
            "std": statistics.stdev(v) if len(v) > 1 else 0.0,
            "n": len(v),
        }
        for j, v in per_level.items()
    }
    for k, v in summary.items():
        print(f"  {k}: mean={v['mean']:.4f}  median={v['median']:.4f}  "
              f"std={v['std']:.4f}  (n={v['n']})")
    return summary


# ---------------------------------------------------------------------------
# E2. Spatial correlation: DT-CWT high-freq vs SFT modulation deviation
# ---------------------------------------------------------------------------

@torch.no_grad()
def e2_spatial_correlation(frames, encoder, dtcwt_model, device, save_panel: Path) -> Dict:
    """Pearson r between DT-CWT j=1 energy map and |gamma_H - 1| + |beta_H|.

    Both maps are reduced to 2D by summing over channels/directions, then
    resampled to a common spatial size before correlation.  Also saves a
    visual panel for the first frame: LR | M_DTCWT(j=1) | SFT deviation.
    """
    print("\n=== E2: spatial correlation (high-freq map vs SFT deviation) ===")
    rs, rs_shuf = [], []
    panel_saved = False

    for k in range(frames.shape[0]):
        lr = frames[k:k + 1].to(device).float() * 2 - 1
        yl, yh = dtcwt_model(lr)
        out = encoder(yh, yl)

        gamma_H = out["high_gamma"]
        beta_H = out["high_beta"]

        # DT-CWT j=1 highest-freq energy map: sum |Y_{j=1,d}| over RGB & 6 dirs.
        M1 = torch.sqrt(yh[0][..., 0] ** 2 + yh[0][..., 1] ** 2 + 1e-8)
        H_LR = M1.sum(dim=(1, 2))  # [1, H_j1, W_j1]

        # SFT deviation map: |gamma_H - 1| + |beta_H| averaged over channels.
        H_SFT = (gamma_H - 1).abs().mean(dim=1, keepdim=True) + \
                beta_H.abs().mean(dim=1, keepdim=True)
        H_SFT = H_SFT.squeeze(1)  # [1, H, W]

        # Resample to common size (smaller one).
        target_h = min(H_LR.shape[-2], H_SFT.shape[-2])
        target_w = min(H_LR.shape[-1], H_SFT.shape[-1])
        H_LR_r = F.interpolate(H_LR.unsqueeze(1).float(), size=(target_h, target_w),
                               mode="bilinear", align_corners=False).flatten()
        H_SFT_r = F.interpolate(H_SFT.unsqueeze(1).float(), size=(target_h, target_w),
                                mode="bilinear", align_corners=False).flatten()

        # Pearson r.
        ah = H_LR_r - H_LR_r.mean()
        bh = H_SFT_r - H_SFT_r.mean()
        denom = ah.norm() * bh.norm() + 1e-12
        r = (ah * bh).sum() / denom
        rs.append(r.item())

        # Shuffled control.
        perm = torch.randperm(H_SFT_r.numel(), device=device)
        bh_s = H_SFT_r[perm] - H_SFT_r.mean()
        r_s = (ah * bh_s).sum() / (ah.norm() * bh_s.norm() + 1e-12)
        rs_shuf.append(r_s.item())

        if not panel_saved:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(((lr + 1) / 2).squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu())
                axs[0].set_title("LR")
                axs[0].axis("off")
                axs[1].imshow(H_LR.squeeze(0).cpu(), cmap="hot")
                axs[1].set_title("DT-CWT j=1 high-freq energy")
                axs[1].axis("off")
                axs[2].imshow(H_SFT.squeeze(0).cpu(), cmap="hot")
                axs[2].set_title("|gamma_H - 1| + |beta_H|")
                axs[2].axis("off")
                fig.tight_layout()
                fig.savefig(save_panel, dpi=140)
                plt.close(fig)
                panel_saved = True
                print(f"  panel saved -> {save_panel}")
            except ImportError:
                pass

    summary = {
        "pearson_r_LRhighfreq_vs_SFTdeviation": {
            "mean": statistics.mean(rs),
            "median": statistics.median(rs),
            "std": statistics.stdev(rs) if len(rs) > 1 else 0.0,
            "n": len(rs),
        },
        "pearson_r_shuffled_control": {
            "mean": statistics.mean(rs_shuf),
            "median": statistics.median(rs_shuf),
            "std": statistics.stdev(rs_shuf) if len(rs_shuf) > 1 else 0.0,
            "n": len(rs_shuf),
        },
    }
    print(f"  Pearson r (signal)   median = {summary['pearson_r_LRhighfreq_vs_SFTdeviation']['median']:+.4f}")
    print(f"  Pearson r (shuffled) median = {summary['pearson_r_shuffled_control']['median']:+.4f}")
    return summary


# ---------------------------------------------------------------------------
# E3. Feature propagation depth
# ---------------------------------------------------------------------------

class FeatureCapture:
    """Register forward hooks on every down/mid/up block of a U-Net.

    Stores the *first* tensor element of each block's output (UNet2D
    blocks return tuples; the primary feature is at index 0).
    """
    def __init__(self, unet):
        self.unet = unet
        self.feats: Dict[str, torch.Tensor] = {}
        self.handles = []
        for i, blk in enumerate(unet.down_blocks):
            self.handles.append(blk.register_forward_hook(self._make_hook(f"down[{i}]")))
        self.handles.append(unet.mid_block.register_forward_hook(self._make_hook("mid")))
        for i, blk in enumerate(unet.up_blocks):
            self.handles.append(blk.register_forward_hook(self._make_hook(f"up[{i}]")))

    def _make_hook(self, name):
        def hook(_module, _inp, output):
            t = output[0] if isinstance(output, tuple) else output
            self.feats[name] = t.detach()
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def e3_propagation_depth(
    frames, encoder, dtcwt_model, vae, unet, weight_dtype, device, save_plot: Path,
) -> Dict:
    """For each frame, compare U-Net intermediate features with vs without
    SFT modulation and report relative deviation per layer."""
    print("\n=== E3: SFT effect propagation through U-Net depth ===")
    unet_with_sft = UNetWithDualSFT(unet)
    unet_with_sft.cond_encoder = encoder
    unet_with_sft.eval()

    per_layer_deltas: Dict[str, List[float]] = {}
    layer_order: List[str] = []

    # Dummy text condition (the model was trained with empty prompts).
    text_dim = unet.config.cross_attention_dim
    enc_hidden = torch.zeros(1, 77, text_dim, device=device, dtype=weight_dtype)
    # ControlNet residuals are absent here; this is a pure "what does SFT
    # change in U-Net features?" experiment.  noisy_latents_cat has
    # channels (latent + LR latent_like). The simplest comparable input is
    # to encode the LR and cat with random noise at the same shape.

    for k in range(frames.shape[0]):
        lr = frames[k:k + 1].to(device).float() * 2 - 1
        lr_hr = F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False)
        z0 = vae.encode(lr_hr.to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(z0)
        timesteps = torch.tensor([500], device=device, dtype=torch.long)
        # Match train.py's input: noisy latent (4 ch) + LR image (3 ch) = 7 ch.
        # The modified U-Net's conv_in expects this 7-channel layout.
        # LR must be resampled to the latent spatial size for concat.
        lr_for_cat = F.interpolate(lr, size=z0.shape[-2:],
                                   mode="bilinear",
                                   align_corners=False).to(weight_dtype)
        noisy_cat = torch.cat([noise, lr_for_cat], dim=1)

        # ---- With SFT ----
        cap_w = FeatureCapture(unet)
        yl, yh = dtcwt_model(lr.float())
        cond = encoder(yh, yl)
        unet_with_sft.current_low = (
            cond["low_gamma"].to(weight_dtype), cond["low_beta"].to(weight_dtype),
        )
        unet_with_sft.current_high = (
            cond["high_gamma"].to(weight_dtype), cond["high_beta"].to(weight_dtype),
        )
        try:
            _ = unet_with_sft.unet(noisy_cat, timesteps, encoder_hidden_states=enc_hidden)
        finally:
            unet_with_sft.current_low = None
            unet_with_sft.current_high = None
        feats_w = {n: t.clone() for n, t in cap_w.feats.items()}
        cap_w.remove()

        # ---- Without SFT (identity gamma=1, beta=0) ----
        cap_wo = FeatureCapture(unet)
        # Hooks installed; with current_low/high = None, UNetWithDualSFT's
        # modulate path passes through unchanged.
        _ = unet_with_sft.unet(noisy_cat, timesteps, encoder_hidden_states=enc_hidden)
        feats_wo = {n: t.clone() for n, t in cap_wo.feats.items()}
        cap_wo.remove()

        for name in feats_w.keys():
            if name not in layer_order:
                layer_order.append(name)
            h_w = feats_w[name].float()
            h_wo = feats_wo[name].float()
            denom = h_w.norm().item() + 1e-12
            delta = (h_w - h_wo).norm().item() / denom
            per_layer_deltas.setdefault(name, []).append(delta)

    # Aggregate.
    summary = {}
    for name in layer_order:
        v = per_layer_deltas[name]
        summary[name] = {
            "mean": statistics.mean(v),
            "median": statistics.median(v),
            "std": statistics.stdev(v) if len(v) > 1 else 0.0,
            "n": len(v),
        }

    print("\n  Relative feature deviation ||h_with - h_without|| / ||h_with||:")
    for name in layer_order:
        v = summary[name]
        bar = "#" * int(v["mean"] * 100)
        print(f"    {name:<10s} mean={v['mean']:.4f}  {bar}")

    # Plot bar chart.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = layer_order
        means = [summary[n]["mean"] for n in names]
        stds = [summary[n]["std"] for n in names]
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ["#1f77b4"] * len(names)
        for i, n in enumerate(names):
            if "down[1]" in n:
                colors[i] = "#d62728"   # LOW injection
            elif "up[1]" in n:
                colors[i] = "#2ca02c"   # HIGH injection
        ax.bar(range(len(names)), means, yerr=stds, color=colors, alpha=0.85)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45)
        ax.set_ylabel("||h_with_SFT - h_without_SFT|| / ||h_with_SFT||")
        ax.set_title("SFT effect propagation through U-Net depth\n"
                     "red=LOW injection (down[1]),  green=HIGH injection (up[1])")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_plot, dpi=140)
        plt.close(fig)
        print(f"\n  plot saved -> {save_plot}")
    except ImportError:
        pass

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--reds_lr_root", required=True, type=Path)
    p.add_argument("--pretrained_model", required=True,
                   help="Base SD model path; used for VAE (E1) and U-Net (E3).")
    p.add_argument("--clips", nargs="+", default=["000", "011", "015", "020"])
    p.add_argument("--max_frames", type=int, default=20)
    p.add_argument("--crop", type=int, default=128)
    p.add_argument("--inject_high", type=int, default=320)
    p.add_argument("--inject_low", type=int, default=320)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", type=Path, default=Path("results_sft_role"))
    p.add_argument("--skip_e1", action="store_true")
    p.add_argument("--skip_e2", action="store_true")
    p.add_argument("--skip_e3", action="store_true")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    # Use fp32 throughout this diagnostic script. fp16 VAE round-trip in E1
    # underflows on small LR patches and propagates NaN through DT-CWT.
    # The script processes batch=1 small crops, so the memory cost is fine.
    weight_dtype = torch.float32

    print("[init] loading models ...")
    encoder = load_encoder(args.ckpt, args.inject_high, args.inject_low, device)
    dtcwt_model = DTCWTForward(J=4, biort="near_sym_a", qshift="qshift_a").to(device).eval()
    for q in dtcwt_model.parameters():
        q.requires_grad_(False)

    vae = unet = None
    if not args.skip_e1 or not args.skip_e3:
        from diffusers import AutoencoderKL, UNet2DConditionModel
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model, subfolder="vae", torch_dtype=weight_dtype,
        ).to(device).eval()
    if not args.skip_e3:
        from diffusers import UNet2DConditionModel
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model, subfolder="unet", torch_dtype=weight_dtype,
        ).to(device).eval()

    # Aggregate frames across clips.
    clip_dirs = find_clip_dirs(args.reds_lr_root, args.clips)
    all_frames = []
    for cd in clip_dirs:
        all_frames.append(load_clip(cd, args.max_frames, args.crop))
    if not all_frames:
        sys.exit("No clips found.")
    frames = torch.cat(all_frames, dim=0)
    print(f"[init] loaded {len(frames)} frames from {len(clip_dirs)} clips")

    results = {}

    if not args.skip_e1:
        results["E1"] = e1_vae_highfreq_loss(frames, vae, dtcwt_model, device)
        with open(args.output_dir / "e1_vae_highfreq_loss.json", "w") as f:
            json.dump(results["E1"], f, indent=2)

    if not args.skip_e2:
        results["E2"] = e2_spatial_correlation(
            frames, encoder, dtcwt_model, device,
            save_panel=args.output_dir / "e2_spatial_panels.png",
        )
        with open(args.output_dir / "e2_spatial_correlation.json", "w") as f:
            json.dump(results["E2"], f, indent=2)

    if not args.skip_e3:
        results["E3"] = e3_propagation_depth(
            frames, encoder, dtcwt_model, vae, unet, weight_dtype, device,
            save_plot=args.output_dir / "e3_propagation_depth.png",
        )
        with open(args.output_dir / "e3_propagation.json", "w") as f:
            json.dump(results["E3"], f, indent=2)

    with open(args.output_dir / "all_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] all results in {args.output_dir}")


if __name__ == "__main__":
    main()
