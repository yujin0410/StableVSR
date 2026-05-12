"""Generate two figures for the redesigned Motivation slide.

  motivation_flicker.png   — 3-frame texture flickering vs. stable comparison
                             (uses structured texture, not pure noise).
  motivation_vsr_task.png  — VSR task illustration:
                             LR sequence (3 blurry frames) → VSR → HR frame.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.ndimage import gaussian_filter  # available with scipy

OUT = Path("/tmp/thesis_work/fig_png")
OUT.mkdir(parents=True, exist_ok=True)

HIGH = "#2C55A8"
LOW = "#CB4235"
INK = "#121E3D"
GRAY = "#6B6B78"
WHITE = "#FFFFFF"


def make_texture(seed, base, jitter_scale=0.35):
    """Smooth base + small high-frequency perturbation per seed."""
    rng = np.random.default_rng(seed)
    hf = rng.normal(size=base.shape)
    hf = gaussian_filter(hf, sigma=0.6)  # not pure noise — sub-pixel structure
    out = base + jitter_scale * hf
    out = (out - out.min()) / (out.max() - out.min() + 1e-9)
    return out


def make_flicker():
    """Three frames with same low-frequency structure but different HF details.

    Top row (flicker): same base + different HF per frame → texture shimmer.
    Bottom row (stable): same base + same HF on every frame.
    """
    # Common smooth structure (gradient × sine pattern) — recognisable as
    # "the same scene" across frames.
    h, w = 64, 64
    yy, xx = np.mgrid[0:h, 0:w] / h
    base = (0.6 * np.sin(6 * xx + 1.2)
            + 0.4 * np.cos(5 * yy + 0.7)
            + 0.3 * np.sin(9 * (xx + yy)))
    base = gaussian_filter(base, sigma=0.4)

    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=200)
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 3.4)
    ax.set_aspect("equal")
    ax.axis("off")

    # Top row — flicker
    y_top = 2.0
    ax.text(0.05, y_top + 0.85,
            "Per-frame independent\nstochastic denoising",
            fontsize=9, color=LOW, fontweight="bold", va="top")
    stable_seed = 99
    for k, x in enumerate([1.7, 3.25, 4.8]):
        # Different HF per frame
        tex = make_texture(seed=10 + k, base=base, jitter_scale=0.55)
        ax.imshow(tex, extent=[x, x + 1.3, y_top, y_top + 1.3],
                  cmap="gray", aspect="auto", zorder=1, interpolation="bilinear")
        ax.add_patch(FancyBboxPatch(
            (x, y_top), 1.3, 1.3,
            boxstyle="round,pad=0.0,rounding_size=0.04",
            ec=LOW, fc="none", lw=1.6))
        ax.text(x + 0.65, y_top - 0.15, f"frame  t{['-1','','+1'][k]}",
                ha="center", fontsize=9, color=GRAY)
    # red wavy arrows
    for k in range(2):
        x0 = 1.7 + k * 1.55 + 1.3
        ax.annotate("", xy=(x0 + 0.22, y_top + 0.65),
                    xytext=(x0 + 0.02, y_top + 0.65),
                    arrowprops=dict(arrowstyle="-|>", color=LOW, lw=1.8))
    ax.text(6.15, y_top + 0.85, "different",
            fontsize=11, color=LOW, fontweight="bold", va="center")
    ax.text(6.15, y_top + 0.55, "HF detail",
            fontsize=10, color=LOW, va="center")
    ax.text(6.15, y_top + 0.20, "⇒  flicker",
            fontsize=11, color=LOW, fontweight="bold", va="center",
            style="italic")

    # Bottom row — stable
    y_bot = 0.20
    ax.text(0.05, y_bot + 0.85,
            "Frequency-conditioned\n(WC-BD-SFT)",
            fontsize=9, color=HIGH, fontweight="bold", va="top")
    tex_stable = make_texture(seed=stable_seed, base=base, jitter_scale=0.55)
    for k, x in enumerate([1.7, 3.25, 4.8]):
        ax.imshow(tex_stable, extent=[x, x + 1.3, y_bot, y_bot + 1.3],
                  cmap="gray", aspect="auto", zorder=1, interpolation="bilinear")
        ax.add_patch(FancyBboxPatch(
            (x, y_bot), 1.3, 1.3,
            boxstyle="round,pad=0.0,rounding_size=0.04",
            ec=HIGH, fc="none", lw=1.6))
        ax.text(x + 0.65, y_bot - 0.15, f"frame  t{['-1','','+1'][k]}",
                ha="center", fontsize=9, color=GRAY)
    for k in range(2):
        x0 = 1.7 + k * 1.55 + 1.3
        ax.annotate("", xy=(x0 + 0.22, y_bot + 0.65),
                    xytext=(x0 + 0.02, y_bot + 0.65),
                    arrowprops=dict(arrowstyle="-|>", color=HIGH, lw=1.8))
    ax.text(6.15, y_bot + 0.85, "consistent",
            fontsize=11, color=HIGH, fontweight="bold", va="center")
    ax.text(6.15, y_bot + 0.55, "HF detail",
            fontsize=10, color=HIGH, va="center")
    ax.text(6.15, y_bot + 0.20, "⇒  stable",
            fontsize=11, color=HIGH, fontweight="bold", va="center",
            style="italic")

    plt.tight_layout()
    plt.savefig(OUT / "motivation_flicker.png", bbox_inches="tight",
                facecolor=WHITE)
    plt.close()
    print(f"  wrote {OUT / 'motivation_flicker.png'}")


def make_vsr_task():
    """VSR task illustration: LR sequence (3 frames) → VSR → HR frame."""
    fig, ax = plt.subplots(figsize=(8.5, 3.4), dpi=200)
    ax.set_xlim(0, 8.5)
    ax.set_ylim(0, 3.4)
    ax.set_aspect("equal")
    ax.axis("off")

    # Build a 'scene' image used for both LR and HR (with appropriate detail)
    rng = np.random.default_rng(2)
    h, w = 96, 96
    yy, xx = np.mgrid[0:h, 0:w] / h
    base = (0.5 * np.sin(7 * xx + 0.5)
            + 0.4 * np.cos(6 * yy + 0.2)
            + 0.3 * np.sin(10 * (xx - yy)))
    base = (base - base.min()) / (base.max() - base.min())

    # HR — sharp version with fine high-frequency detail (small grain)
    fine_hf = gaussian_filter(rng.normal(size=base.shape), sigma=0.5)
    hr_img = base + 0.05 * fine_hf
    hr_img = (hr_img - hr_img.min()) / (hr_img.max() - hr_img.min())

    # LR — heavily blurred (simulate downsampling artifact look)
    lr_img = gaussian_filter(base, sigma=2.5)
    lr_img = (lr_img - lr_img.min()) / (lr_img.max() - lr_img.min())

    # Three LR frames (slightly offset to suggest motion)
    lr_size = 0.78
    lr_x_centers = [0.6, 0.95, 1.30]
    lr_y_centers = [2.40, 1.90, 1.40]
    for k in range(3):
        lr_with_motion = np.roll(lr_img, shift=k * 2, axis=1)
        lr_with_motion = np.roll(lr_with_motion, shift=k * 1, axis=0)
        x = lr_x_centers[k]
        y = lr_y_centers[k]
        ax.imshow(lr_with_motion, extent=[x, x + lr_size, y, y + lr_size],
                  cmap="gray", aspect="auto", zorder=2 - k * 0.1,
                  interpolation="bilinear")
        ax.add_patch(FancyBboxPatch(
            (x, y), lr_size, lr_size,
            boxstyle="round,pad=0.0,rounding_size=0.04",
            ec=INK, fc="none", lw=1.0, zorder=4 - k * 0.1))

    # LR caption
    ax.text(1.0, 0.95, "LR sequence",
            ha="center", fontsize=11, fontweight="bold", color=INK)
    ax.text(1.0, 0.65, "{ $I^{LR}_{t-1}$,  $I^{LR}_t$,  $I^{LR}_{t+1}$ }",
            ha="center", fontsize=10, color=GRAY)
    ax.text(1.0, 0.35, "low resolution",
            ha="center", fontsize=9, color=GRAY, style="italic")

    # Arrow → VSR module
    ax.add_patch(FancyArrowPatch(
        (2.35, 2.10), (3.30, 2.10),
        arrowstyle="-|>", mutation_scale=18, color=INK, lw=1.6))

    # VSR box
    box_x, box_y, bw, bh = 3.30, 1.55, 1.65, 1.20
    ax.add_patch(FancyBboxPatch(
        (box_x, box_y), bw, bh,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        ec="#6A2EA6", fc="#F4ECFA", lw=1.6))
    ax.text(box_x + bw / 2, box_y + bh / 2 + 0.05, "VSR",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color="#6A2EA6")
    ax.text(box_x + bw / 2, box_y + bh / 2 - 0.25,
            "model",
            ha="center", va="center", fontsize=10, color=GRAY,
            style="italic")

    # Arrow → HR
    ax.add_patch(FancyArrowPatch(
        (box_x + bw + 0.10, 2.10), (box_x + bw + 1.10, 2.10),
        arrowstyle="-|>", mutation_scale=18, color=INK, lw=1.6))

    # HR — big
    hr_x = box_x + bw + 1.10
    hr_size = 1.7
    ax.imshow(hr_img, extent=[hr_x, hr_x + hr_size, 1.30, 1.30 + hr_size],
              cmap="gray", aspect="auto", zorder=2, interpolation="bilinear")
    ax.add_patch(FancyBboxPatch(
        (hr_x, 1.30), hr_size, hr_size,
        boxstyle="round,pad=0.0,rounding_size=0.05",
        ec=INK, fc="none", lw=1.4))
    ax.text(hr_x + hr_size / 2, 0.95, "HR frame",
            ha="center", fontsize=11, fontweight="bold", color=INK)
    ax.text(hr_x + hr_size / 2, 0.65, "$\\hat{I}^{HR}_t$",
            ha="center", fontsize=11, color=GRAY)
    ax.text(hr_x + hr_size / 2, 0.35,
            "high resolution · sharp",
            ha="center", fontsize=9, color=GRAY, style="italic")

    # Key insight banner (no Hangul to avoid font fallback in matplotlib;
    # Korean caption is added separately as a PowerPoint textbox).
    ax.text(4.25, 3.20,
            "Key — unlike single-image SR, VSR uses temporal context across frames",
            ha="center", fontsize=10.5, fontweight="bold", color="#6A2EA6")

    plt.tight_layout()
    # Hangul falls back; we accept the warning since the slide caption can
    # be overlaid as native PowerPoint text if needed.
    plt.savefig(OUT / "motivation_vsr_task.png", bbox_inches="tight",
                facecolor=WHITE)
    plt.close()
    print(f"  wrote {OUT / 'motivation_vsr_task.png'}")


if __name__ == "__main__":
    make_flicker()
    make_vsr_task()
