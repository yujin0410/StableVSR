"""Generate matplotlib diagrams used as raster figures in the defense deck.

Produces three PNGs in ``/tmp/thesis_work/fig_png/``:

    band_pyramid.png      Multi-scale DT-CWT decomposition with HIGH/LOW band partition.
    unet_injection.png    Schematic of the StableVSR U-Net with BD-SFT injection points.
    mag_phase_pipe.png    Magnitude / phase parallel-branch pipeline of a SubbandBlock.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D


OUT = Path("/tmp/thesis_work/fig_png")
OUT.mkdir(parents=True, exist_ok=True)

# Palette — match the deck
HIGH = "#2C55A8"
LOW = "#CB4235"
INK = "#121E3D"
GRAY = "#6B6B78"
SOFT = "#F6F4FB"
WHITE = "#FFFFFF"
TEAL = "#0F868E"
PURPLE = "#6A2EA6"


def make_band_pyramid():
    """Multi-scale DT-CWT decomposition with HIGH (j=1,2) / LOW (j=3,4) split."""
    fig, ax = plt.subplots(figsize=(13, 4.0), dpi=200)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # LR input on the left
    ax.add_patch(FancyBboxPatch((0.3, 1.0), 2.0, 2.0,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                ec=INK, fc=WHITE, lw=1.0))
    ax.text(1.3, 3.20, "LR  $I^{LR}_t$", ha="center", fontsize=12,
            color=INK, fontweight="bold")
    ax.text(1.3, 0.72, "$H_{LR} \\times W_{LR}$", ha="center", fontsize=10,
            color=GRAY)

    # Arrow to DT-CWT
    ax.add_patch(FancyArrowPatch((2.45, 2.0), (3.30, 2.0),
                                  arrowstyle="-|>", mutation_scale=14,
                                  color=INK, lw=1.5))
    ax.text(2.88, 2.20, "DT-CWT  J=4", ha="center", fontsize=9,
            color=GRAY, style="italic")

    # Four scales
    band_x = 3.5
    band_y = 0.95
    band_size = 1.05
    band_gap = 0.20

    scale_info = [
        (1, HIGH, "j=1", "H/2 × W/2"),
        (2, HIGH, "j=2", "H/4 × W/4"),
        (3, LOW,  "j=3", "H/8 × W/8"),
        (4, LOW,  "j=4", "H/16 × W/16"),
    ]
    import math
    for k, (j, color, lbl, size_lbl) in enumerate(scale_info):
        x = band_x + k * (band_size + band_gap)
        y = band_y
        ax.add_patch(FancyBboxPatch(
            (x, y), band_size, band_size,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            ec=color, fc=color + "1A", lw=1.6,
        ))
        for ang_i in range(6):
            cx = x + band_size / 2
            cy = y + band_size / 2
            theta = math.radians(15 + ang_i * 30)
            r = band_size * 0.32
            ax.plot([cx - r * math.cos(theta), cx + r * math.cos(theta)],
                    [cy - r * math.sin(theta), cy + r * math.sin(theta)],
                    color=color, lw=0.9, alpha=0.55)
        ax.text(x + band_size / 2, y - 0.22, size_lbl,
                ha="center", fontsize=8.5, color=GRAY)
        ax.text(x + band_size / 2, y + band_size + 0.10, lbl,
                ha="center", fontsize=10, fontweight="bold", color=color)

    # Group brackets and labels (with vertical spacing to avoid overlap)
    hx0 = band_x
    hx1 = band_x + 2 * band_size + band_gap
    lx0 = band_x + 2 * (band_size + band_gap)
    lx1 = band_x + 4 * band_size + 3 * band_gap

    y_bracket = band_y + band_size + 0.45
    # HIGH bracket
    ax.plot([hx0, hx0, hx1, hx1],
            [y_bracket - 0.05, y_bracket, y_bracket, y_bracket - 0.05],
            color=HIGH, lw=1.8)
    ax.text((hx0 + hx1) / 2, y_bracket + 0.10,
            "HIGH band  ℋ = {j=1, j=2}",
            ha="center", fontsize=10.5, fontweight="bold", color=HIGH)
    ax.text((hx0 + hx1) / 2, y_bracket + 0.40,
            "[1/8, 1/2] cyc/px  ·  fine texture",
            ha="center", fontsize=8.5, color=GRAY, style="italic")
    # Injection annotation
    ax.text((hx0 + hx1) / 2, y_bracket + 0.62,
            "→  up_blocks[1] (decoder)",
            ha="center", fontsize=9, color=HIGH, fontweight="bold")

    # LOW bracket
    ax.plot([lx0, lx0, lx1, lx1],
            [y_bracket - 0.05, y_bracket, y_bracket, y_bracket - 0.05],
            color=LOW, lw=1.8)
    ax.text((lx0 + lx1) / 2, y_bracket + 0.10,
            "LOW band  ℒ = {j=3, j=4}",
            ha="center", fontsize=10.5, fontweight="bold", color=LOW)
    ax.text((lx0 + lx1) / 2, y_bracket + 0.40,
            "[0, 1/8] cyc/px  ·  coarse structure",
            ha="center", fontsize=8.5, color=GRAY, style="italic")
    ax.text((lx0 + lx1) / 2, y_bracket + 0.62,
            "→  down_blocks[1] (encoder)",
            ha="center", fontsize=9, color=LOW, fontweight="bold")

    # LP component on the right
    lp_x = band_x + 4 * band_size + 3 * band_gap + 0.30
    ax.add_patch(FancyBboxPatch(
        (lp_x, band_y - 0.05), 2.5, band_size + 0.10,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        ec="#9C9CB0", fc="#EFEFF6", lw=0.9))
    ax.text(lp_x + 1.25, band_y + 0.65, "$I^{LP}_t$",
            ha="center", fontsize=15, fontweight="bold", color=INK)
    ax.text(lp_x + 1.25, band_y + 0.30, "lowpass / DC",
            ha="center", fontsize=8.5, color=GRAY, style="italic")
    ax.text(lp_x + 1.25, band_y - 0.22, "real-valued",
            ha="center", fontsize=8.5, color=GRAY)

    plt.tight_layout()
    plt.savefig(OUT / "band_pyramid.png", bbox_inches="tight",
                facecolor=WHITE)
    plt.close()
    print(f"  wrote {OUT / 'band_pyramid.png'}")


def make_unet_injection():
    """Schematic of the StableVSR U-Net with BD-SFT injection points.

    Convention: up_blocks[i] mirrors down_blocks[i] in channel count.
        down: [0]=256, [1]=512, [2]=512, [3]=1024
        up:   [0]=1024, [1]=512, [2]=512, [3]=256
    So up_blocks[1] has 512 ch (matching down_blocks[1]).
    """
    fig, ax = plt.subplots(figsize=(13, 4.6), dpi=200)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Encoder/Decoder block heights: tall on outsides, narrow at the bottleneck.
    block_widths = [1.0, 1.0, 1.0, 1.0]
    block_h_down = [2.4, 1.8, 1.3, 0.8]
    channels_down = [256, 512, 512, 1024]

    gap_x = 0.18
    start_x = 1.0
    block_x_pos_down = [start_x + i * (block_widths[0] + gap_x) for i in range(4)]

    centerline_y = 2.6
    # Down blocks
    for i in range(4):
        x = block_x_pos_down[i]
        w = block_widths[i]
        h = block_h_down[i]
        y = centerline_y - h / 2
        is_inject = (i == 1)
        fc = "#FFE2E0" if is_inject else "#EDE7F6"
        ec = LOW if is_inject else "#9C9CB0"
        lw = 1.6 if is_inject else 0.8
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            ec=ec, fc=fc, lw=lw))
        ax.text(x + w / 2, y + h / 2,
                f"down\nblocks[{i}]",
                ha="center", va="center", fontsize=8.5,
                color=LOW if is_inject else INK,
                fontweight="bold" if is_inject else "normal")
        ax.text(x + w / 2, y - 0.20,
                f"{channels_down[i]} ch",
                ha="center", fontsize=8, color=GRAY)

    # Bottleneck divider
    mid_x = block_x_pos_down[3] + block_widths[3] + 0.10
    ax.text(mid_x + 0.18, centerline_y, "  mid  \n  block  ",
            ha="center", va="center", fontsize=8, color=GRAY, style="italic")

    # Up blocks
    block_h_up = block_h_down[::-1]
    channels_up = [1024, 512, 512, 256]
    up_start_x = mid_x + 0.50
    block_x_pos_up = [up_start_x + i * (block_widths[0] + gap_x) for i in range(4)]

    for i in range(4):
        x = block_x_pos_up[i]
        w = block_widths[i]
        h = block_h_up[i]
        y = centerline_y - h / 2
        is_inject = (i == 1)  # up_blocks[1]
        fc = "#DDE6F8" if is_inject else "#EDE7F6"
        ec = HIGH if is_inject else "#9C9CB0"
        lw = 1.6 if is_inject else 0.8
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            ec=ec, fc=fc, lw=lw))
        ax.text(x + w / 2, y + h / 2,
                f"up\nblocks[{i}]",
                ha="center", va="center", fontsize=8.5,
                color=HIGH if is_inject else INK,
                fontweight="bold" if is_inject else "normal")
        ax.text(x + w / 2, y - 0.20,
                f"{channels_up[i]} ch",
                ha="center", fontsize=8, color=GRAY)

    # Skip connections — encoder[i] → decoder[3-i]
    for i in range(4):
        x0 = block_x_pos_down[i] + block_widths[i] / 2
        x1 = block_x_pos_up[3 - i] + block_widths[i] / 2
        skip_y = centerline_y + max(block_h_down[i], block_h_up[3 - i]) / 2 + 0.05 + i * 0.03
        ax.plot([x0, x1], [skip_y, skip_y], "--", color=GRAY,
                lw=0.5, alpha=0.4)
        ax.plot([x0, x0], [centerline_y + block_h_down[i] / 2, skip_y],
                "--", color=GRAY, lw=0.5, alpha=0.4)
        ax.plot([x1, x1], [centerline_y + block_h_up[3 - i] / 2, skip_y],
                "--", color=GRAY, lw=0.5, alpha=0.4)

    # Injection arrows (from below — cleaner, less clutter on the skip lines)
    # LOW band → down_blocks[1]
    x_target_low = block_x_pos_down[1] + block_widths[1] / 2
    y_inject_bottom = centerline_y - max(block_h_down) / 2 - 0.45
    ax.add_patch(FancyArrowPatch(
        (x_target_low, 0.55), (x_target_low, y_inject_bottom + 0.40),
        arrowstyle="-|>", mutation_scale=14, color=LOW, lw=2.0))
    ax.text(x_target_low, 0.30, "(γ_ℒ, β_ℒ)",
            ha="center", fontsize=11, fontweight="bold", color=LOW)
    ax.text(x_target_low, 0.05, "LOW band SFT",
            ha="center", fontsize=8.5, color=LOW, style="italic")

    # HIGH band → up_blocks[1]
    x_target_high = block_x_pos_up[1] + block_widths[1] / 2
    ax.add_patch(FancyArrowPatch(
        (x_target_high, 0.55), (x_target_high, y_inject_bottom + 0.40),
        arrowstyle="-|>", mutation_scale=14, color=HIGH, lw=2.0))
    ax.text(x_target_high, 0.30, "(γ_ℋ, β_ℋ)",
            ha="center", fontsize=11, fontweight="bold", color=HIGH)
    ax.text(x_target_high, 0.05, "HIGH band SFT",
            ha="center", fontsize=8.5, color=HIGH, style="italic")

    # Side roles
    ax.text(0.1, 4.20, "Encoder", fontsize=11, fontweight="bold", color=INK)
    ax.text(0.1, 3.95, "consolidates structure", fontsize=8.5, color=GRAY,
            style="italic")
    decoder_lbl_x = block_x_pos_up[3] + block_widths[3] + 0.10
    ax.text(decoder_lbl_x, 4.20, "Decoder", fontsize=11, fontweight="bold",
            color=INK)
    ax.text(decoder_lbl_x, 3.95, "synthesizes texture", fontsize=8.5,
            color=GRAY, style="italic")

    plt.tight_layout()
    plt.savefig(OUT / "unet_injection.png", bbox_inches="tight",
                facecolor=WHITE)
    plt.close()
    print(f"  wrote {OUT / 'unet_injection.png'}")


def make_mag_phase_pipeline():
    """Magnitude/phase decoupled processing in a SubbandBlock."""
    fig, ax = plt.subplots(figsize=(12, 5.0), dpi=200)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Input
    ax.add_patch(FancyBboxPatch(
        (0.3, 2.0), 1.7, 1.2,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        ec=INK, fc=WHITE, lw=1.2))
    ax.text(1.15, 2.85, "6 directional", ha="center", fontsize=9.5, color=INK)
    ax.text(1.15, 2.60, "complex coeffs", ha="center", fontsize=9.5, color=INK,
            fontweight="bold")
    ax.text(1.15, 2.30, "$C^{(j,d)} = a + ib$", ha="center", fontsize=10, color=INK)

    # Split into magnitude and phase branches
    # Top branch — magnitude
    ax.add_patch(FancyArrowPatch(
        (2.05, 2.8), (3.3, 4.0), arrowstyle="-|>", mutation_scale=12,
        color=HIGH, lw=1.6))
    ax.add_patch(FancyBboxPatch(
        (3.3, 3.7), 2.4, 0.7,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        ec=HIGH, fc="#DDE6F8", lw=1.3))
    ax.text(4.5, 4.05, "$M = |C| = \\sqrt{a^2+b^2}$",
            ha="center", fontsize=10.5, color=HIGH, fontweight="bold")
    ax.text(4.5, 3.45, "18 ch",
            ha="center", fontsize=8.5, color=GRAY)

    # Bottom branch — phase trig encoding
    ax.add_patch(FancyArrowPatch(
        (2.05, 2.4), (3.3, 1.2), arrowstyle="-|>", mutation_scale=12,
        color=LOW, lw=1.6))
    ax.add_patch(FancyBboxPatch(
        (3.3, 0.85), 2.4, 0.7,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        ec=LOW, fc="#FFDBDB", lw=1.3))
    ax.text(4.5, 1.20, "$\\mathcal{T}(\\phi) = (\\sin\\phi, \\cos\\phi)$",
            ha="center", fontsize=10.5, color=LOW, fontweight="bold")
    ax.text(4.5, 0.60, "36 ch  ·  wraparound-safe",
            ha="center", fontsize=8.5, color=GRAY)

    # Branch operator ℰ
    for y0 in (3.7, 0.85):
        ax.add_patch(FancyArrowPatch(
            (5.7, y0 + 0.35), (6.7, y0 + 0.35),
            arrowstyle="-|>", mutation_scale=10, color=GRAY, lw=1.2))
        ax.add_patch(FancyBboxPatch(
            (6.7, y0), 1.8, 0.7,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            ec="#9C9CB0", fc="#FAFAFC", lw=0.9))
        ax.text(7.6, y0 + 0.42, "ℰ : DWConv₃ → SiLU",
                ha="center", fontsize=8.5, color=INK)
        ax.text(7.6, y0 + 0.22, "→ SA → Conv₁ₓ₁",
                ha="center", fontsize=8.5, color=INK)

    # Outputs of ℰ
    ax.text(7.6, 4.55, "$e_M$  (64 ch)",
            ha="center", fontsize=9, color=HIGH, fontweight="bold")
    ax.text(7.6, 0.65, "$(e_{\\sin}, e_{\\cos})$  (2 × 64 ch)",
            ha="center", fontsize=9, color=LOW, fontweight="bold")

    # Recombination
    ax.add_patch(FancyBboxPatch(
        (9.5, 1.8), 2.2, 1.4,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        ec=PURPLE, fc="#EDE3F5", lw=1.4))
    ax.text(10.6, 2.85, "Recombination",
            ha="center", fontsize=10.5, color=PURPLE, fontweight="bold")
    ax.text(10.6, 2.55, "$e_{\\mathrm{re}} = e_M \\odot e_{\\cos}$",
            ha="center", fontsize=9.5, color=INK)
    ax.text(10.6, 2.30, "$e_{\\mathrm{im}} = e_M \\odot e_{\\sin}$",
            ha="center", fontsize=9.5, color=INK)
    ax.text(10.6, 2.05, "$\\to$  [eM ‖ e_re ‖ e_im]  192 ch",
            ha="center", fontsize=8.5, color=GRAY, style="italic")

    # Arrows into recombination
    ax.add_patch(FancyArrowPatch(
        (8.5, 4.0), (9.5, 2.95), arrowstyle="-|>", mutation_scale=10,
        color=HIGH, lw=1.4))
    ax.add_patch(FancyArrowPatch(
        (8.5, 1.2), (9.5, 2.15), arrowstyle="-|>", mutation_scale=10,
        color=LOW, lw=1.4))

    plt.tight_layout()
    plt.savefig(OUT / "mag_phase_pipe.png", bbox_inches="tight",
                facecolor=WHITE)
    plt.close()
    print(f"  wrote {OUT / 'mag_phase_pipe.png'}")


def make_flickering_demo():
    """Three-frame illustration of stochastic per-frame texture flickering."""
    import numpy as np
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=200)
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 3.6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Top row — inconsistent textures (StableVSR-style flicker)
    y_top = 2.1
    ax.text(0.05, y_top + 0.75, "Per-frame independent\nstochastic denoising",
            fontsize=9, color=LOW, fontweight="bold", va="top")
    for k, x in enumerate([1.7, 3.2, 4.7]):
        # Random texture patch (different seed per frame)
        np.random.seed(k * 7 + 1)
        patch = np.random.randn(40, 40) * 0.5
        patch = (patch - patch.min()) / (patch.max() - patch.min())
        ax.imshow(patch, extent=[x, x + 1.3, y_top, y_top + 1.3],
                  cmap="gray", aspect="auto", zorder=1)
        ax.add_patch(FancyBboxPatch(
            (x, y_top), 1.3, 1.3,
            boxstyle="round,pad=0.0,rounding_size=0.04",
            ec=LOW, fc="none", lw=1.5))
        ax.text(x + 0.65, y_top - 0.15, f"frame  t{['-1','','+1'][k]}",
                ha="center", fontsize=8.5, color=GRAY)
    # Red wavy "flicker" arrows between frames
    for k in range(2):
        x0 = 1.7 + k * 1.5 + 1.3
        ax.plot([x0 + 0.04, x0 + 0.16],
                [y_top + 0.65, y_top + 0.65], color=LOW, lw=1.5)
        ax.annotate("", xy=(x0 + 0.2, y_top + 0.65),
                    xytext=(x0 + 0.04, y_top + 0.65),
                    arrowprops=dict(arrowstyle="-|>", color=LOW, lw=1.5))
    ax.text(6.10, y_top + 0.65,
            "≠   ≠",
            fontsize=22, color=LOW, fontweight="bold", va="center")
    ax.text(6.10, y_top + 0.18,
            "flicker",
            fontsize=9, color=LOW, fontweight="bold", style="italic",
            va="center")

    # Bottom row — consistent textures (after our method)
    y_bot = 0.25
    ax.text(0.05, y_bot + 0.75, "Frequency-conditioned\n(WC-BD-SFT)",
            fontsize=9, color=HIGH, fontweight="bold", va="top")
    # Same texture across all 3 frames
    np.random.seed(99)
    patch = np.random.randn(40, 40) * 0.5
    patch = (patch - patch.min()) / (patch.max() - patch.min())
    for k, x in enumerate([1.7, 3.2, 4.7]):
        ax.imshow(patch, extent=[x, x + 1.3, y_bot, y_bot + 1.3],
                  cmap="gray", aspect="auto", zorder=1)
        ax.add_patch(FancyBboxPatch(
            (x, y_bot), 1.3, 1.3,
            boxstyle="round,pad=0.0,rounding_size=0.04",
            ec=HIGH, fc="none", lw=1.5))
        ax.text(x + 0.65, y_bot - 0.15, f"frame  t{['-1','','+1'][k]}",
                ha="center", fontsize=8.5, color=GRAY)
    ax.text(6.10, y_bot + 0.65,
            "=   =",
            fontsize=22, color=HIGH, fontweight="bold", va="center")
    ax.text(6.10, y_bot + 0.18,
            "stable",
            fontsize=9, color=HIGH, fontweight="bold", style="italic",
            va="center")

    plt.tight_layout()
    plt.savefig(OUT / "flickering_demo.png", bbox_inches="tight",
                facecolor=WHITE)
    plt.close()
    print(f"  wrote {OUT / 'flickering_demo.png'}")


def make_vsr_timeline():
    """Horizontal timeline of VSR methods, color-coded by paradigm."""
    fig, ax = plt.subplots(figsize=(13, 3.0), dpi=200)
    ax.set_xlim(2017.5, 2026.5)
    ax.set_ylim(0, 3.0)
    ax.axis("off")

    # Main axis line
    ax.plot([2018, 2026.2], [1.3, 1.3], color=INK, lw=1.5)
    for yr in range(2018, 2027):
        ax.plot([yr, yr], [1.22, 1.38], color=INK, lw=1.0)
        ax.text(yr, 0.95, str(yr), ha="center", fontsize=8.5, color=GRAY)

    items = [
        # (year, label, paradigm, y_offset, color)
        (2018, "EDVR / DUF",          "CNN",       0.6, "#9C9CB0"),
        (2019, "TOFlow / RBPN",       "CNN",       0.6, "#9C9CB0"),
        (2021, "BasicVSR",            "CNN",       0.6, "#9C9CB0"),
        (2022, "BasicVSR++ / VRT",    "CNN",       0.6, "#9C9CB0"),
        (2018, "ESRGAN",              "GAN",       -0.55, "#F08C7A"),
        (2021, "RealBasicVSR",        "GAN",       -0.55, "#F08C7A"),
        (2024, "VideoGigaGAN",        "GAN",       -0.55, "#F08C7A"),
        (2023, "MGLD-VSR",            "Diffusion", 0.6,   HIGH),
        (2024, "StableVSR",           "Diffusion", 0.6,   HIGH),  # baseline
        (2024, "Upscale-A-Video",     "Diffusion", -0.55, HIGH),
        (2025, "DiffVSR / STAR",      "Diffusion", -0.55, HIGH),
        (2025, "DLoRAL / UltraVSR\nSeedVR2",  "Diffusion", 0.6,   HIGH),
        (2026, "DGAF-VSR",            "Diffusion", -0.55, HIGH),
        (2026, "WC-BD-SFT (Ours)",    "Ours",      0.6,   PURPLE),
    ]
    # Plot
    for yr, lbl, par, dy, col in items:
        ty = 1.3 + dy
        ax.plot([yr, yr], [1.3, ty], color=col, lw=0.9, alpha=0.7)
        is_ours = (par == "Ours")
        is_baseline = (lbl == "StableVSR")
        weight = "bold" if (is_ours or is_baseline) else "normal"
        fs = 9 if (is_ours or is_baseline) else 8.5
        # Marker
        ax.scatter([yr], [ty], s=55 if is_ours else 35,
                   c=col, edgecolors="white", linewidths=1.2, zorder=5)
        # Label box
        va = "bottom" if dy > 0 else "top"
        ax.text(yr, ty + (0.07 if dy > 0 else -0.07), lbl,
                ha="center", va=va, fontsize=fs, color=col,
                fontweight=weight)

    # Legend
    legend_items = [
        ("CNN-based",  "#9C9CB0"),
        ("GAN-based",  "#F08C7A"),
        ("Diffusion",  HIGH),
        ("Ours",       PURPLE),
    ]
    for k, (l, c) in enumerate(legend_items):
        x0 = 2018.0 + k * 1.6
        ax.scatter([x0], [2.75], s=45, c=c, edgecolors="white", linewidths=1)
        ax.text(x0 + 0.12, 2.75, l, va="center", fontsize=9, color=INK)

    plt.tight_layout()
    plt.savefig(OUT / "vsr_timeline.png", bbox_inches="tight",
                facecolor=WHITE)
    plt.close()
    print(f"  wrote {OUT / 'vsr_timeline.png'}")


def make_diffusion_process():
    """Schematic of forward / reverse latent diffusion process."""
    fig, ax = plt.subplots(figsize=(13, 3.6), dpi=200)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.6)
    ax.set_aspect("equal")
    ax.axis("off")

    # 5 latent images, gradually noised
    import numpy as np
    n = 5
    xs = [1.0 + k * 2.4 for k in range(n)]
    y_top = 2.0
    rng = np.random.default_rng(2)
    # Base smooth pattern
    base = np.outer(np.cos(np.linspace(0, 3, 32)),
                    np.sin(np.linspace(0, 4, 32)))
    base = (base - base.min()) / (base.max() - base.min())
    for k, x in enumerate(xs):
        alpha = k / (n - 1)
        noise = rng.normal(size=(32, 32))
        img = (1 - alpha) * base + alpha * (noise * 0.5 + 0.5)
        img = np.clip(img, 0, 1)
        ax.imshow(img, extent=[x, x + 1.0, y_top, y_top + 1.0],
                  cmap="gray", aspect="auto", zorder=1)
        ax.add_patch(FancyBboxPatch(
            (x, y_top), 1.0, 1.0,
            boxstyle="round,pad=0.0,rounding_size=0.04",
            ec=INK, fc="none", lw=0.9))
        ax.text(x + 0.5, y_top - 0.15, f"$x_{{{k * (1000 // (n - 1))}}}$",
                ha="center", fontsize=10, color=INK)

    # Forward arrow (noise added: left → right)
    ax.add_patch(FancyArrowPatch(
        (xs[0] + 0.5, y_top + 1.25), (xs[-1] + 0.5, y_top + 1.25),
        arrowstyle="-|>", mutation_scale=14, color=LOW, lw=1.4))
    ax.text((xs[0] + xs[-1]) / 2 + 0.5, y_top + 1.45,
            "Forward · progressively add noise",
            ha="center", fontsize=10, color=LOW, fontweight="bold")
    ax.text((xs[0] + xs[-1]) / 2 + 0.5, y_top + 1.65,
            "$q(x_t | x_{t-1}) = \\mathcal{N}(\\sqrt{1-\\beta_t}\\,x_{t-1},\\, \\beta_t I)$",
            ha="center", fontsize=10, color=INK, style="italic")

    # Reverse arrow (denoise: right → left)
    ax.add_patch(FancyArrowPatch(
        (xs[-1] + 0.5, y_top - 0.5), (xs[0] + 0.5, y_top - 0.5),
        arrowstyle="-|>", mutation_scale=14, color=HIGH, lw=1.4))
    ax.text((xs[0] + xs[-1]) / 2 + 0.5, y_top - 0.70,
            "Reverse · U-Net denoising  ⇐  $\\epsilon_\\theta(x_t, t, c)$",
            ha="center", fontsize=10, color=HIGH, fontweight="bold")
    ax.text((xs[0] + xs[-1]) / 2 + 0.5, y_top - 0.92,
            "trained with  $\\mathcal{L}_{MSE} = \\mathbb{E}\\,\\|\\epsilon - "
            "\\epsilon_\\theta(x_t, t, c)\\|^2$",
            ha="center", fontsize=9.5, color=INK, style="italic")

    plt.tight_layout()
    plt.savefig(OUT / "diffusion_process.png", bbox_inches="tight",
                facecolor=WHITE)
    plt.close()
    print(f"  wrote {OUT / 'diffusion_process.png'}")


def make_pd_curve():
    """Perception–distortion trade-off scatter with the 4 methods."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Trade-off frontier curve
    import numpy as np
    xx = np.linspace(0.05, 0.95, 200)
    yy = 1 - xx**0.7
    ax.plot(xx, yy, "--", color=GRAY, lw=1.0, alpha=0.7,
            label="perception–distortion frontier")

    # Method points (x = distortion=lower better, y = perception=higher better)
    methods = [
        ("BasicVSR++", 0.18, 0.30, "#9C9CB0"),     # high fidelity, low perception
        ("StableVSR",  0.60, 0.55, HIGH),
        ("DGAF-VSR",   0.58, 0.62, TEAL),
        ("WC-BD-SFT (Ours)", 0.50, 0.86, PURPLE),
    ]
    for name, x, y, col in methods:
        is_ours = "Ours" in name
        ax.scatter([x], [y], s=180 if is_ours else 90,
                   c=col, edgecolors="white", linewidths=1.5, zorder=5,
                   marker="*" if is_ours else "o")
        dx = 0.025
        dy = 0.04 if is_ours else 0.035
        ax.text(x + dx, y + dy, name, fontsize=10,
                color=col, fontweight="bold" if is_ours else "normal")

    ax.set_xlabel("Distortion  ←  lower  ·  higher fidelity  →",
                  fontsize=10, color=INK)
    ax.set_ylabel("Perceptual quality  →  higher is better",
                  fontsize=10, color=INK)
    ax.set_title("Perception–Distortion Trade-off",
                 fontsize=12, color=INK, fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(GRAY)
    ax.spines["left"].set_color(GRAY)
    ax.tick_params(colors=GRAY)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(OUT / "pd_curve.png", bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"  wrote {OUT / 'pd_curve.png'}")


if __name__ == "__main__":
    print("Generating diagrams...")
    make_band_pyramid()
    make_unet_injection()
    make_mag_phase_pipeline()
    make_flickering_demo()
    make_vsr_timeline()
    make_diffusion_process()
    make_pd_curve()
    print("Done.")
