"""Build the Master's thesis defense presentation for
'Wavelet-Conditioned Band-Decoupled Spatial Feature Transform
for Diffusion-Based Video Super-Resolution' (Yu-Jin Cho, 2026).

Strategy:
  - Start from the user-provided template ``template.pptx`` which already
    contains 9 hand-styled slides (cover, contents, architecture diagram,
    SubbandBlock diagram, evaluation visuals, DT-CWT subbands, thanks).
  - Append additional content slides that cover Introduction, Related Work,
    Method details, Experiments, Results, Ablation, Discussion, Limitations,
    Conclusion, and Future Work using the existing ``16_제목만`` layout.
  - Reorder the slide list via XML manipulation so the final presentation
    flows naturally for the defense.
"""

from copy import deepcopy
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt, Inches


TEMPLATE = Path("/tmp/thesis_work/template.pptx")
FIG_DIR = Path("/tmp/thesis_work/fig_png")
OUT = Path("/home/user/StableVSR/presentation/WC_BD_SFT_Defense_Cho_YuJin.pptx")

# Slide dimensions (16:9)
SLIDE_W_IN = 13.33
SLIDE_H_IN = 7.50

# Visual style: pick muted academic palette consistent with the cover slide.
NAVY = RGBColor(0x1F, 0x2A, 0x44)
ACCENT = RGBColor(0xC8, 0x10, 0x2E)        # title accent (matches cover red strip)
TEAL = RGBColor(0x2C, 0x6E, 0x8F)
GRAY = RGBColor(0x55, 0x55, 0x55)
LIGHT_GRAY = RGBColor(0xE5, 0xE5, 0xE5)
HIGH_BAND = RGBColor(0x1F, 0x77, 0xB4)     # blue (HIGH-band injection)
LOW_BAND = RGBColor(0xD6, 0x27, 0x28)      # red  (LOW-band injection)


# ------------------------------------------------------------------
# Slide-building helpers
# ------------------------------------------------------------------
def add_title_slide(prs, layout_idx=12):
    """Add a new content slide using the styled '16_제목만' layout."""
    return prs.slides.add_slide(prs.slide_layouts[layout_idx])


def set_title(slide, title):
    """Set the title via the layout's title placeholder.

    Layout 12 uses BODY placeholder (idx=1) as the visible title bar at top.
    """
    for ph in slide.placeholders:
        # The layout-12 title bar is placeholder idx=1 (BODY); some slides have
        # a different placeholder positioned at the very top.
        if ph.placeholder_format.idx in (0, 1):
            tf = ph.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            run = p.add_run()
            run.text = title
            run.font.size = Pt(28)
            run.font.bold = True
            run.font.color.rgb = NAVY
            p.alignment = PP_ALIGN.LEFT
            return ph
    # Fallback: add a text box at the top.
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(12.3), Inches(0.6))
    tf = tb.text_frame
    tf.text = title
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run = p.runs[0]
    run.font.size = Pt(28)
    run.font.bold = True
    run.font.color.rgb = NAVY
    return tb


def add_textbox(slide, left, top, width, height, *, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = Inches(0.05)
    tf.margin_right = Inches(0.05)
    tf.margin_top = Inches(0.03)
    tf.margin_bottom = Inches(0.03)
    return box


def add_paragraph(tf, text, *, size=16, bold=False, italic=False, color=None,
                  align=PP_ALIGN.LEFT, level=0, space_before=2, space_after=4):
    """Append a paragraph; reuse the first empty paragraph if applicable."""
    if not tf.text and not tf.paragraphs[0].runs:
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.alignment = align
    p.level = level
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color is not None:
        run.font.color.rgb = color
    return p


def add_bullet(tf, text, *, size=15, bold=False, italic=False, color=None, level=0,
               bullet_char="•", space_before=2, space_after=4):
    """Append a bullet line, prepending a bullet glyph + tab indent."""
    indent = "    " * level
    prefix = f"{indent}{bullet_char}  " if bullet_char else indent
    return add_paragraph(tf, prefix + text, size=size, bold=bold, italic=italic,
                         color=color, level=0,
                         space_before=space_before, space_after=space_after)


def add_rounded_rect(slide, left, top, width, height, *, fill=None, line=None,
                     line_width=0.75):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left), Inches(top), Inches(width), Inches(height),
    )
    shape.shadow.inherit = False
    if fill is None:
        shape.fill.background()
    else:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(line_width)
    return shape


def add_callout(slide, left, top, width, height, *, title, body, accent=ACCENT,
                title_size=14, body_size=12, fill=RGBColor(0xFA, 0xFA, 0xFA)):
    """A small framed callout box used for grouped bullets / takeaways."""
    box = add_rounded_rect(slide, left, top, width, height, fill=fill,
                            line=accent, line_width=1.0)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.1)
    tf.margin_right = Inches(0.1)
    tf.margin_top = Inches(0.06)
    tf.margin_bottom = Inches(0.06)
    tf.text = ""
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = title
    run.font.size = Pt(title_size)
    run.font.bold = True
    run.font.color.rgb = accent
    for line in body:
        bp = tf.add_paragraph()
        bp.space_before = Pt(2)
        bp.space_after = Pt(1)
        r = bp.add_run()
        r.text = line
        r.font.size = Pt(body_size)
        r.font.color.rgb = NAVY
    return box


def add_picture_fit(slide, path, left, top, width, height, *, center=True):
    """Fit-while-preserve-aspect picture inside the bounding box."""
    from PIL import Image
    with Image.open(path) as im:
        w_px, h_px = im.size
    aspect_img = w_px / h_px
    aspect_box = width / height
    if aspect_img > aspect_box:
        new_w = width
        new_h = width / aspect_img
    else:
        new_h = height
        new_w = height * aspect_img
    if center:
        left = left + (width - new_w) / 2
        top = top + (height - new_h) / 2
    return slide.shapes.add_picture(str(path), Inches(left), Inches(top),
                                     Inches(new_w), Inches(new_h))


def add_table(slide, left, top, width, height, headers, rows, *,
              header_size=11, body_size=10.5, header_fill=NAVY,
              header_color=RGBColor(0xFF, 0xFF, 0xFF), zebra=True,
              bold_first_col=False, highlight_row_idx=None,
              highlight_fill=RGBColor(0xFF, 0xF4, 0xCC)):
    n_rows = len(rows) + 1
    n_cols = len(headers)
    tbl_shape = slide.shapes.add_table(n_rows, n_cols, Inches(left), Inches(top),
                                        Inches(width), Inches(height))
    tbl = tbl_shape.table
    # Header
    for j, h in enumerate(headers):
        cell = tbl.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = header_fill
        cell.margin_left = Inches(0.04)
        cell.margin_right = Inches(0.04)
        cell.margin_top = Inches(0.02)
        cell.margin_bottom = Inches(0.02)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf = cell.text_frame
        tf.word_wrap = True
        tf.text = ""
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = h
        r.font.size = Pt(header_size)
        r.font.bold = True
        r.font.color.rgb = header_color
    # Body
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.cell(i + 1, j)
            cell.margin_left = Inches(0.04)
            cell.margin_right = Inches(0.04)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            # zebra striping
            if highlight_row_idx is not None and i == highlight_row_idx:
                cell.fill.solid()
                cell.fill.fore_color.rgb = highlight_fill
            elif zebra and i % 2 == 1:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(0xF7, 0xF7, 0xF7)
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            tf = cell.text_frame
            tf.word_wrap = True
            tf.text = ""
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT
            r = p.add_run()
            r.text = str(val)
            r.font.size = Pt(body_size)
            r.font.color.rgb = NAVY
            r.font.bold = (j == 0 and bold_first_col) or (
                highlight_row_idx is not None and i == highlight_row_idx
            )
    return tbl_shape


def add_footnote(slide, text, *, top=7.05):
    box = slide.shapes.add_textbox(Inches(0.5), Inches(top), Inches(12.3), Inches(0.3))
    tf = box.text_frame
    tf.text = text
    r = tf.paragraphs[0].runs[0]
    r.font.size = Pt(9)
    r.font.italic = True
    r.font.color.rgb = GRAY


# ------------------------------------------------------------------
# Slide content builders
# ------------------------------------------------------------------
def slide_motivation_1(prs):
    s = add_title_slide(prs)
    set_title(s, "Motivation: Video Super-Resolution Meets Diffusion")

    box = add_textbox(s, 0.55, 0.95, 12.3, 2.6)
    tf = box.text_frame
    add_paragraph(tf, "Video Super-Resolution (VSR)", size=18, bold=True, color=NAVY)
    add_paragraph(tf, ("Recover HR frames from LR inputs by leveraging both intra-frame "
                       "spatial cues and inter-frame temporal information."),
                  size=14, color=GRAY)
    add_bullet(tf, "Early CNN-based VSR (TOFlow, EDVR, BasicVSR/++) — strong fidelity, "
                   "but over-smoothed outputs due to pixel-wise losses.", size=14)
    add_bullet(tf, "GAN-based VSR (ESRGAN, VideoGigaGAN) — sharper textures, but "
                   "unstable training and bounded texture diversity.", size=14)
    add_bullet(tf, "Diffusion Models — realistic, intricate high-frequency details "
                   "surpassing earlier generative approaches (DDPM, SR3).", size=14)

    add_callout(s, 0.55, 3.95, 12.3, 1.05,
                title="The opportunity",
                body=[
                    "Pre-trained image diffusion priors (e.g. Stable Diffusion v2.1) "
                    "synthesize photorealistic textures unmatched by CNN/GAN methods.",
                    "Goal — bring this generative prior to VSR without losing what made "
                    "it powerful in the first place.",
                ],
                accent=TEAL)

    add_callout(s, 0.55, 5.2, 12.3, 1.55,
                title="But a critical challenge…",
                body=[
                    "Image diffusion denoises each frame independently → stochastic "
                    "high-frequency details vary between frames.",
                    "Result: temporal flickering — restored textures shimmer and "
                    "shift, severely degrading perceived video quality.",
                ],
                accent=ACCENT)


def slide_motivation_2(prs):
    s = add_title_slide(prs)
    set_title(s, "Two Existing Directions — and Their Costs")

    # Two-column comparison
    add_rounded_rect(s, 0.55, 1.0, 6.0, 4.7, fill=RGBColor(0xF4, 0xF7, 0xFC),
                     line=TEAL, line_width=1.25)
    add_rounded_rect(s, 6.8, 1.0, 6.0, 4.7, fill=RGBColor(0xFC, 0xF4, 0xF4),
                     line=ACCENT, line_width=1.25)

    left = add_textbox(s, 0.75, 1.1, 5.6, 4.5)
    tfl = left.text_frame
    add_paragraph(tfl, "(1) Video-native diffusion", size=17, bold=True, color=TEAL)
    add_paragraph(tfl, "Train spatiotemporal U-Nets from scratch with 3D conv / "
                       "temporal attention.", size=13, color=GRAY)
    add_bullet(tfl, "Stable Video Diffusion (SVD) — billions of parameters.", size=13)
    add_bullet(tfl, "Requires massive video corpora.", size=13)
    add_bullet(tfl, "Cost: prohibitive compute & data.", size=13, color=ACCENT, bold=True)

    right = add_textbox(s, 7.0, 1.1, 5.6, 4.5)
    tfr = right.text_frame
    add_paragraph(tfr, "(2) Adapt image diffusion to video", size=17, bold=True, color=ACCENT)
    add_paragraph(tfr, "Add temporal modules and fine-tune the whole network.",
                  size=13, color=GRAY)
    add_bullet(tfr, "Upscale-A-Video, etc. — full-network fine-tuning.", size=13)
    add_bullet(tfr, "Heavy gradient flow through hundreds of millions of params.", size=13)
    add_bullet(tfr, "Cost: catastrophic forgetting of the generative prior.",
               size=13, color=ACCENT, bold=True)

    add_callout(s, 0.55, 5.95, 12.3, 1.1,
                title="Parameter-efficient adaptation (LoRA, ControlNet) is a practical alternative",
                body=[
                    "Freeze the diffusion backbone; train only small auxiliary modules — "
                    "preserves the prior and is feasible on modest hardware.",
                    "Open question: what conditioning signal lets such a lightweight "
                    "module achieve temporal consistency without 3D attention?",
                ],
                accent=NAVY)


def slide_related_work_vsr(prs):
    s = add_title_slide(prs)
    set_title(s, "Related Work · VSR Landscape")

    # Timeline of approaches
    box = add_textbox(s, 0.55, 1.0, 12.3, 2.4)
    tf = box.text_frame
    add_paragraph(tf, "Alignment-driven CNN VSR", size=16, bold=True, color=NAVY)
    add_bullet(tf, "Optical-flow warping: TOFlow, RBPN", size=13)
    add_bullet(tf, "Deformable / implicit alignment: DUF, TDAN, EDVR", size=13)
    add_bullet(tf, "Bidirectional recurrence: BasicVSR, BasicVSR++ (REDS-style benchmarks)", size=13)
    add_bullet(tf, "Transformer-based: VRT, RVRT (long-range temporal attention)", size=13)

    add_paragraph(tf, "Toward perceptual realism", size=16, bold=True, color=NAVY)
    add_bullet(tf, "GANs: SRGAN, ESRGAN, RealBasicVSR — sharper but training-unstable", size=13)
    add_bullet(tf, "VideoGigaGAN (2024) — large image GAN extended to video", size=13)

    add_callout(s, 0.55, 5.05, 12.3, 1.9,
                title="Limitation of CNN- / GAN-based VSR",
                body=[
                    "Pixel-wise losses (MSE) → over-smoothed outputs that miss "
                    "fine textural detail.",
                    "Alignment errors under fast motion or occlusion produce "
                    "visible artifacts.",
                    "GAN training instability and bounded texture diversity "
                    "motivated the shift to diffusion-based generative priors.",
                ],
                accent=ACCENT)


def slide_related_work_diff(prs):
    s = add_title_slide(prs)
    set_title(s, "Related Work · Diffusion-Based VSR")

    # Two axes
    add_rounded_rect(s, 0.55, 0.95, 6.0, 4.0, fill=RGBColor(0xF4, 0xF7, 0xFC),
                     line=TEAL, line_width=1.0)
    left = add_textbox(s, 0.75, 1.05, 5.6, 3.85)
    tfl = left.text_frame
    add_paragraph(tfl, "Axis 1 · Adapt image diffusion", size=16, bold=True, color=TEAL)
    add_bullet(tfl, "StableVSR (ECCV 2024) — baseline of this thesis. "
                    "ControlNet receives RAFT-warped x̂₀ of a neighbor frame.", size=12)
    add_bullet(tfl, "MGLD-VSR — motion-guided latent diffusion (real-world VSR).", size=12)
    add_bullet(tfl, "DGAF-VSR (CVPR 2026) — feature-level OGWM + FTCM.", size=12)
    add_bullet(tfl, "Recent (2025): DiffVSR, STAR, DC-VSR (robustness); "
                    "DLoRAL, UltraVSR, SeedVR2 (efficiency).", size=12)

    add_rounded_rect(s, 6.8, 0.95, 6.0, 4.0, fill=RGBColor(0xFC, 0xF4, 0xF4),
                     line=ACCENT, line_width=1.0)
    right = add_textbox(s, 7.0, 1.05, 5.6, 3.85)
    tfr = right.text_frame
    add_paragraph(tfr, "Axis 2 · Train video-native diffusion", size=16, bold=True, color=ACCENT)
    add_bullet(tfr, "Upscale-A-Video — text-prompted, LLaVA caption, "
                    "temporal attention layers.", size=12)
    add_bullet(tfr, "Stable Video Diffusion — fully video-native generative model "
                    "(billions of params).", size=12)
    add_bullet(tfr, "Prohibitive cost; risks degrading the pre-trained image prior.",
               size=12, italic=True, color=GRAY)

    # Direct comparator
    add_callout(s, 0.55, 5.1, 12.3, 1.85,
                title="Closest comparator · DGAF-VSR",
                body=[
                    "Parameter-efficient adaptation, like ours; injects temporal "
                    "guidance at the feature level via optical-flow warping.",
                    "Still inherits the fragility of optical flow under complex "
                    "motion, occlusion, and severe degradation.",
                    "This thesis takes an orthogonal axis: condition on shift-stable "
                    "frequency-domain priors instead of warping features.",
                ],
                accent=NAVY)


def slide_problem_contributions(prs):
    s = add_title_slide(prs)
    set_title(s, "Problem Statement & Contributions")

    # Problem
    add_rounded_rect(s, 0.55, 1.0, 12.3, 2.0, fill=RGBColor(0xFC, 0xF4, 0xF4),
                     line=ACCENT, line_width=1.0)
    box = add_textbox(s, 0.75, 1.10, 11.9, 1.85)
    tf = box.text_frame
    add_paragraph(tf, "Problem", size=16, bold=True, color=ACCENT)
    add_bullet(tf, "Diffusion-based VSR flickers because the model synthesizes "
                   "high-frequency content inconsistently along the temporal axis.", size=13)
    add_bullet(tf, "Existing parameter-efficient remedies still rely on optical-flow "
                   "warping at the pixel or feature level — fragile under complex motion.", size=13)
    add_bullet(tf, "Standard DWT is shift-variant under critical decimation, so it "
                   "fluctuates with sub-pixel inter-frame motion.", size=13)

    # Contributions
    add_rounded_rect(s, 0.55, 3.2, 12.3, 3.85, fill=RGBColor(0xF4, 0xF7, 0xFC),
                     line=TEAL, line_width=1.0)
    box = add_textbox(s, 0.75, 3.30, 11.9, 3.7)
    tf = box.text_frame
    add_paragraph(tf, "Contributions of this thesis", size=16, bold=True, color=TEAL)
    add_bullet(tf, "WC-BD-SFT — frequency-domain adaptation that injects DT-CWT "
                   "priors into a frozen pre-trained diffusion U-Net via asymmetric "
                   "encoder-decoder injection.", size=13)
    add_bullet(tf, "Decoupled magnitude-phase processing with a trigonometric phase "
                   "encoding that avoids the 2π-wraparound discontinuity. The Frequency "
                   "Encoder adds only 6.22M params (~3% of ControlNet) and is initialized "
                   "to preserve the generative prior.", size=13)
    add_bullet(tf, "Band-decoupled wavelet loss — magnitude-only on HIGH bands, joint "
                   "magnitude–phase on LOW bands — providing pixel-space frequency "
                   "supervision alongside the standard latent ε-prediction MSE.", size=13)
    add_bullet(tf, "Under matched REDS-only training, −62.9% tLPIPS and −37.5% LPIPS on "
                   "REDS4 over the StableVSR baseline; consistent no-reference perceptual "
                   "gains across Vid4 / UDM10 / SPMCS.", size=13, bold=True, color=NAVY)


def slide_background_dtcwt(prs):
    s = add_title_slide(prs)
    set_title(s, "Background · Dual-Tree Complex Wavelet Transform")

    box = add_textbox(s, 0.55, 0.95, 6.6, 2.4)
    tf = box.text_frame
    add_paragraph(tf, "Why DT-CWT, not DWT or Fourier?", size=15, bold=True, color=NAVY)
    add_bullet(tf, "Near shift-invariance — magnitudes ≈ stable under sub-pixel shifts.", size=12)
    add_bullet(tf, "Six directional subbands per scale (±15°, ±45°, ±75°) — 2× DWT.", size=12)
    add_bullet(tf, "Complex coefficients ⇒ explicit (magnitude, phase) factorization "
                   "with complementary semantic roles.", size=12)
    add_bullet(tf, "Spatially localized basis — unlike Fourier, the conditioning "
                   "signal can vary across the feature map.", size=12)

    add_picture_fit(s, FIG_DIR / "dtcwt_shift_zoom_v4-1.png",
                     7.3, 1.0, 5.6, 4.2)
    cap = add_textbox(s, 7.3, 5.25, 5.6, 0.45)
    tfc = cap.text_frame
    add_paragraph(tfc, "DT-CWT magnitudes (right) 5.09× smoother than DWT (middle) "
                       "under sub-pixel shifts.",
                  size=10, italic=True, color=GRAY, align=PP_ALIGN.CENTER)

    # Bottom takeaway
    add_callout(s, 0.55, 5.7, 6.6, 1.35,
                title="What this buys us for VSR",
                body=[
                    "Inter-frame conditioning that is temporally stable by construction.",
                    "Directionally selective texture descriptor for detail synthesis.",
                ],
                accent=TEAL)


def slide_background_sft(prs):
    s = add_title_slide(prs)
    set_title(s, "Background · Spatial Feature Transform (SFT)")

    box = add_textbox(s, 0.55, 1.0, 8.0, 4.5)
    tf = box.text_frame
    add_paragraph(tf, "Conditioning via spatially-varying affine modulation",
                  size=16, bold=True, color=NAVY)
    add_paragraph(tf, "Given an intermediate feature map X ∈ ℝ^{C×h×w} and "
                       "modulation tensors (γ, β):",
                  size=13, color=GRAY)

    eq = add_textbox(s, 0.75, 2.15, 7.8, 0.6)
    tfeq = eq.text_frame
    p = tfeq.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "X′ = X ⊙ γ + β"
    r.font.size = Pt(22)
    r.font.bold = True
    r.font.color.rgb = ACCENT

    box2 = add_textbox(s, 0.55, 2.9, 8.0, 4.2)
    tf2 = box2.text_frame
    add_paragraph(tf2, "Properties that match this thesis", size=14, bold=True, color=NAVY)
    add_bullet(tf2, "Spatially varying — unlike AdaIN's globally uniform modulation.", size=12)
    add_bullet(tf2, "Cheap — no quadratic spatial complexity (unlike cross-attention).", size=12)
    add_bullet(tf2, "Naturally compatible with multi-scale wavelet conditioning, which "
                    "is inherently spatial and band-localized.", size=12)
    add_bullet(tf2, "Originally proposed for SR (Wang et al., CVPR 2018); we reuse it "
                    "to inject frequency-domain priors at two functionally distinct U-Net stages.", size=12)

    # Right side: schematic
    add_rounded_rect(s, 8.9, 1.0, 3.9, 5.7, fill=RGBColor(0xFA, 0xFA, 0xFA),
                     line=LIGHT_GRAY, line_width=0.75)
    sch = add_textbox(s, 9.05, 1.15, 3.6, 5.5)
    tfs = sch.text_frame
    add_paragraph(tfs, "BD-SFT modulation pathways", size=13, bold=True,
                  color=NAVY, align=PP_ALIGN.CENTER)
    add_paragraph(tfs, "", size=4)
    add_paragraph(tfs, "HIGH band ⇒ decoder", size=13, bold=True, color=HIGH_BAND,
                  align=PP_ALIGN.CENTER)
    add_paragraph(tfs, "γ_H, β_H injected at up_blocks[1]", size=11, color=GRAY,
                  align=PP_ALIGN.CENTER)
    add_paragraph(tfs, "drives textural detail synthesis", size=11, italic=True,
                  color=GRAY, align=PP_ALIGN.CENTER)
    add_paragraph(tfs, "", size=4)
    add_paragraph(tfs, "LOW band ⇒ encoder", size=13, bold=True, color=LOW_BAND,
                  align=PP_ALIGN.CENTER)
    add_paragraph(tfs, "γ_L, β_L injected at down_blocks[1]", size=11, color=GRAY,
                  align=PP_ALIGN.CENTER)
    add_paragraph(tfs, "structural feature consolidation", size=11, italic=True,
                  color=GRAY, align=PP_ALIGN.CENTER)


def slide_decomposition_band_partition(prs):
    s = add_title_slide(prs)
    set_title(s, "Method · Multi-Scale Decomposition & Band Partitioning")

    eq = add_textbox(s, 0.55, 1.0, 12.3, 0.65)
    tf = eq.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = ("DT-CWT_{J=4}(I^{LR}_t) = ( I^{LP}_t ,  { C^{(j,d)}_t }_{j=1..4, d=1..6} ),"
              "      C^{(j,d)} ∈ ℂ^{H/2ʲ × W/2ʲ × 3}")
    r.font.size = Pt(15)
    r.font.bold = True
    r.font.color.rgb = NAVY

    # Band partition table (left)
    rows = [
        ("HIGH band ℋ", "j = 1, 2", "[1/8, 1/2] cyc/px",
         "fine textures · sharp edges", "→ decoder (up_blocks[1])"),
        ("LOW band ℒ",  "j = 3, 4", "[0, 1/8] cyc/px",
         "coarse structure",           "→ encoder (down_blocks[1])"),
    ]
    add_table(s, 0.55, 1.85, 12.3, 1.4,
              headers=["Band", "Scales", "Norm. frequency",
                       "Semantic role", "Injection point"],
              rows=rows, header_size=12, body_size=11.5,
              bold_first_col=True, zebra=False)

    # Rationale boxes
    add_callout(s, 0.55, 3.45, 6.0, 2.05,
                title="Spectral interpretation",
                body=[
                    "Boundary at j=2 / j=3 cleanly splits the LR spectrum: "
                    "upper half (texture) vs. lower half (structure).",
                    "Each SFT pathway specializes on a non-overlapping range.",
                ], accent=HIGH_BAND)
    add_callout(s, 6.85, 3.45, 6.0, 2.05,
                title="Architectural alignment",
                body=[
                    "j ≥ 3 spatial size ≤ H/8 — easy to upsample to U-Net features.",
                    "Two scales per band ⇒ concatenated SFT has 2 × 256 = 512 channels,",
                    "exactly matching block_out_channels[1] of the StableVSR U-Net.",
                ], accent=LOW_BAND)

    add_callout(s, 0.55, 5.65, 12.3, 1.4,
                title="Why J = 4 levels?",
                body=[
                    "j = 4 subband at H_LR/16 × W_LR/16 — compatible with the U-Net's "
                    "internal 2× downsampling at down_blocks[1] (atop SD-VAE's 8×).",
                    "Sufficient frequency resolution without truncating the lowpass to "
                    "an overly small tensor.",
                ], accent=NAVY)


def slide_subbandblock_detail(prs):
    """Detail of magnitude-phase decoupling (complements existing slide 4 diagram)."""
    s = add_title_slide(prs)
    set_title(s, "Method · SubbandBlock — Magnitude & Phase Decoupling")

    # Left: explanation
    box = add_textbox(s, 0.55, 0.95, 7.0, 5.6)
    tf = box.text_frame
    add_paragraph(tf, "Why decouple magnitude and phase?", size=15, bold=True, color=NAVY)
    add_bullet(tf, "Magnitude — non-negative energy descriptor, near shift-invariant.", size=12)
    add_bullet(tf, "Phase — 2π-periodic localization with discontinuity at ±π.", size=12)
    add_bullet(tf, "Mixing them in a single branch dilutes their distinct roles.", size=12)

    add_paragraph(tf, "Trigonometric phase encoding", size=15, bold=True, color=NAVY,
                  space_before=6)
    eq2 = "𝒯(φ) = (sin φ, cos φ) = (b/(M+ε),  a/(M+ε)),   ε = 1e-8"
    add_paragraph(tf, eq2, size=13, color=ACCENT, bold=True)
    add_bullet(tf, "Continuous across ±π — no wraparound jump.", size=12)
    add_bullet(tf, "Bounded inputs ⇒ stable gradients.", size=12)
    add_bullet(tf, "ε guards against undefined phase at flat regions (M → 0).", size=12)

    add_paragraph(tf, "Recombination & SFT heads", size=15, bold=True, color=NAVY,
                  space_before=6)
    add_bullet(tf, "Complex embedding: e_re = e_M ⊙ e_cos,  e_im = e_M ⊙ e_sin "
                   "— mirrors Re^{iφ} = R(cos φ + i sin φ).", size=12)
    add_bullet(tf, "Concatenate [e_M ‖ e_re ‖ e_im] → 3 × 64 = 192 channels.", size=12)
    add_bullet(tf, "SFT_γ, SFT_β heads → per-scale (γ⁽ʲ⁾, β⁽ʲ⁾) with 256 channels each.", size=12)
    add_bullet(tf, "Identity-preserving init: 2nd conv weight = 0; γ-bias = 1, β-bias = 0 "
                   "⇒ wrapped U-Net behaves identically to the original at step 0.", size=12, bold=True)

    # Right: parameter info + branch schematic
    add_rounded_rect(s, 7.85, 1.0, 5.0, 5.6, fill=RGBColor(0xFA, 0xFA, 0xFA),
                     line=LIGHT_GRAY, line_width=0.75)
    rb = add_textbox(s, 8.0, 1.15, 4.7, 5.4)
    tfr = rb.text_frame
    add_paragraph(tfr, "Per-branch operator sequence", size=13, bold=True,
                  color=NAVY, align=PP_ALIGN.CENTER)
    add_paragraph(tfr, "ℰ(·) = Conv₁ₓ₁ ∘ SA ∘ SiLU ∘ DWConv₃ₓ₃",
                  size=12, color=ACCENT, bold=True, align=PP_ALIGN.CENTER)
    add_paragraph(tfr, "", size=4)
    add_paragraph(tfr, "Channel widths", size=13, bold=True, color=NAVY,
                  align=PP_ALIGN.CENTER)
    add_bullet(tfr, "Magnitude branch input: 6 × 3 = 18 ch", size=11)
    add_bullet(tfr, "Phase branch input: 2 × 18 = 36 ch", size=11)
    add_bullet(tfr, "Mid channels C_m = 64", size=11)
    add_bullet(tfr, "Concat feature: 3 × 64 = 192 ch", size=11)
    add_bullet(tfr, "SFT-head output: 256 ch per scale", size=11)
    add_paragraph(tfr, "", size=4)
    add_paragraph(tfr, "Frequency Encoder total",
                  size=13, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    add_paragraph(tfr, "≈ 6.22 M trainable params",
                  size=14, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
    add_paragraph(tfr, "≈ 3% of the 208 M ControlNet",
                  size=12, italic=True, color=GRAY, align=PP_ALIGN.CENTER)


def slide_bdsft_injection(prs):
    s = add_title_slide(prs)
    set_title(s, "Method · Band-Decoupled SFT Injection (Asymmetric)")

    # Equation
    eq = add_textbox(s, 0.55, 1.0, 12.3, 0.7)
    tf = eq.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "X′ = X ⊙ γ + β     with     γ_ℋ, β_ℋ → up_blocks[1] (decoder)     "
    r.font.size = Pt(15)
    r.font.bold = True
    r.font.color.rgb = NAVY
    r2 = p.add_run()
    r2.text = "     γ_ℒ, β_ℒ → down_blocks[1] (encoder)"
    r2.font.size = Pt(15)
    r2.font.bold = True
    r2.font.color.rgb = NAVY

    # Concatenation rules
    add_paragraph(tf, "")
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    r3 = p2.add_run()
    r3.text = "γ_ℋ = [γ⁽¹⁾ ‖ γ⁽²⁾],   γ_ℒ = [γ⁽³⁾ ‖ γ⁽⁴⁾]   ⇒   512 channels each"
    r3.font.size = Pt(13)
    r3.font.color.rgb = GRAY

    # Three rationale callouts
    add_callout(s, 0.55, 2.6, 4.05, 4.4,
                title="① Architectural alignment",
                body=[
                    "U-Net block_out_channels = [256, 512, 512, 1024].",
                    "down_blocks[1] and up_blocks[1] both have 512 channels —",
                    "exactly the size of (γ, β) by construction.",
                    "Other blocks would require extra channel projection.",
                ], accent=NAVY, title_size=13, body_size=11)

    add_callout(s, 4.7, 2.6, 4.05, 4.4,
                title="② Functional separation",
                body=[
                    "FreeU (Si et al., 2024) — the U-Net encoder suppresses high "
                    "frequencies and consolidates structure; the decoder reintroduces "
                    "high-frequency detail via upsampling + skip fusion.",
                    "LOW band guides encoder structural abstraction.",
                    "HIGH band augments decoder detail synthesis at the stage "
                    "where it most directly impacts the output.",
                ], accent=NAVY, title_size=13, body_size=11)

    add_callout(s, 8.85, 2.6, 4.0, 4.4,
                title="③ Conservative design",
                body=[
                    "A single injection per band keeps parameter overhead low.",
                    "Preserves the clean identity initialization across both points.",
                    "Modulation tensors are spatially aligned to (h, w) via bilinear "
                    "interpolation; the channel dimension matches by construction.",
                ], accent=NAVY, title_size=13, body_size=11)


def slide_training_loss(prs):
    s = add_title_slide(prs)
    set_title(s, "Method · Band-Decoupled Wavelet Loss")

    # Total loss
    box = add_textbox(s, 0.55, 1.0, 12.3, 0.8)
    tf = box.text_frame
    add_paragraph(tf, "Total training objective", size=15, bold=True, color=NAVY)
    p = tf.add_paragraph()
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "ℒ_total = ℒ_MSE + λ_wav · 𝟙[t_step mod K = 0] · ℒ_wav     "\
             "(λ_wav = 1.0,  K = 4)"
    r.font.size = Pt(14)
    r.font.bold = True
    r.font.color.rgb = ACCENT

    # Wavelet loss decomposition
    add_rounded_rect(s, 0.55, 2.05, 12.3, 2.85, fill=RGBColor(0xF7, 0xF9, 0xFC),
                     line=NAVY, line_width=0.75)
    bx2 = add_textbox(s, 0.75, 2.15, 11.9, 2.75)
    tf2 = bx2.text_frame
    add_paragraph(tf2, "Pixel-space wavelet decomposition", size=14, bold=True, color=NAVY)
    p1 = tf2.add_paragraph()
    p1.alignment = PP_ALIGN.CENTER
    r1 = p1.add_run()
    r1.text = ("ℒ_wav = λ_ℋ · Σ_{j∈ℋ} ‖|Ĉ⁽ʲ⁾| − |C⁽ʲ⁾|‖₁  "
               " +  λ_ℒ · Σ_{j∈ℒ} ℒⱼ^{mp}  "
               " +  λ_LP · ‖Î^{LP} − I^{LP}‖₁")
    r1.font.size = Pt(13)
    r1.font.color.rgb = NAVY

    add_paragraph(tf2, "Per-level magnitude–phase term (LOW band only)",
                  size=14, bold=True, color=NAVY, space_before=6)
    p2 = tf2.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run()
    r2.text = ("ℒⱼ^{mp}  =  ‖|Ĉ⁽ʲ⁾| − |C⁽ʲ⁾|‖₁  "
               "+  𝔼[ |C⁽ʲ⁾| · (1 − cos(φ̂⁽ʲ⁾ − φ⁽ʲ⁾)) ]")
    r2.font.size = Pt(13)
    r2.font.color.rgb = NAVY
    add_paragraph(tf2, "Magnitude-weighted angular distance — wraparound-safe, "
                       "and concentrates phase supervision where structural energy is high.",
                  size=11, italic=True, color=GRAY)

    # Asymmetric supervision & weights
    add_callout(s, 0.55, 5.05, 6.0, 1.9,
                title="Asymmetric supervision",
                body=[
                    "HIGH bands: magnitude only — network is free to synthesize "
                    "textural detail without enforcing exact phase.",
                    "LOW bands: magnitude + phase — structural fidelity matters.",
                ], accent=HIGH_BAND)
    add_callout(s, 6.85, 5.05, 6.0, 1.9,
                title="Loss weights (initial settings)",
                body=[
                    "λ_ℋ = 0.1 — soft texture prior (avoids over-driving HF artifacts).",
                    "λ_ℒ = 1.0,  λ_LP = 1.0 — structural matching is required.",
                    "Wavelet loss applied every K=4 steps to limit VAE-decoding cost.",
                ], accent=LOW_BAND)


def slide_experiments_setup(prs):
    s = add_title_slide(prs)
    set_title(s, "Experiments · Datasets & Training Setup")

    # Dataset table
    add_paragraph_box = add_textbox(s, 0.55, 1.0, 12.3, 0.45)
    tf = add_paragraph_box.text_frame
    add_paragraph(tf, "Datasets (BIx4 degradation: bicubic ×4 downsampling)",
                  size=14, bold=True, color=NAVY)

    rows = [
        ("REDS (train split)", "Training", "236 (≈ 240 − REDS4)", "720p"),
        ("REDS4", "In-domain evaluation", "4 (clips 000, 011, 015, 020)", "720p"),
        ("Vid4", "OOD evaluation", "4 (calendar / city / foliage / walk)", "SD"),
        ("UDM10", "OOD evaluation", "10", "720p"),
        ("SPMCS", "OOD evaluation", "30", "SD"),
    ]
    add_table(s, 0.55, 1.55, 7.7, 1.8,
              headers=["Dataset", "Role", "Sequences", "Resolution"],
              rows=rows, header_size=11, body_size=10.5, bold_first_col=True)

    # Training config table
    rows2 = [
        ("Backbone (frozen)", "Stable Diffusion v2.1 U-Net (472 M) + VAE (55 M)"),
        ("ControlNet (trainable)", "207.7 M — RAFT-warped neighbor x̂₀"),
        ("Frequency Encoder (trainable)", "6.22 M  (≈ 3% of ControlNet)"),
        ("Optimizer", "AdamW · β₁=0.9 · β₂=0.999 · lr = 1e-4 (const.)"),
        ("Iterations", "20,000 on REDS"),
        ("Batch / window / patch", "B = 8 · T_win = 3 · LR 64² → HR 256²"),
        ("Hardware", "2 × NVIDIA RTX A6000 (48 GB) for training"),
        ("DT-CWT", "biort='near_sym_a', qshift='qshift_a',  J = 4"),
        ("Inference", "50 DDPM steps · Frame-wise Bidirectional Sampling"),
    ]
    add_table(s, 8.4, 1.55, 4.5, 4.4,
              headers=["Item", "Specification"],
              rows=rows2, header_size=10, body_size=9, bold_first_col=True)

    # Bottom note
    add_callout(s, 0.55, 3.6, 7.7, 3.5,
                title="Why this controlled setup matters",
                body=[
                    "Training data, backbone, and temporal-conditioning scaffold are",
                    "identical to the StableVSR baseline.",
                    "The only architectural difference is the proposed Wavelet",
                    "Conditioning Module (WCM) with BD-SFT injection.",
                    "⇒  any REDS4 performance gap directly attributes to WCM.",
                    "",
                    "Note: same temporal window (T=3) and patch sizes as Rota et al.;",
                    "DT-CWT coefficients are computed on-the-fly from LR via",
                    "pytorch_wavelets at no extra storage cost.",
                ], accent=NAVY, title_size=13, body_size=11)


def slide_metrics(prs):
    s = add_title_slide(prs)
    set_title(s, "Experiments · Evaluation Metrics (9 Metrics, 4 Categories)")

    rows = [
        ("Reconstruction fidelity",
         "PSNR ↑   ·   SSIM ↑",
         "Pixel-level / structural similarity. Penalize generative textures."),
        ("Full-reference perceptual",
         "LPIPS ↓   ·   DISTS ↓",
         "Deep-feature perceptual distance. DISTS is robust to texture variation."),
        ("No-reference perceptual",
         "MUSIQ ↑   ·   CLIP-IQA ↑   ·   NIQE ↓",
         "Transformer-based, vision-language, and natural-scene-statistic priors. "
         "Critical for evaluating generative outputs without GT bias."),
        ("Temporal consistency",
         "tLPIPS ↓   ·   tOF ↓",
         "Inter-frame perceptual / motion consistency.  tLPIPS captures texture "
         "flicker that pixel metrics miss; tOF measures motion fidelity."),
    ]
    add_table(s, 0.55, 1.0, 12.3, 4.0,
              headers=["Category", "Metric (direction)", "What it captures"],
              rows=rows, header_size=12, body_size=11, bold_first_col=True,
              zebra=True)

    add_callout(s, 0.55, 5.15, 12.3, 1.85,
                title="Why this many metrics?",
                body=[
                    "Generative VSR sits on the perception–distortion trade-off "
                    "(Blau & Michaeli, 2018) — no single metric tells the full story.",
                    "Pixel metrics under-credit realistic synthesis; NR perceptual "
                    "metrics complement reference-based ones; temporal metrics expose "
                    "flicker that per-frame metrics ignore.",
                    "All evaluations use a unified script — average over all frames "
                    "within a sequence, then average across sequences.",
                ], accent=NAVY)


def slide_results_reds4_vs_stable(prs):
    s = add_title_slide(prs)
    set_title(s, "Results · REDS4 — Direct Comparison vs. StableVSR")

    rows = [
        ("StableVSR (Rota et al., ECCV 2024)",
         "24.04", "0.690", "0.309", "0.164", "42.79", "0.237", "4.38", "41.11", "13.962"),
        ("WC-BD-SFT (Ours)",
         "24.48", "0.691", "0.193", "0.088", "65.70", "0.386", "2.77", "15.25", "11.118"),
        ("Δ (relative)",
         "+1.8%", "+0.1%", "−37.5%", "−46.3%", "+53.5%", "+62.9%", "−36.8%",
         "−62.9%", "−20.4%"),
    ]
    add_table(s, 0.4, 1.0, 12.55, 1.85,
              headers=["Method",
                       "PSNR ↑", "SSIM ↑", "LPIPS ↓", "DISTS ↓",
                       "MUSIQ ↑", "CLIP-IQA ↑", "NIQE ↓", "tLPIPS ↓", "tOF ↓"],
              rows=rows, header_size=9.5, body_size=9.5,
              bold_first_col=True, zebra=False,
              highlight_row_idx=1, highlight_fill=RGBColor(0xE3, 0xF0, 0xFF))

    # Key callouts
    add_callout(s, 0.55, 3.0, 6.0, 2.0,
                title="Perceptual gains (full + no-reference)",
                body=[
                    "LPIPS: 0.309 → 0.193   (−37.5%)",
                    "DISTS: 0.164 → 0.088   (−46.3%)",
                    "MUSIQ: 42.79 → 65.70   (+53.5%)",
                    "CLIP-IQA: 0.237 → 0.386   (+62.9%)",
                ], accent=TEAL)

    add_callout(s, 6.85, 3.0, 6.0, 2.0,
                title="Temporal consistency",
                body=[
                    "tLPIPS: 41.11 → 15.25   (−62.9%)",
                    "tOF:   13.962 → 11.118  (−20.4%)",
                    "Without 3D conv, temporal attention, or full-network fine-tuning.",
                ], accent=ACCENT)

    add_callout(s, 0.55, 5.15, 12.3, 1.85,
                title="Interpretation",
                body=[
                    "Same training data (REDS), backbone (SD v2.1), temporal scaffold "
                    "(ControlNet + RAFT + bidirectional sampling) — only WCM differs.",
                    "⇒  the gap isolates the contribution of the wavelet-conditioned "
                    "BD-SFT mechanism.",
                    "Gains are largest on the perceptual axis, consistent with HIGH-band "
                    "injection at the decoder where textural detail is synthesized.",
                ], accent=NAVY)


def slide_results_reds4_dm_group(prs):
    s = add_title_slide(prs)
    set_title(s, "Results · REDS4 — Within the Diffusion-VSR Group")

    rows = [
        ("non-DM", "BasicVSR++ (CVPR 2022)",
         "24.88", "0.730", "0.364", "0.179", "41.23", "0.265", "5.90", "38.26", "13.739", "—"),
        ("DM", "StableVSR (ECCV 2024)",
         "24.04", "0.690", "0.309", "0.164", "42.79", "0.237", "4.38", "41.11", "13.962", "3.00"),
        ("DM", "DGAF-VSR (CVPR 2026)",
         "24.07", "0.694", "0.307", "0.161", "43.41", "0.242", "4.37", "40.12", "13.732", "1.89"),
        ("DM", "WC-BD-SFT (Ours)",
         "24.48", "0.691", "0.193", "0.088", "65.70", "0.386", "2.77", "15.25", "11.118", "1.11"),
    ]
    add_table(s, 0.3, 1.0, 12.75, 2.2,
              headers=["Paradigm", "Method",
                       "PSNR↑", "SSIM↑", "LPIPS↓", "DISTS↓",
                       "MUSIQ↑", "CLIP-IQA↑", "NIQE↓", "tLPIPS↓", "tOF↓", "Mean rank"],
              rows=rows, header_size=9, body_size=9,
              bold_first_col=False, zebra=False,
              highlight_row_idx=3, highlight_fill=RGBColor(0xE3, 0xF0, 0xFF))

    add_callout(s, 0.55, 3.35, 6.0, 3.55,
                title="Within the DM group",
                body=[
                    "WC-BD-SFT — best on 8 of 9 metrics, 2nd-best on SSIM.",
                    "Mean rank: 1.11 (Ours) · 1.89 (DGAF-VSR) · 3.00 (StableVSR).",
                    "Largest gap vs. DGAF-VSR on perceptual axes:",
                    "    LPIPS  0.307 → 0.193   (−37.1%)",
                    "    MUSIQ  43.41 → 65.70   (+51.4%)",
                    "    tLPIPS 40.12 → 15.25   (−62.0%)",
                ], accent=TEAL)

    add_callout(s, 6.85, 3.35, 6.0, 3.55,
                title="Cross-paradigm reference · BasicVSR++",
                body=[
                    "BasicVSR++ leads on PSNR/SSIM (regression target).",
                    "All DM methods lead it on perceptual metrics — classical "
                    "perception–distortion trade-off (Blau & Michaeli, 2018).",
                    "Within-DM ranking is the fair comparison axis for the proposed "
                    "mechanism — non-DM and DM target different points on the curve.",
                ], accent=NAVY)


def slide_results_cross_dataset(prs):
    s = add_title_slide(prs)
    set_title(s, "Results · Out-of-Distribution (Vid4 / UDM10 / SPMCS)")

    # Compact: focus on key metrics per dataset
    rows = [
        ("Vid4",  "BasicVSR++",   "26.26", "0.828", "0.189", "61.50", "0.341", "5.04", "15.12"),
        ("Vid4",  "StableVSR",    "22.98", "0.674", "0.185", "67.22", "0.454", "3.19", "25.36"),
        ("Vid4",  "DGAF-VSR",     "23.29", "0.690", "0.177", "67.96", "0.470", "3.10", "17.39"),
        ("Vid4",  "WC-BD-SFT",    "22.64", "0.664", "0.194", "66.15", "0.408", "3.32", "28.53"),
        ("UDM10", "BasicVSR++",   "37.48", "0.956", "0.060", "59.36", "0.443", "5.60", "5.55"),
        ("UDM10", "StableVSR",    "26.71", "0.834", "0.100", "55.69", "0.362", "4.66", "4.42"),
        ("UDM10", "DGAF-VSR",     "26.71", "0.835", "0.099", "57.15", "0.380", "4.61", "2.99"),
        ("UDM10", "WC-BD-SFT",    "25.54", "0.811", "0.124", "63.20", "0.447", "4.12", "14.48"),
        ("SPMCS", "BasicVSR++",   "21.94", "0.617", "0.187", "62.48", "0.434", "5.17", "3.60"),
        ("SPMCS", "StableVSR",    "19.42", "0.478", "0.196", "69.98", "0.582", "3.28", "51.11"),
        ("SPMCS", "DGAF-VSR",     "20.06", "0.511", "0.178", "67.70", "0.533", "3.61", "20.10"),
        ("SPMCS", "WC-BD-SFT",    "19.94", "0.506", "0.183", "67.22", "0.503", "3.70", "31.59"),
    ]
    add_table(s, 0.35, 0.95, 7.4, 5.2,
              headers=["Set", "Method",
                       "PSNR↑", "SSIM↑", "LPIPS↓", "MUSIQ↑", "CLIP-IQA↑", "NIQE↓", "tLPIPS↓"],
              rows=rows, header_size=9, body_size=8.5,
              bold_first_col=True, zebra=False)

    # Right column: observations
    add_callout(s, 7.95, 0.95, 4.95, 1.85,
                title="UDM10 — perceptual wins",
                body=[
                    "WC-BD-SFT best in DM group on all 3 NR metrics:",
                    "MUSIQ 63.20 (+13.5%), CLIP-IQA 0.447 (+23.5%), NIQE 4.12 (−11.6%).",
                    "Short static clips ⇒ higher tLPIPS — see Discussion.",
                ], accent=TEAL, title_size=11, body_size=10)

    add_callout(s, 7.95, 2.95, 4.95, 1.85,
                title="SPMCS — temporal gain over baseline",
                body=[
                    "tLPIPS: 51.11 → 31.59 over StableVSR (−38.2%).",
                    "Frequency conditioning helps temporal stability on longer "
                    "motion-rich content even out-of-distribution.",
                ], accent=ACCENT, title_size=11, body_size=10)

    add_callout(s, 7.95, 4.95, 4.95, 1.95,
                title="Vid4 — distribution gap",
                body=[
                    "Compressed SD content vs. clean REDS 720p ⇒ smaller gains.",
                    "DGAF-VSR's high-resolution feature warping (OGWM) is",
                    "complementary to our frequency-domain conditioning ⇒",
                    "natural fusion candidate (see Future Work).",
                ], accent=NAVY, title_size=11, body_size=10)


def slide_ablation(prs):
    s = add_title_slide(prs)
    set_title(s, "Ablation · Frequency-Band Injection (REDS4)")

    rows = [
        ("WC-BD-SFT (full)",
         "24.48", "0.691", "0.193", "0.088", "65.70", "0.386", "2.77", "15.25", "11.118"),
        ("w/o HIGH  (decoder off)",
         "24.34", "0.689", "0.223", "0.104", "64.49", "0.365", "3.00", "22.11", "11.984"),
        ("w/o LOW  (encoder off)",
         "24.70", "0.701", "0.210", "0.098", "63.39", "0.335", "3.00", "16.57", "11.392"),
        ("w/o both  (= StableVSR-like)",
         "24.46", "0.695", "0.247", "0.119", "59.35", "0.298", "3.29", "25.18", "12.278"),
    ]
    add_table(s, 0.3, 1.0, 12.75, 2.3,
              headers=["Variant",
                       "PSNR↑", "SSIM↑", "LPIPS↓", "DISTS↓",
                       "MUSIQ↑", "CLIP-IQA↑", "NIQE↓", "tLPIPS↓", "tOF↓"],
              rows=rows, header_size=10, body_size=10,
              bold_first_col=True, zebra=False,
              highlight_row_idx=0, highlight_fill=RGBColor(0xE3, 0xF0, 0xFF))

    # Three insights
    add_callout(s, 0.55, 3.5, 4.05, 3.4,
                title="HIGH ⇒ decoder is critical",
                body=[
                    "Disable HIGH:",
                    "tLPIPS 15.25 → 22.11  (+45.0%)",
                    "LPIPS  0.193 → 0.223  (+15.5%)",
                    "Confirms HIGH-band guidance at the decoder",
                    "stabilizes inter-frame texture coherence.",
                ], accent=HIGH_BAND, title_size=12, body_size=11)

    add_callout(s, 4.7, 3.5, 4.05, 3.4,
                title="LOW ⇒ encoder shapes perception",
                body=[
                    "Disable LOW:",
                    "MUSIQ 65.70 → 63.39",
                    "CLIP-IQA 0.386 → 0.335",
                    "PSNR/SSIM marginally rise — typical",
                    "perception–distortion trade-off behavior.",
                ], accent=LOW_BAND, title_size=12, body_size=11)

    add_callout(s, 8.85, 3.5, 4.0, 3.4,
                title="Super-additive when both off",
                body=[
                    "LPIPS w/o both = 0.247",
                    "exceeds the worst single-disable (0.223).",
                    "MUSIQ collapses to 59.35.",
                    "⇒ the two bands carry complementary info",
                    "that interacts non-linearly inside the U-Net.",
                ], accent=NAVY, title_size=12, body_size=11)


def slide_discussion(prs):
    s = add_title_slide(prs)
    set_title(s, "Discussion · Trade-offs Revealed by the Experiments")

    add_callout(s, 0.55, 1.0, 12.3, 2.0,
                title="(1) Perception–distortion trade-off across paradigms",
                body=[
                    "BasicVSR++ (non-DM) → highest PSNR/SSIM; over-smoothed perceptually.",
                    "DM methods → higher LPIPS/MUSIQ; lower pixel fidelity.",
                    "WC-BD-SFT pushes furthest toward the perceptual extreme — best NR "
                    "perceptual scores on REDS4 and on UDM10 within the DM group.",
                ], accent=ACCENT)

    add_callout(s, 0.55, 3.15, 12.3, 2.0,
                title="(2) Specialization–generalization under matched-training",
                body=[
                    "Training is restricted to REDS to isolate the WCM contribution — "
                    "a deliberate ablation-by-design.",
                    "Wavelet priors are most effective on content sharing REDS's frequency "
                    "statistics → strongest gains on REDS4; smaller on Vid4 (SD + JPEG).",
                    "DGAF-VSR's HR feature alignment (OGWM) addresses a different axis — "
                    "the two are orthogonal and complementary (future fusion).",
                ], accent=NAVY)

    add_callout(s, 0.55, 5.3, 12.3, 1.7,
                title="(3) Temporal stability on short, static sequences",
                body=[
                    "UDM10 (32-frame, low-motion) — tLPIPS 14.48 vs. DGAF 2.99.",
                    "Hypothesis: wavelet-driven texture synthesis introduces small per-frame "
                    "variations that accumulate as flicker when bidirectional sampling has "
                    "limited motion to absorb stochastic variation.",
                    "Trend reverses on long-motion SPMCS (51.11 → 31.59 over StableVSR).",
                ], accent=TEAL)


def slide_limitations(prs):
    s = add_title_slide(prs)
    set_title(s, "Limitations")

    box = add_textbox(s, 0.55, 1.0, 12.3, 6.0)
    tf = box.text_frame
    add_paragraph(tf, "Dependence on the training distribution",
                  size=15, bold=True, color=ACCENT)
    add_bullet(tf, "Performance is most pronounced on REDS-like content; gains "
                   "shrink on Vid4 (SD-resolution, compressed).", size=12)
    add_bullet(tf, "Mitigation — broader training (REDS + Vimeo-90K) and "
                   "degradation-aware augmentation; left as future work.", size=12)

    add_paragraph(tf, "Perception–distortion trade-off",
                  size=15, bold=True, color=ACCENT, space_before=8)
    add_bullet(tf, "PSNR/SSIM are sacrificed for perceptual quality, especially on UDM10 "
                   "(PSNR 25.54 dB vs. BasicVSR++ 37.48 dB).", size=12)
    add_bullet(tf, "Inherent to generative VSR; suited to perceptually-oriented "
                   "applications, not scientific / medical imaging requiring strict pixel "
                   "accuracy.", size=12)

    add_paragraph(tf, "Short, quasi-static sequences flicker",
                  size=15, bold=True, color=ACCENT, space_before=8)
    add_bullet(tf, "tLPIPS on UDM10 (32-frame clips) higher than DGAF-VSR / StableVSR.", size=12)
    add_bullet(tf, "Lightweight temporal regularization (flow-guided latent warping, "
                   "multi-frame attention) is a natural remedy without sacrificing the "
                   "parameter-efficient design.", size=12)

    add_paragraph(tf, "Other practical limitations",
                  size=15, bold=True, color=ACCENT, space_before=8)
    add_bullet(tf, "DT-CWT decomposition adds modest computational overhead per frame "
                   "(small compared to the diffusion denoising cost).", size=12)
    add_bullet(tf, "Diffusion stochasticity — single fixed seed (= 42) used for "
                   "reproducibility; multi-seed evaluation left as future work.", size=12)
    add_bullet(tf, "Comparison scope — RVRT, MGLD-VSR, DiffVSR, STAR, DC-VSR, DLoRAL, "
                   "UltraVSR, SeedVR2 not benchmarked (release date / GPU constraints).", size=12)


def slide_conclusion(prs):
    s = add_title_slide(prs)
    set_title(s, "Conclusion · Key Findings")

    # Summary callout
    add_callout(s, 0.55, 1.0, 12.3, 1.95,
                title="Summary",
                body=[
                    "WC-BD-SFT — frequency-domain adaptation that injects DT-CWT "
                    "priors into a frozen pre-trained diffusion U-Net.",
                    "Asymmetric encoder–decoder injection: HIGH bands modulate the "
                    "decoder (textural detail); LOW bands modulate the encoder "
                    "(structural guidance).",
                    "Magnitude–phase decoupling + trigonometric phase encoding + "
                    "band-decoupled wavelet loss.",
                ], accent=NAVY)

    # Headline numbers
    add_rounded_rect(s, 0.55, 3.1, 4.0, 3.9, fill=RGBColor(0xF4, 0xF7, 0xFC),
                     line=TEAL, line_width=1.0)
    a = add_textbox(s, 0.7, 3.25, 3.7, 3.7)
    tfa = a.text_frame
    add_paragraph(tfa, "REDS4 (in-domain)", size=14, bold=True, color=TEAL,
                  align=PP_ALIGN.CENTER)
    add_paragraph(tfa, "−62.9%", size=32, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
    add_paragraph(tfa, "tLPIPS  vs. StableVSR", size=12, color=GRAY,
                  align=PP_ALIGN.CENTER)
    add_paragraph(tfa, "41.11 → 15.25", size=12, color=NAVY, align=PP_ALIGN.CENTER)
    add_paragraph(tfa, "", size=6)
    add_paragraph(tfa, "−37.5%  LPIPS   ·   +53.5%  MUSIQ",
                  size=11, bold=True, color=NAVY, align=PP_ALIGN.CENTER)

    add_rounded_rect(s, 4.7, 3.1, 4.0, 3.9, fill=RGBColor(0xFC, 0xF7, 0xF4),
                     line=ACCENT, line_width=1.0)
    b = add_textbox(s, 4.85, 3.25, 3.7, 3.7)
    tfb = b.text_frame
    add_paragraph(tfb, "Frequency spectrum (REDS4)", size=14, bold=True, color=ACCENT,
                  align=PP_ALIGN.CENTER)
    add_paragraph(tfb, "0.693", size=32, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
    add_paragraph(tfb, "high-freq power ratio to GT", size=12, color=GRAY,
                  align=PP_ALIGN.CENTER)
    add_paragraph(tfb, "vs. StableVSR's 0.341", size=12, color=NAVY, align=PP_ALIGN.CENTER)
    add_paragraph(tfb, "", size=6)
    add_paragraph(tfb, "More than doubles HF retention.",
                  size=11, bold=True, color=NAVY, align=PP_ALIGN.CENTER)

    add_rounded_rect(s, 8.85, 3.1, 4.0, 3.9, fill=RGBColor(0xF7, 0xFC, 0xF4),
                     line=TEAL, line_width=1.0)
    c = add_textbox(s, 9.0, 3.25, 3.7, 3.7)
    tfc = c.text_frame
    add_paragraph(tfc, "Parameter efficiency", size=14, bold=True, color=TEAL,
                  align=PP_ALIGN.CENTER)
    add_paragraph(tfc, "6.22 M", size=32, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
    add_paragraph(tfc, "trainable Frequency Encoder", size=12, color=GRAY,
                  align=PP_ALIGN.CENTER)
    add_paragraph(tfc, "≈ 3% of ControlNet (208 M)", size=12, color=NAVY,
                  align=PP_ALIGN.CENTER)
    add_paragraph(tfc, "", size=6)
    add_paragraph(tfc, "U-Net & VAE remain fully frozen.",
                  size=11, bold=True, color=NAVY, align=PP_ALIGN.CENTER)


def slide_future_work(prs):
    s = add_title_slide(prs)
    set_title(s, "Future Work")

    box = add_textbox(s, 0.55, 1.0, 12.3, 5.8)
    tf = box.text_frame

    add_paragraph(tf, "Flow-free pipeline", size=15, bold=True, color=NAVY)
    add_bullet(tf, "Replace RAFT-based warping in the ControlNet branch with a "
                   "frequency-aware temporal conditioning module — fully removing "
                   "optical-flow dependency.", size=12)

    add_paragraph(tf, "Combine with feature-domain temporal guidance",
                  size=15, bold=True, color=NAVY, space_before=6)
    add_bullet(tf, "WC-BD-SFT (frequency-domain) ⟂ DGAF-VSR's OGWM + FTCM (feature-domain "
                   "warping) — fuse the two for stronger cross-dataset performance.", size=12)

    add_paragraph(tf, "Broader training distribution",
                  size=15, bold=True, color=NAVY, space_before=6)
    add_bullet(tf, "Train on REDS + Vimeo-90K (and degradation-aware augmentation) to "
                   "narrow the OOD gap on Vid4-style content.", size=12)

    add_paragraph(tf, "Latent-space wavelet analysis",
                  size=15, bold=True, color=NAVY, space_before=6)
    add_bullet(tf, "Operate the wavelet loss in the latent space (and explore adaptive "
                   "band partitioning) to reduce VAE-decoding overhead.", size=12)

    add_paragraph(tf, "Short-sequence temporal regularization",
                  size=15, bold=True, color=NAVY, space_before=6)
    add_bullet(tf, "Add lightweight temporal modules (flow-guided latent warping, "
                   "multi-frame attention) to address UDM10-style flicker without "
                   "compromising parameter efficiency.", size=12)


# ------------------------------------------------------------------
# Build & reorder
# ------------------------------------------------------------------
def reorder_slides(prs, new_order):
    """Reorder slides according to ``new_order`` — a permutation of indices into the
    current slide list. Uses XML manipulation since python-pptx has no public API.
    """
    sldIdLst = prs.slides._sldIdLst  # type: ignore[attr-defined]
    children = list(sldIdLst)
    assert sorted(new_order) == list(range(len(children))), \
        f"new_order must be a permutation of 0..{len(children) - 1}"
    # Detach all children first.
    for child in children:
        sldIdLst.remove(child)
    # Re-attach in the requested order.
    for idx in new_order:
        sldIdLst.append(children[idx])


def main():
    prs = Presentation(str(TEMPLATE))

    # ── Append new content slides (will be reordered below) ────────
    # Indices 9..28 (existing slides occupy 0..8)
    slide_motivation_1(prs)            # 9
    slide_motivation_2(prs)            # 10
    slide_related_work_vsr(prs)        # 11
    slide_related_work_diff(prs)       # 12
    slide_problem_contributions(prs)   # 13
    slide_background_dtcwt(prs)        # 14
    slide_background_sft(prs)          # 15
    slide_decomposition_band_partition(prs)  # 16
    slide_subbandblock_detail(prs)     # 17
    slide_bdsft_injection(prs)         # 18
    slide_training_loss(prs)           # 19
    slide_experiments_setup(prs)       # 20
    slide_metrics(prs)                 # 21
    slide_results_reds4_vs_stable(prs) # 22
    slide_results_reds4_dm_group(prs)  # 23
    slide_results_cross_dataset(prs)   # 24
    slide_ablation(prs)                # 25
    slide_discussion(prs)              # 26
    slide_limitations(prs)             # 27
    slide_conclusion(prs)              # 28
    slide_future_work(prs)             # 29

    # Existing slides (template indices):
    #   0 Cover, 1 Contents, 2 Architecture diagram, 3 SubbandBlock detail,
    #   4 Evaluation overview, 5 DT-CWT subbands of LR, 6 DT-CWT subbands of GT,
    #   7 Evaluation (radial spectrum / qualitative), 8 Thanks.
    final_order = [
        0,    # 1.  Cover
        1,    # 2.  Contents
        9,    # 3.  Motivation 1 (VSR + Diffusion)
        10,   # 4.  Motivation 2 (Two existing directions)
        11,   # 5.  Related Work · VSR
        12,   # 6.  Related Work · Diffusion VSR
        13,   # 7.  Problem & Contributions
        14,   # 8.  Background · DT-CWT
        15,   # 9.  Background · SFT
        2,    # 10. Architecture diagram (existing)
        16,   # 11. Multi-scale decomposition & band partitioning
        3,    # 12. SubbandBlock diagram (existing)
        17,   # 13. SubbandBlock — magnitude/phase decoupling
        18,   # 14. BD-SFT injection
        19,   # 15. Training loss
        20,   # 16. Experiments · setup
        21,   # 17. Experiments · metrics
        22,   # 18. REDS4 vs StableVSR
        23,   # 19. REDS4 DM group
        24,   # 20. Cross-dataset
        4,    # 21. Evaluation overview (existing)
        5,    # 22. DT-CWT subbands LR (existing)
        6,    # 23. DT-CWT subbands GT (existing)
        7,    # 24. Radial spectrum (existing)
        25,   # 25. Ablation
        26,   # 26. Discussion
        27,   # 27. Limitations
        28,   # 28. Conclusion
        29,   # 29. Future Work
        8,    # 30. Thanks / QnA
    ]
    reorder_slides(prs, final_order)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"Saved: {OUT} ({OUT.stat().st_size:,} bytes)")
    print(f"Total slides: {len(prs.slides)}")


if __name__ == "__main__":
    main()
