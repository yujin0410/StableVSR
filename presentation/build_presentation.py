"""Build v2 of the Master's thesis defense deck.

Changes vs. v1 (from user review):
  1. Cover — replace "SEMINAR" with the full thesis title.
  2. Contents — numbered sections starting from "Introduction"; show subsections.
  3. Title text on content slides made WHITE (the layout has a dark navy bar at
     y=0 that hid the previous navy title).
  4. Delete the layout's default body placeholder ("마스터 텍스트 스타일을 편집합니다")
     on every new content slide.
  5. Visual refresh — section-coded accent colors, numbered badges, more varied
     cards, and roughly 35% less prose so the slides read at a glance.
  6. Speaker notes added to every slide.
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

THESIS_TITLE = ("Wavelet-Conditioned Band-Decoupled Spatial Feature Transform "
                "for Diffusion-Based Video Super-Resolution")

# ── Palette ────────────────────────────────────────────────────────
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x12, 0x1E, 0x3D)       # near-black ink for body text
DEEP = RGBColor(0x15, 0x12, 0x7C)      # template's navy bar — primary identity
SOFT_BG = RGBColor(0xF6, 0xF4, 0xFB)   # very soft lavender for card fills
LIGHT_GRAY = RGBColor(0xE5, 0xE5, 0xE5)
GRAY = RGBColor(0x6B, 0x6B, 0x78)
MUTED_INK = RGBColor(0x3D, 0x3D, 0x4A)

# Section accent colors — varied but cohesive (each section gets a hue)
SEC = {
    1: RGBColor(0x2C, 0x55, 0xA8),   # Introduction — blue
    2: RGBColor(0x0F, 0x86, 0x8E),   # Background — teal
    3: RGBColor(0x6A, 0x2E, 0xA6),   # Method — purple (primary content)
    4: RGBColor(0xC5, 0x8A, 0x12),   # Experiments — gold
    5: RGBColor(0xCB, 0x42, 0x35),   # Results & Ablation — coral red
    6: RGBColor(0x1F, 0x6E, 0x5C),   # Conclusion — forest green
}
# Frequency-band colors stay consistent across slides
HIGH_BAND = RGBColor(0x2C, 0x55, 0xA8)
LOW_BAND = RGBColor(0xCB, 0x42, 0x35)


# ── XML helper ──────────────────────────────────────────────────────
def _remove_shape(shape):
    sp = shape._element
    sp.getparent().remove(sp)


def add_content_slide(prs, section: int):
    """Add a slide using layout 12 ('16_제목만') and strip the body placeholder.

    Returns: (slide, section_accent_color)
    """
    slide = prs.slides.add_slide(prs.slide_layouts[12])
    # Layout 12 brings in a body placeholder (idx=10) with default Korean text.
    # We do not use it — remove it so the placeholder text never renders.
    for ph in list(slide.placeholders):
        if ph.placeholder_format.idx == 10:
            _remove_shape(ph)
    return slide, SEC[section]


def set_section_title(slide, section: int, title: str, subtitle: str = ""):
    """Write the slide title onto the layout's dark navy bar (white text)."""
    # Locate the title placeholder (idx=1 on layout 12).
    ph = None
    for cand in slide.placeholders:
        if cand.placeholder_format.idx == 1:
            ph = cand
            break
    if ph is None:
        ph = slide.shapes.add_textbox(Inches(0.45), Inches(0.0),
                                       Inches(12.44), Inches(0.6))

    tf = ph.text_frame
    tf.clear()
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.05)
    tf.margin_top = Inches(0.0)
    tf.margin_bottom = Inches(0.0)

    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    p.space_before = Pt(0)
    p.space_after = Pt(0)

    badge = p.add_run()
    badge.text = f"  {section}  "
    badge.font.size = Pt(13)
    badge.font.bold = True
    badge.font.color.rgb = SEC[section]
    # We can't easily set a per-run highlight; instead add a small filled
    # rounded rectangle behind the section number visually (see below).

    sep = p.add_run()
    sep.text = "   "
    sep.font.size = Pt(20)

    main = p.add_run()
    main.text = title
    main.font.size = Pt(22)
    main.font.bold = True
    main.font.color.rgb = WHITE

    if subtitle:
        sub = p.add_run()
        sub.text = f"   ·   {subtitle}"
        sub.font.size = Pt(14)
        sub.font.bold = False
        sub.font.color.rgb = RGBColor(0xDD, 0xD8, 0xF0)

    # Section-number badge — a small filled circle to the right of the title bar
    # (placed at the right edge for a clean visual anchor).
    badge_box = slide.shapes.add_shape(
        MSO_SHAPE.OVAL,
        Inches(12.55), Inches(0.10), Inches(0.55), Inches(0.40),
    )
    badge_box.fill.solid()
    badge_box.fill.fore_color.rgb = SEC[section]
    badge_box.line.fill.background()
    badge_box.shadow.inherit = False
    btf = badge_box.text_frame
    btf.text = ""
    btf.margin_left = Inches(0)
    btf.margin_right = Inches(0)
    btf.margin_top = Inches(0.02)
    btf.margin_bottom = Inches(0.0)
    btf.vertical_anchor = MSO_ANCHOR.MIDDLE
    bp = btf.paragraphs[0]
    bp.alignment = PP_ALIGN.CENTER
    br = bp.add_run()
    br.text = str(section)
    br.font.size = Pt(15)
    br.font.bold = True
    br.font.color.rgb = WHITE


# ── Generic primitives ─────────────────────────────────────────────
def add_textbox(slide, left, top, width, height, *, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(left), Inches(top),
                                    Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = Inches(0.06)
    tf.margin_right = Inches(0.06)
    tf.margin_top = Inches(0.04)
    tf.margin_bottom = Inches(0.04)
    return box


def add_para(tf, text, *, size=14, bold=False, italic=False, color=INK,
              align=PP_ALIGN.LEFT, space_before=2, space_after=4):
    if not tf.text and not tf.paragraphs[0].runs:
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    if color is not None:
        r.font.color.rgb = color
    return p


def add_card(slide, left, top, width, height, *, fill=SOFT_BG, line=None,
              line_width=0.75, radius_corner=True):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius_corner else MSO_SHAPE.RECTANGLE,
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
    shape.text_frame.text = ""
    return shape


def add_accent_strip(slide, left, top, width, color, *, thickness=0.04):
    """Thin horizontal accent line under a card header."""
    s = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(left), Inches(top), Inches(width), Inches(thickness),
    )
    s.fill.solid()
    s.fill.fore_color.rgb = color
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def add_numbered_card(slide, left, top, width, height, *, number, header,
                       lines, accent, header_size=13, body_size=11):
    """Card with a colored numbered circle, header, and 2-3 short lines."""
    add_card(slide, left, top, width, height, fill=WHITE, line=LIGHT_GRAY,
              line_width=0.75)
    # Number badge — a small filled circle
    badge_d = 0.45
    add_card(slide, left + 0.16, top + 0.15, badge_d, badge_d, fill=accent,
              line=None, radius_corner=False)  # rectangle disabled below
    # Make it an oval instead
    _ = None
    # Replace the rectangle with an oval — easier to just add one fresh.
    badge_shape = slide.shapes.add_shape(
        MSO_SHAPE.OVAL,
        Inches(left + 0.16), Inches(top + 0.15),
        Inches(badge_d), Inches(badge_d),
    )
    badge_shape.fill.solid()
    badge_shape.fill.fore_color.rgb = accent
    badge_shape.line.fill.background()
    badge_shape.shadow.inherit = False
    btf = badge_shape.text_frame
    btf.margin_left = Inches(0)
    btf.margin_right = Inches(0)
    btf.margin_top = Inches(0.02)
    btf.margin_bottom = Inches(0)
    btf.vertical_anchor = MSO_ANCHOR.MIDDLE
    bp = btf.paragraphs[0]
    bp.alignment = PP_ALIGN.CENTER
    br = bp.add_run()
    br.text = str(number)
    br.font.size = Pt(13)
    br.font.bold = True
    br.font.color.rgb = WHITE
    # Remove the rectangle "filler" we made (the first add_card placed an
    # extra rect at the same spot — remove it).
    # Iterate shapes and remove the rectangle exactly under the oval.
    target_l = Inches(left + 0.16)
    target_t = Inches(top + 0.15)
    for sh in list(slide.shapes):
        if sh is badge_shape:
            continue
        if (sh.shape_type == 1 and sh.left == target_l and sh.top == target_t
                and sh.width == Inches(badge_d) and sh.height == Inches(badge_d)):
            _remove_shape(sh)
            break

    # Header text
    hdr = add_textbox(slide, left + 0.78, top + 0.12, width - 0.95, 0.5)
    add_para(hdr.text_frame, header, size=header_size, bold=True, color=INK)
    add_accent_strip(slide, left + 0.18, top + 0.66, width - 0.32, accent,
                      thickness=0.03)

    # Body lines
    body = add_textbox(slide, left + 0.18, top + 0.78, width - 0.32,
                       height - 0.88)
    for line in lines:
        add_para(body.text_frame, line, size=body_size, color=MUTED_INK,
                  space_before=2, space_after=4)


def add_kpi_card(slide, left, top, width, height, *, label, value, sub,
                  accent):
    """Big-number stat card used on the Conclusion slide."""
    add_card(slide, left, top, width, height, fill=WHITE, line=accent,
              line_width=1.25)
    add_accent_strip(slide, left, top, width, accent, thickness=0.06)
    tb = add_textbox(slide, left + 0.1, top + 0.18, width - 0.2, height - 0.28)
    add_para(tb.text_frame, label, size=12, bold=True, color=accent,
              align=PP_ALIGN.CENTER, space_after=2)
    add_para(tb.text_frame, value, size=34, bold=True, color=INK,
              align=PP_ALIGN.CENTER, space_before=2, space_after=2)
    add_para(tb.text_frame, sub, size=11, color=GRAY,
              align=PP_ALIGN.CENTER, space_before=0, space_after=0)


def add_picture_fit(slide, path, left, top, width, height, *, center=True):
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
              header_size=10.5, body_size=10, header_fill=DEEP,
              header_color=WHITE, zebra=True, highlight_row_idx=None,
              highlight_fill=RGBColor(0xFF, 0xF4, 0xCC)):
    n_rows = len(rows) + 1
    n_cols = len(headers)
    tbl_shape = slide.shapes.add_table(n_rows, n_cols, Inches(left),
                                        Inches(top), Inches(width),
                                        Inches(height))
    tbl = tbl_shape.table
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
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.cell(i + 1, j)
            cell.margin_left = Inches(0.04)
            cell.margin_right = Inches(0.04)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            if highlight_row_idx is not None and i == highlight_row_idx:
                cell.fill.solid()
                cell.fill.fore_color.rgb = highlight_fill
            elif zebra and i % 2 == 1:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(0xF7, 0xF7, 0xFB)
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE
            tf = cell.text_frame
            tf.word_wrap = True
            tf.text = ""
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT
            r = p.add_run()
            r.text = str(val)
            r.font.size = Pt(body_size)
            r.font.color.rgb = INK
            if highlight_row_idx is not None and i == highlight_row_idx:
                r.font.bold = True
    return tbl_shape


def set_notes(slide, text: str):
    """Set speaker notes for a slide."""
    notes_tf = slide.notes_slide.notes_text_frame
    notes_tf.clear()
    # Split into paragraphs on blank lines so the note reads as a script.
    parts = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    for idx, part in enumerate(parts):
        p = notes_tf.paragraphs[0] if idx == 0 else notes_tf.add_paragraph()
        r = p.add_run()
        r.text = part
        r.font.size = Pt(11)


# ──────────────────────────────────────────────────────────────────────────
# Cover & contents
# ──────────────────────────────────────────────────────────────────────────
def update_cover(prs):
    """Slide 1 already has cover styling. Replace 'SEMINAR' with thesis title."""
    slide = prs.slides[0]
    for sh in slide.shapes:
        if not sh.has_text_frame:
            continue
        txt = sh.text_frame.text.strip()
        if txt == "SEMINAR":
            tf = sh.text_frame
            # Use a smaller font that fits the long title; preserve placement.
            tf.clear()
            tf.word_wrap = True
            # Two-line wrap: split at "Spatial Feature Transform"
            line1 = "Wavelet-Conditioned Band-Decoupled Spatial Feature Transform"
            line2 = "for Diffusion-Based Video Super-Resolution"
            p1 = tf.paragraphs[0]
            p1.alignment = PP_ALIGN.LEFT
            r1 = p1.add_run()
            r1.text = line1
            r1.font.size = Pt(28)
            r1.font.bold = True
            r1.font.color.rgb = INK
            p2 = tf.add_paragraph()
            p2.alignment = PP_ALIGN.LEFT
            p2.space_before = Pt(4)
            r2 = p2.add_run()
            r2.text = line2
            r2.font.size = Pt(28)
            r2.font.bold = True
            r2.font.color.rgb = INK
            # Widen the textbox so the long title fits on two lines instead of
            # wrapping awkwardly.
            sh.width = Inches(11.5)
            break

    set_notes(slide, (
        "Good morning / afternoon. I am Yu-Jin Cho from the Department of IT "
        "Engineering at Sookmyung Women's University.\n\n"
        "Today I will present my Master's thesis: 'Wavelet-Conditioned "
        "Band-Decoupled Spatial Feature Transform for Diffusion-Based Video "
        "Super-Resolution.'\n\n"
        "This work proposes a parameter-efficient frequency-domain "
        "conditioning module that improves the temporal consistency of "
        "diffusion-based VSR without retraining the pre-trained backbone."
    ))


def update_contents(prs):
    """Rebuild the Contents textbox with numbered sections and subsections."""
    slide = prs.slides[1]
    # Find the listing textbox.
    target = None
    for sh in slide.shapes:
        if sh.name == "TextBox 5" and sh.has_text_frame:
            target = sh
            break
    if target is None:
        return

    # Widen to fit subsections.
    target.left = Inches(4.2)
    target.top = Inches(1.95)
    target.width = Inches(8.5)
    target.height = Inches(5.0)

    tf = target.text_frame
    tf.clear()
    tf.word_wrap = True

    sections = [
        ("1.  Introduction",     "Motivation · Related Work · Contributions"),
        ("2.  Background",       "DT-CWT · Spatial Feature Transform (SFT)"),
        ("3.  Method",           "Architecture · SubbandBlock · BD-SFT Injection · Training Loss"),
        ("4.  Experiments",      "Datasets & Setup · Evaluation Metrics"),
        ("5.  Results & Ablation",
                                 "REDS4 · Cross-Dataset · Band Injection Ablation · Discussion"),
        ("6.  Conclusion",       "Key Findings · Limitations · Future Work"),
    ]

    for idx, (title, sub) in enumerate(sections):
        # Title line
        p_t = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p_t.alignment = PP_ALIGN.LEFT
        p_t.space_before = Pt(10 if idx > 0 else 0)
        p_t.space_after = Pt(2)
        rt = p_t.add_run()
        rt.text = title
        rt.font.size = Pt(18)
        rt.font.bold = True
        rt.font.color.rgb = SEC[idx + 1]
        # Sub-line
        p_s = tf.add_paragraph()
        p_s.alignment = PP_ALIGN.LEFT
        p_s.space_before = Pt(0)
        p_s.space_after = Pt(0)
        # indent the sub-line visually with extra spaces
        rs = p_s.add_run()
        rs.text = "         " + sub
        rs.font.size = Pt(11)
        rs.font.italic = True
        rs.font.color.rgb = GRAY

    set_notes(slide, (
        "The presentation is organised into six sections.\n\n"
        "Sections 1 and 2 cover the motivation, related work, and the "
        "theoretical background needed for the proposed method — DT-CWT "
        "and Spatial Feature Transform.\n\n"
        "Section 3 details the proposed WC-BD-SFT framework: the Wavelet "
        "Conditioning Module, the SubbandBlock, the asymmetric BD-SFT "
        "injection, and the band-decoupled wavelet loss.\n\n"
        "Sections 4 and 5 present the experiments and quantitative / "
        "qualitative results, including a controlled comparison against "
        "the StableVSR baseline and an ablation study.\n\n"
        "Section 6 summarizes findings, limitations and future work."
    ))


# ──────────────────────────────────────────────────────────────────────────
# Content slide builders — every slide gets speaker notes
# ──────────────────────────────────────────────────────────────────────────
def slide_motivation(prs):
    s, accent = add_content_slide(prs, section=1)
    set_section_title(s, 1, "Motivation",
                       subtitle="Why VSR + Diffusion, and why is it hard?")

    # Three numbered cards across the top half
    cards = [
        (1, "Demand for HR video",
         ["Multimedia platforms drive HR video demand.",
          "VSR recovers HR from LR using spatial + temporal cues."]),
        (2, "Diffusion priors win on detail",
         ["CNN/GAN VSR — over-smoothing or unstable training.",
          "Diffusion priors synthesize realistic textures (SR3, DDPM)."]),
        (3, "But: temporal flickering",
         ["Per-frame stochastic denoising → inconsistent HF detail.",
          "Restored textures shimmer between frames → key bottleneck."]),
    ]
    x = 0.55
    for i, (n, header, lines) in enumerate(cards):
        add_numbered_card(s, x + i * 4.16, 1.0, 4.0, 2.7, number=n,
                           header=header, lines=lines, accent=accent)

    # Lower band — research question
    add_card(s, 0.55, 4.0, 12.3, 2.4, fill=SOFT_BG, line=accent, line_width=1.0)
    add_accent_strip(s, 0.55, 4.0, 12.3, accent, thickness=0.05)
    tb = add_textbox(s, 0.85, 4.18, 11.7, 2.1)
    add_para(tb.text_frame, "Research question",
              size=14, bold=True, color=accent)
    add_para(tb.text_frame,
              "Can a parameter-efficient module make a pre-trained image "
              "diffusion model temporally consistent for VSR — without 3D "
              "attention or full-network fine-tuning?",
              size=16, bold=True, color=INK, space_before=4)
    add_para(tb.text_frame,
              "Key idea — condition the U-Net on shift-stable frequency-domain "
              "priors, not warped pixels.",
              size=13, italic=True, color=GRAY, space_before=8)

    set_notes(s, (
        "Three motivating points.\n\n"
        "First, HR video demand keeps growing. VSR aggregates spatial and "
        "temporal information from low-resolution frames to recover high "
        "resolution.\n\n"
        "Second, diffusion-based generative priors outperform CNN and GAN "
        "approaches in perceptual detail. So a natural question is whether "
        "we can bring image-trained diffusion priors to video.\n\n"
        "Third — and this is the bottleneck — applying image diffusion frame "
        "by frame produces temporal flickering, because high-frequency "
        "detail is synthesized stochastically and independently per frame.\n\n"
        "The research question I address is: can a *lightweight* module make "
        "the pre-trained image diffusion model temporally consistent for "
        "VSR? My answer is yes — by conditioning on shift-stable "
        "frequency-domain priors rather than warped pixels or features."
    ))


def slide_related_work(prs):
    s, accent = add_content_slide(prs, section=1)
    set_section_title(s, 1, "Related Work", subtitle="VSR & diffusion-based VSR")

    # Left timeline-style column
    add_card(s, 0.55, 1.0, 6.0, 5.95, fill=WHITE, line=LIGHT_GRAY)
    add_accent_strip(s, 0.55, 1.0, 6.0, accent, thickness=0.06)
    L = add_textbox(s, 0.7, 1.15, 5.7, 5.7)
    add_para(L.text_frame, "VSR landscape", size=14, bold=True, color=accent)

    # Mini-timeline rows
    rows = [
        ("CNN · alignment", "TOFlow · EDVR · BasicVSR/++ · VRT"),
        ("CNN · real-world", "RealBasicVSR (degradation augmentation)"),
        ("GAN", "SRGAN / ESRGAN / VideoGigaGAN (2024)"),
        ("Diffusion · image-prior", "StableVSR (baseline) · DGAF-VSR · MGLD-VSR"),
        ("Diffusion · video-native", "Upscale-A-Video · Stable Video Diffusion"),
        ("Recent 2025", "DiffVSR · STAR · DC-VSR · DLoRAL · SeedVR2"),
    ]
    for label, names in rows:
        p = L.text_frame.add_paragraph()
        p.space_before = Pt(4)
        r1 = p.add_run()
        r1.text = f"▸  {label}  "
        r1.font.size = Pt(12)
        r1.font.bold = True
        r1.font.color.rgb = INK
        r2 = p.add_run()
        r2.text = names
        r2.font.size = Pt(11)
        r2.font.color.rgb = GRAY

    # Right — gap analysis
    add_card(s, 6.85, 1.0, 6.0, 5.95, fill=SOFT_BG, line=accent, line_width=1.0)
    add_accent_strip(s, 6.85, 1.0, 6.0, accent, thickness=0.06)
    R = add_textbox(s, 7.05, 1.15, 5.6, 5.7)
    add_para(R.text_frame, "Closest comparator · StableVSR",
              size=14, bold=True, color=accent)
    add_para(R.text_frame,
              "ControlNet + RAFT-warped x̂₀ of a neighbor frame, "
              "with bidirectional sampling.",
              size=12, color=MUTED_INK)
    add_para(R.text_frame, "Gap we identified", size=13, bold=True,
              color=INK, space_before=10)
    for line in [
        "• Optical-flow warping is fragile under complex motion, occlusion.",
        "• Standard DWT is shift-variant ⇒ inconsistent inter-frame conditioning.",
        "• Full-network fine-tuning risks catastrophic forgetting of the prior.",
    ]:
        add_para(R.text_frame, line, size=12, color=MUTED_INK,
                  space_before=2, space_after=2)
    add_para(R.text_frame, "Our axis — frequency-domain conditioning",
              size=13, bold=True, color=accent, space_before=10)
    add_para(R.text_frame,
              "Shift-stable DT-CWT priors injected into the frozen U-Net via "
              "asymmetric SFT — orthogonal to feature-warping methods.",
              size=12, color=MUTED_INK)

    set_notes(s, (
        "Briefly on the landscape. CNN-based VSR went from explicit flow "
        "(TOFlow), to deformable alignment (EDVR), to recurrent propagation "
        "(BasicVSR / BasicVSR++). Transformer variants like VRT, RVRT push "
        "further on long-range temporal attention.\n\n"
        "Diffusion-based VSR splits into two directions. The first adapts "
        "pre-trained image diffusion — StableVSR is the canonical example "
        "and is the direct baseline of this thesis. DGAF-VSR (CVPR 2026) "
        "is the most recent comparator, using feature-domain temporal "
        "guidance via optical-flow warping. The second direction trains "
        "video-native diffusion from scratch, which is prohibitively "
        "expensive.\n\n"
        "The gap I target is the conditioning signal: existing "
        "parameter-efficient methods still depend on warped pixels or "
        "features. Optical flow is fragile, and standard DWT-based "
        "conditioning is shift-variant — both create temporal "
        "inconsistency. I propose frequency-domain conditioning via "
        "DT-CWT, which is shift-stable by construction."
    ))


def slide_contributions(prs):
    s, accent = add_content_slide(prs, section=1)
    set_section_title(s, 1, "Contributions")

    contribs = [
        (1, "WC-BD-SFT framework",
         ["Frequency-domain adaptation injecting DT-CWT priors into a frozen "
          "pre-trained diffusion U-Net.",
          "Asymmetric encoder-decoder injection — HIGH bands → decoder "
          "(texture), LOW bands → encoder (structure)."]),
        (2, "Decoupled magnitude-phase processing",
         ["Trigonometric phase encoding avoids the 2π-wraparound "
          "discontinuity.",
          "Frequency Encoder = 6.22 M params (≈3% of ControlNet); "
          "identity-preserving initialization."]),
        (3, "Band-decoupled wavelet loss",
         ["Magnitude-only on HIGH bands; magnitude-weighted angular distance "
          "on LOW bands.",
          "Pixel-space frequency supervision alongside latent ε-MSE."]),
        (4, "Strong empirical gains (REDS4)",
         ["−62.9 % tLPIPS  (41.11 → 15.25)",
          "−37.5 % LPIPS   (0.309 → 0.193)",
          "+103 % HF spectral power retention vs. StableVSR."]),
    ]
    for i, (n, header, lines) in enumerate(contribs):
        col = i % 2
        row = i // 2
        x = 0.55 + col * 6.3
        y = 1.0 + row * 2.95
        add_numbered_card(s, x, y, 6.05, 2.75, number=n, header=header,
                           lines=lines, accent=accent,
                           header_size=14, body_size=11)

    set_notes(s, (
        "Four contributions.\n\n"
        "First — the WC-BD-SFT framework itself. The novelty is asymmetric "
        "band-decoupled injection: high-frequency wavelet bands go to the "
        "U-Net decoder where textures are synthesized; low-frequency bands "
        "go to the encoder where structural abstraction happens.\n\n"
        "Second — magnitude-phase decoupling. Phase is encoded as (sin, cos) "
        "to be continuous across ±π. The resulting Frequency Encoder is "
        "only 6.22 M parameters and is initialized to preserve the "
        "pre-trained prior at step zero.\n\n"
        "Third — the band-decoupled wavelet loss. HIGH bands are supervised "
        "in magnitude only so the network can synthesize freely; LOW bands "
        "include a magnitude-weighted phase term for structural fidelity.\n\n"
        "Fourth — the empirical headline. Under matched REDS-only training, "
        "we cut tLPIPS by 63 %, LPIPS by 37.5 %, and more than double the "
        "high-frequency spectral retention versus StableVSR."
    ))


def slide_background_dtcwt(prs):
    s, accent = add_content_slide(prs, section=2)
    set_section_title(s, 2, "Dual-Tree Complex Wavelet Transform",
                       subtitle="Why DT-CWT — not DWT or Fourier")

    # Image left
    add_picture_fit(s, FIG_DIR / "dtcwt_shift_zoom_v4-1.png",
                     0.55, 1.0, 6.6, 5.4)
    cap = add_textbox(s, 0.55, 6.45, 6.6, 0.45)
    add_para(cap.text_frame,
              "DT-CWT magnitudes 5.09× more stable than DWT under sub-pixel shifts.",
              size=10, italic=True, color=GRAY, align=PP_ALIGN.CENTER)

    # Right — three property cards
    props = [
        ("Near shift-invariance",
         "Magnitudes stable under sub-pixel shifts → temporally consistent "
         "conditioning between adjacent frames."),
        ("6 directional subbands",
         "±15° / ±45° / ±75° at each scale — 2× DWT's directional "
         "resolution; preserves edge orientation."),
        ("Magnitude + phase",
         "Complex coefficients factor into shift-stable magnitude (texture) "
         "and localized phase (structure)."),
    ]
    for i, (h, body) in enumerate(props):
        y = 1.0 + i * 1.85
        add_card(s, 7.3, y, 5.55, 1.7, fill=WHITE, line=LIGHT_GRAY)
        add_accent_strip(s, 7.3, y, 5.55, accent, thickness=0.05)
        tb = add_textbox(s, 7.45, y + 0.13, 5.3, 1.55)
        add_para(tb.text_frame, h, size=13, bold=True, color=accent)
        add_para(tb.text_frame, body, size=11.5, color=MUTED_INK,
                  space_before=4)

    set_notes(s, (
        "Why DT-CWT? Three properties matter for diffusion-based VSR.\n\n"
        "First, near shift-invariance. Magnitudes of DT-CWT coefficients are "
        "approximately invariant under sub-pixel translations — unlike "
        "standard DWT, whose coefficients oscillate under critical "
        "decimation. The figure on the left empirically shows DT-CWT "
        "magnitudes are about 5 times more stable than DWT under "
        "sub-pixel shifts.\n\n"
        "Second, six directional subbands per scale instead of three — this "
        "preserves edge orientation that DWT would collapse onto its three "
        "axes.\n\n"
        "Third, complex coefficients give us an explicit magnitude-phase "
        "factorization. Magnitude is shift-stable energy — great for "
        "textures. Phase is localized — great for structure. We exploit "
        "this asymmetry in the SubbandBlock."
    ))


def slide_background_sft(prs):
    s, accent = add_content_slide(prs, section=2)
    set_section_title(s, 2, "Spatial Feature Transform (SFT)")

    # Equation block
    add_card(s, 0.55, 1.0, 12.3, 1.3, fill=SOFT_BG, line=accent, line_width=1.0)
    add_accent_strip(s, 0.55, 1.0, 12.3, accent, thickness=0.06)
    tb = add_textbox(s, 0.55, 1.18, 12.3, 1.05, anchor=MSO_ANCHOR.MIDDLE)
    add_para(tb.text_frame, "Spatially-varying affine modulation",
              size=12, bold=True, color=accent, align=PP_ALIGN.CENTER)
    add_para(tb.text_frame, "X′  =  X ⊙ γ  +  β",
              size=26, bold=True, color=INK,
              align=PP_ALIGN.CENTER, space_before=2)

    # Comparison cards
    cards = [
        ("AdaIN", "Globally uniform modulation. Loses spatial detail."),
        ("Cross-attention", "Spatially varying — but O(N²) cost in feature size."),
        ("SFT (ours)", "Spatially varying + linear-time. Ideal for wavelet conditioning."),
    ]
    for i, (h, body) in enumerate(cards):
        x = 0.55 + i * 4.16
        add_card(s, x, 2.6, 4.0, 1.65, fill=WHITE, line=LIGHT_GRAY)
        add_accent_strip(s, x, 2.6, 4.0, accent, thickness=0.05)
        col = accent if i == 2 else GRAY
        tb = add_textbox(s, x + 0.13, 2.74, 3.75, 1.45)
        add_para(tb.text_frame, h, size=14, bold=True, color=col)
        add_para(tb.text_frame, body, size=11, color=MUTED_INK,
                  space_before=4)

    # BD-SFT preview
    add_card(s, 0.55, 4.55, 12.3, 2.4, fill=SOFT_BG, line=DEEP, line_width=1.0)
    add_accent_strip(s, 0.55, 4.55, 12.3, DEEP, thickness=0.06)
    tb = add_textbox(s, 0.55, 4.75, 12.3, 2.2, anchor=MSO_ANCHOR.MIDDLE)
    add_para(tb.text_frame, "Band-Decoupled SFT — preview",
              size=13, bold=True, color=DEEP, align=PP_ALIGN.CENTER)

    # Two side-by-side mini-blocks
    add_card(s, 1.5, 5.25, 4.8, 1.5, fill=WHITE, line=HIGH_BAND, line_width=1.0)
    tb1 = add_textbox(s, 1.65, 5.4, 4.5, 1.3)
    add_para(tb1.text_frame, "(γ_ℋ, β_ℋ)  →  up_blocks[1]",
              size=13, bold=True, color=HIGH_BAND, align=PP_ALIGN.CENTER)
    add_para(tb1.text_frame, "HIGH bands modulate decoder",
              size=11, color=MUTED_INK, align=PP_ALIGN.CENTER, space_before=4)
    add_para(tb1.text_frame, "→ textural detail synthesis",
              size=10.5, italic=True, color=GRAY, align=PP_ALIGN.CENTER)

    add_card(s, 7.05, 5.25, 4.8, 1.5, fill=WHITE, line=LOW_BAND, line_width=1.0)
    tb2 = add_textbox(s, 7.2, 5.4, 4.5, 1.3)
    add_para(tb2.text_frame, "(γ_ℒ, β_ℒ)  →  down_blocks[1]",
              size=13, bold=True, color=LOW_BAND, align=PP_ALIGN.CENTER)
    add_para(tb2.text_frame, "LOW bands modulate encoder",
              size=11, color=MUTED_INK, align=PP_ALIGN.CENTER, space_before=4)
    add_para(tb2.text_frame, "→ structural consolidation",
              size=10.5, italic=True, color=GRAY, align=PP_ALIGN.CENTER)

    set_notes(s, (
        "SFT — Spatial Feature Transform — is the modulation primitive we "
        "use. It performs spatially varying affine modulation of "
        "intermediate features: X' = X ⊙ γ + β.\n\n"
        "It was originally proposed for image super-resolution. The reason "
        "I chose it over AdaIN or cross-attention: AdaIN is globally "
        "uniform and would lose spatial detail; cross-attention is "
        "spatially varying but quadratic in spatial size; SFT is both "
        "spatially varying and linear-time, which is a perfect fit for "
        "multi-scale wavelet conditioning that is itself "
        "spatially-localized.\n\n"
        "Preview of the band-decoupled scheme — HIGH-band SFT is injected "
        "at the decoder (up_blocks[1]) for texture synthesis; LOW-band SFT "
        "is injected at the encoder (down_blocks[1]) for structural "
        "consolidation. I'll detail the reasoning in the Method section."
    ))


def slide_band_partition(prs):
    s, accent = add_content_slide(prs, section=3)
    set_section_title(s, 3, "Multi-Scale Decomposition",
                       subtitle="DT-CWT → band partitioning")

    # Equation header
    add_card(s, 0.55, 1.0, 12.3, 0.85, fill=SOFT_BG, line=accent, line_width=1.0)
    tb = add_textbox(s, 0.55, 1.1, 12.3, 0.7, anchor=MSO_ANCHOR.MIDDLE)
    add_para(tb.text_frame,
              "DT-CWT_{J=4}(I^{LR}_t) =  ( I^{LP}_t ,  {C^{(j,d)}_t}_{j=1..4, d=1..6} )      "
              "with C ∈ ℂ^{H/2ʲ × W/2ʲ × 3}",
              size=14, bold=True, color=INK, align=PP_ALIGN.CENTER)

    # Two band cards
    add_card(s, 0.55, 2.1, 6.0, 2.4, fill=WHITE, line=HIGH_BAND, line_width=1.2)
    add_accent_strip(s, 0.55, 2.1, 6.0, HIGH_BAND, thickness=0.06)
    tb = add_textbox(s, 0.7, 2.28, 5.7, 2.2)
    add_para(tb.text_frame, "HIGH band  ℋ = {j=1, j=2}",
              size=14, bold=True, color=HIGH_BAND)
    add_para(tb.text_frame, "[1/8, 1/2] cyc/px  ·  fine texture & sharp edges",
              size=11, color=MUTED_INK, space_before=4)
    add_para(tb.text_frame, "→  injected at up_blocks[1]  (decoder)",
              size=12, bold=True, color=INK, space_before=8)

    add_card(s, 6.85, 2.1, 6.0, 2.4, fill=WHITE, line=LOW_BAND, line_width=1.2)
    add_accent_strip(s, 6.85, 2.1, 6.0, LOW_BAND, thickness=0.06)
    tb = add_textbox(s, 7.0, 2.28, 5.7, 2.2)
    add_para(tb.text_frame, "LOW band  ℒ = {j=3, j=4}",
              size=14, bold=True, color=LOW_BAND)
    add_para(tb.text_frame, "[0, 1/8] cyc/px  ·  coarse structure",
              size=11, color=MUTED_INK, space_before=4)
    add_para(tb.text_frame, "→  injected at down_blocks[1]  (encoder)",
              size=12, bold=True, color=INK, space_before=8)

    # Bottom — rationale
    add_card(s, 0.55, 4.7, 12.3, 2.25, fill=SOFT_BG, line=accent, line_width=1.0)
    add_accent_strip(s, 0.55, 4.7, 12.3, accent, thickness=0.06)
    tb = add_textbox(s, 0.85, 4.9, 11.7, 2.05)
    add_para(tb.text_frame, "Why this 2:2 split?",
              size=13, bold=True, color=accent)
    add_para(tb.text_frame,
              "▸  Spectral — clean split at the LR spectrum midpoint; "
              "each pathway specializes on a non-overlapping range.",
              size=12, color=MUTED_INK, space_before=4)
    add_para(tb.text_frame,
              "▸  Architectural — 2 scales × 256 ch = 512 ch  matches the "
              "U-Net's block_out_channels[1] = 512.",
              size=12, color=MUTED_INK, space_before=2)
    add_para(tb.text_frame,
              "▸  J = 4 — j=4 spatial size H/16 × W/16 fits cleanly atop "
              "SD-VAE's 8× and U-Net's 2× downsampling.",
              size=12, color=MUTED_INK, space_before=2)

    set_notes(s, (
        "I decompose each LR frame with a 4-level DT-CWT. Each scale yields "
        "six complex-valued directional subbands.\n\n"
        "The four scales are partitioned at the boundary between j=2 and "
        "j=3. Scales 1 and 2 — covering normalized frequencies from 1/8 to "
        "1/2 cycles per pixel — form the HIGH band; they capture fine "
        "texture and sharp edges. Scales 3 and 4 — from 0 to 1/8 — form the "
        "LOW band; they capture coarse structure.\n\n"
        "Three reasons for this 2:2 split. Spectral — it cleanly splits the "
        "LR signal in half. Architectural — two scales × 256 channels "
        "concatenates to 512 channels, exactly matching the U-Net's "
        "block_out_channels[1] at down_blocks[1] and up_blocks[1]. And "
        "J=4 is the largest number of levels whose smallest subband still "
        "fits cleanly into the U-Net feature resolution given SD-VAE's "
        "8× and U-Net's 2× downsampling."
    ))


def slide_subbandblock_detail(prs):
    s, accent = add_content_slide(prs, section=3)
    set_section_title(s, 3, "SubbandBlock",
                       subtitle="Magnitude-phase decoupling")

    # Left — diagram-ish vertical pipeline
    add_card(s, 0.55, 1.0, 5.8, 5.9, fill=WHITE, line=LIGHT_GRAY)
    add_accent_strip(s, 0.55, 1.0, 5.8, accent, thickness=0.06)
    tb = add_textbox(s, 0.7, 1.18, 5.5, 5.6)
    add_para(tb.text_frame, "Operator pipeline", size=14, bold=True, color=accent)
    add_para(tb.text_frame, "Magnitude branch  ℰ_M",
              size=12, bold=True, color=INK, space_before=8)
    add_para(tb.text_frame,
              "DWConv₃ₓ₃ → SiLU → SA → Conv₁ₓ₁     (18 → 64 ch)",
              size=11, color=MUTED_INK)
    add_para(tb.text_frame, "Phase encoding branch  ℰ_φ",
              size=12, bold=True, color=INK, space_before=8)
    add_para(tb.text_frame, "𝒯(φ) = (sin φ, cos φ)   ⇒   wraparound-safe",
              size=11, color=MUTED_INK)
    add_para(tb.text_frame,
              "DWConv₃ₓ₃ → SiLU → SA → Conv₁ₓ₁     (36 → 128 ch)",
              size=11, color=MUTED_INK)
    add_para(tb.text_frame, "Recombination",
              size=12, bold=True, color=INK, space_before=8)
    add_para(tb.text_frame,
              "e_re = e_M ⊙ e_cos,   e_im = e_M ⊙ e_sin",
              size=11, color=MUTED_INK)
    add_para(tb.text_frame, "[e_M ‖ e_re ‖ e_im]   →   3 × 64 = 192 ch",
              size=11, color=MUTED_INK)
    add_para(tb.text_frame, "SFT heads", size=12, bold=True, color=INK,
              space_before=8)
    add_para(tb.text_frame, "Conv₃ → SiLU → Conv₃   →   (γ⁽ʲ⁾, β⁽ʲ⁾) @ 256 ch each",
              size=11, color=MUTED_INK)

    # Right top — key insight callout
    add_card(s, 6.55, 1.0, 6.3, 2.7, fill=SOFT_BG, line=accent, line_width=1.0)
    add_accent_strip(s, 6.55, 1.0, 6.3, accent, thickness=0.06)
    tb = add_textbox(s, 6.75, 1.18, 5.95, 2.5)
    add_para(tb.text_frame, "Why decouple magnitude & phase?",
              size=13, bold=True, color=accent)
    add_para(tb.text_frame,
              "▸  Magnitude — non-negative, shift-stable, energy descriptor.",
              size=11.5, color=MUTED_INK, space_before=4)
    add_para(tb.text_frame,
              "▸  Phase — 2π-periodic, localized, discontinuous at ±π.",
              size=11.5, color=MUTED_INK, space_before=2)
    add_para(tb.text_frame,
              "Mixing them in one branch dilutes their roles. Decoupling "
              "lets each branch learn its native statistics.",
              size=11.5, color=MUTED_INK, space_before=4)

    # Right bottom — parameter callout
    add_card(s, 6.55, 3.85, 6.3, 3.05, fill=WHITE, line=DEEP, line_width=1.0)
    add_accent_strip(s, 6.55, 3.85, 6.3, DEEP, thickness=0.06)
    tb = add_textbox(s, 6.7, 4.05, 6.0, 2.9, anchor=MSO_ANCHOR.MIDDLE)
    add_para(tb.text_frame, "Frequency Encoder", size=13, bold=True, color=DEEP,
              align=PP_ALIGN.CENTER)
    add_para(tb.text_frame, "6.22 M  params", size=30, bold=True, color=INK,
              align=PP_ALIGN.CENTER, space_before=4)
    add_para(tb.text_frame, "≈ 3 % of the 208 M ControlNet",
              size=12, italic=True, color=GRAY, align=PP_ALIGN.CENTER,
              space_before=2)
    add_para(tb.text_frame,
              "Identity-preserving init  —  γ ≡ 1, β ≡ 0 at step 0",
              size=11.5, color=MUTED_INK, align=PP_ALIGN.CENTER,
              space_before=8)

    set_notes(s, (
        "The SubbandBlock processes one DT-CWT scale.\n\n"
        "Two parallel branches. The magnitude branch takes 18 input channels "
        "(6 directions × 3 colors). The phase branch first encodes phase "
        "as a (sin φ, cos φ) pair — that's 36 channels — and that "
        "trigonometric encoding makes the representation continuous across "
        "the ±π wraparound.\n\n"
        "Why decouple? Magnitude is non-negative shift-stable energy. Phase "
        "is 2π-periodic and discontinuous. Mixing them at the first conv "
        "would force the network to learn both statistics simultaneously, "
        "diluting each. Decoupling lets each branch specialize.\n\n"
        "After recombination — multiplying magnitude with cos and sin to "
        "reconstruct complex-valued embeddings — we concatenate "
        "[magnitude ‖ real ‖ imag] for 192 channels, then two SFT heads "
        "output γ and β at 256 channels each.\n\n"
        "The whole Frequency Encoder is 6.22 million parameters — about 3 "
        "percent of the ControlNet backbone. It's initialised so the "
        "U-Net behaves identically to the original at step zero, then "
        "gradually learns frequency modulation."
    ))


def slide_bdsft_injection(prs):
    s, accent = add_content_slide(prs, section=3)
    set_section_title(s, 3, "BD-SFT Injection",
                       subtitle="Asymmetric encoder-decoder modulation")

    # Equation strip
    add_card(s, 0.55, 1.0, 12.3, 0.85, fill=SOFT_BG, line=accent, line_width=1.0)
    tb = add_textbox(s, 0.55, 1.1, 12.3, 0.7, anchor=MSO_ANCHOR.MIDDLE)
    add_para(tb.text_frame,
              "γ_ℋ = [γ⁽¹⁾ ‖ γ⁽²⁾]  → up_blocks[1]      "
              "γ_ℒ = [γ⁽³⁾ ‖ γ⁽⁴⁾]  → down_blocks[1]      (each: 512 ch)",
              size=14, bold=True, color=INK, align=PP_ALIGN.CENTER)

    # Three rationale cards
    cards = [
        ("Architectural alignment",
         ["block_out_channels = [256, 512, 512, 1024].",
          "down_blocks[1] / up_blocks[1] both = 512 channels.",
          "Matches (γ, β) ∈ ℝ^{512 × h × w} by construction."]),
        ("Functional separation",
         ["U-Net encoder suppresses HF; consolidates structure (FreeU, 2024).",
          "Decoder reintroduces HF via upsampling + skip fusion.",
          "Align LOW ↔ encoder, HIGH ↔ decoder."]),
        ("Conservative design",
         ["One injection per band — minimal parameter overhead.",
          "Clean identity initialisation at both injection points.",
          "γ, β are bilinear-aligned to (h, w) when needed."]),
    ]
    for i, (h, lines) in enumerate(cards):
        x = 0.55 + i * 4.16
        add_card(s, x, 2.1, 4.0, 4.85, fill=WHITE, line=LIGHT_GRAY)
        add_accent_strip(s, x, 2.1, 4.0, accent, thickness=0.06)
        # Big number label
        nbox = slide_number_box(s, x + 0.18, 2.28, 0.55, 0.55,
                                 number=i + 1, color=accent)
        tb = add_textbox(s, x + 0.85, 2.28, 3.05, 0.6)
        add_para(tb.text_frame, h, size=13, bold=True, color=accent)
        body = add_textbox(s, x + 0.18, 2.95, 3.72, 3.9)
        for line in lines:
            add_para(body.text_frame, "▸  " + line, size=11.5,
                      color=MUTED_INK, space_before=4, space_after=2)

    set_notes(s, (
        "The injection scheme is asymmetric. HIGH-band modulation tensors — "
        "obtained by concatenating γ from scales 1 and 2 — go into "
        "up_blocks[1], the decoder. LOW-band tensors go into "
        "down_blocks[1], the encoder. Each has exactly 512 channels.\n\n"
        "Three reasons. First, architectural alignment — both U-Net blocks "
        "have 512 channels, so the modulation tensors match without "
        "extra projection.\n\n"
        "Second, functional separation. The FreeU analysis showed the "
        "encoder progressively suppresses high frequencies while "
        "consolidating coarse structure, and the decoder reintroduces "
        "high-frequency detail via upsampling and skip-fusion. So we "
        "align LOW with encoder and HIGH with decoder — each band "
        "augments the stage where it has the most direct impact.\n\n"
        "Third, conservative design — one injection per band keeps "
        "parameter overhead low and preserves the clean identity "
        "initialization at both points."
    ))


def slide_number_box(slide, left, top, w, h, *, number, color):
    """Small numbered circle for accent."""
    badge = slide.shapes.add_shape(
        MSO_SHAPE.OVAL,
        Inches(left), Inches(top), Inches(w), Inches(h),
    )
    badge.fill.solid()
    badge.fill.fore_color.rgb = color
    badge.line.fill.background()
    badge.shadow.inherit = False
    tf = badge.text_frame
    tf.margin_left = Inches(0)
    tf.margin_right = Inches(0)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = str(number)
    r.font.size = Pt(15)
    r.font.bold = True
    r.font.color.rgb = WHITE
    return badge


def slide_training_loss(prs):
    s, accent = add_content_slide(prs, section=3)
    set_section_title(s, 3, "Training Loss",
                       subtitle="Band-decoupled wavelet supervision")

    # Total loss
    add_card(s, 0.55, 1.0, 12.3, 1.05, fill=SOFT_BG, line=accent, line_width=1.0)
    tb = add_textbox(s, 0.55, 1.13, 12.3, 0.85, anchor=MSO_ANCHOR.MIDDLE)
    add_para(tb.text_frame, "Total objective",
              size=11, bold=True, color=accent, align=PP_ALIGN.CENTER)
    add_para(tb.text_frame,
              "ℒ_total  =  ℒ_MSE  +  λ_wav · 𝟙[t_step mod K = 0] · ℒ_wav      "
              "(λ_wav = 1.0,  K = 4)",
              size=16, bold=True, color=INK, align=PP_ALIGN.CENTER,
              space_before=2)

    # Wavelet loss card
    add_card(s, 0.55, 2.25, 12.3, 2.0, fill=WHITE, line=DEEP, line_width=1.0)
    add_accent_strip(s, 0.55, 2.25, 12.3, DEEP, thickness=0.06)
    tb = add_textbox(s, 0.55, 2.45, 12.3, 1.8, anchor=MSO_ANCHOR.MIDDLE)
    add_para(tb.text_frame, "Band-decoupled wavelet loss",
              size=12, bold=True, color=DEEP, align=PP_ALIGN.CENTER)
    add_para(tb.text_frame,
              "ℒ_wav  =  λ_ℋ Σ_{j∈ℋ} ‖|Ĉ⁽ʲ⁾| − |C⁽ʲ⁾|‖₁   +   "
              "λ_ℒ Σ_{j∈ℒ} ℒⱼ^{mp}   +   λ_LP ‖Î^{LP} − I^{LP}‖₁",
              size=13.5, bold=True, color=INK, align=PP_ALIGN.CENTER,
              space_before=4)
    add_para(tb.text_frame,
              "ℒⱼ^{mp}  =  ‖|Ĉ⁽ʲ⁾| − |C⁽ʲ⁾|‖₁  +  𝔼[ |C⁽ʲ⁾| · (1 − cos(φ̂ − φ)) ]",
              size=12, italic=True, color=MUTED_INK, align=PP_ALIGN.CENTER,
              space_before=2)

    # Two pillars
    add_card(s, 0.55, 4.45, 6.0, 2.5, fill=WHITE, line=HIGH_BAND, line_width=1.0)
    add_accent_strip(s, 0.55, 4.45, 6.0, HIGH_BAND, thickness=0.06)
    tb = add_textbox(s, 0.75, 4.63, 5.6, 2.3)
    add_para(tb.text_frame, "HIGH band — magnitude only",
              size=13, bold=True, color=HIGH_BAND)
    add_para(tb.text_frame,
              "Texture is supervised loosely so the network can synthesize "
              "freely without phase-matching constraints.",
              size=11.5, color=MUTED_INK, space_before=4)
    add_para(tb.text_frame, "λ_ℋ = 0.1  ·  soft texture prior",
              size=12, bold=True, color=INK, space_before=8)

    add_card(s, 6.85, 4.45, 6.0, 2.5, fill=WHITE, line=LOW_BAND, line_width=1.0)
    add_accent_strip(s, 6.85, 4.45, 6.0, LOW_BAND, thickness=0.06)
    tb = add_textbox(s, 7.05, 4.63, 5.6, 2.3)
    add_para(tb.text_frame, "LOW band — mag-weighted angular distance",
              size=13, bold=True, color=LOW_BAND)
    add_para(tb.text_frame,
              "Structural phase is penalised in proportion to local energy. "
              "Wraparound-safe by construction.",
              size=11.5, color=MUTED_INK, space_before=4)
    add_para(tb.text_frame, "λ_ℒ = λ_LP = 1.0  ·  strict structural match",
              size=12, bold=True, color=INK, space_before=8)

    set_notes(s, (
        "The total objective combines two terms. The standard latent ε-MSE, "
        "and a band-decoupled wavelet loss applied every K=4 steps in "
        "pixel space. The K=4 schedule keeps VAE-decoding overhead "
        "manageable.\n\n"
        "The wavelet loss decomposes by band. HIGH bands are supervised in "
        "magnitude only — we don't pin down phase, so the network can "
        "synthesize textures freely. LOW bands include a "
        "magnitude-weighted angular distance term — 1 minus cosine of the "
        "phase difference — weighted by the ground-truth magnitude so "
        "phase fidelity matters most where structural energy is high. "
        "This is wraparound-safe.\n\n"
        "Weights: λ_H = 0.1 — a soft texture prior. λ_L and λ_LP = 1.0 — "
        "strict structural matching. We empirically found that λ_H = 1.0 "
        "produces visible HF artifacts, while λ_H near zero degrades "
        "perceptual quality. 0.1 was the stable choice."
    ))


def slide_experiments_setup(prs):
    s, accent = add_content_slide(prs, section=4)
    set_section_title(s, 4, "Datasets & Training Setup")

    # Datasets table
    rows = [
        ("REDS (train split)", "Training",            "236",       "720p"),
        ("REDS4",              "In-domain eval",      "4",         "720p"),
        ("Vid4",               "OOD eval",            "4",         "SD"),
        ("UDM10",              "OOD eval",            "10",        "720p"),
        ("SPMCS",              "OOD eval",            "30",        "SD"),
    ]
    add_table(s, 0.55, 1.0, 7.0, 2.2,
              headers=["Dataset", "Role", "Seqs", "Resolution"],
              rows=rows, header_size=11, body_size=11)

    # Training config table
    rows2 = [
        ("Backbone (frozen)", "SD v2.1 U-Net 472 M  +  VAE 55 M"),
        ("Trainable",         "ControlNet 208 M  +  Freq. Encoder 6.22 M"),
        ("Optimizer",         "AdamW · lr = 1e-4 · 20 000 iters"),
        ("Batch / window",    "B = 8 · T_win = 3 · 64²→256² patches"),
        ("DT-CWT",            "biort=near_sym_a · qshift=qshift_a · J = 4"),
        ("Inference",         "50 DDPM · Bidirectional Sampling"),
    ]
    add_table(s, 7.75, 1.0, 5.1, 3.6,
              headers=["Item", "Specification"],
              rows=rows2, header_size=10, body_size=10)

    # Key takeaway
    add_card(s, 0.55, 3.55, 7.0, 3.4, fill=SOFT_BG, line=accent, line_width=1.0)
    add_accent_strip(s, 0.55, 3.55, 7.0, accent, thickness=0.06)
    tb = add_textbox(s, 0.75, 3.75, 6.6, 3.2)
    add_para(tb.text_frame, "Controlled comparison",
              size=14, bold=True, color=accent)
    add_para(tb.text_frame,
              "Training data, backbone, and temporal scaffold are identical "
              "to StableVSR.",
              size=12.5, color=MUTED_INK, space_before=6)
    add_para(tb.text_frame,
              "The only architectural difference is the Wavelet Conditioning "
              "Module (WCM) with BD-SFT injection.",
              size=12.5, color=MUTED_INK, space_before=6)
    add_para(tb.text_frame,
              "⇒  Any REDS4 gap directly attributes to the WCM mechanism.",
              size=12.5, bold=True, color=INK, space_before=8)

    set_notes(s, (
        "Experimental setup. Training uses the REDS dataset minus the four "
        "REDS4 sequences — 236 sequences total — following the standard "
        "convention from BasicVSR and StableVSR. Evaluation uses REDS4 "
        "for in-domain, plus Vid4, UDM10, and SPMCS for "
        "out-of-distribution.\n\n"
        "Training config: AdamW with constant 1e-4 learning rate for 20 000 "
        "iterations. Batch size 8, temporal window of 3 frames, 64×64 LR "
        "patches mapped to 256×256 HR. Two RTX A6000 48GB GPUs.\n\n"
        "The crucial point: all training hyperparameters and the backbone "
        "are identical to StableVSR. The only architectural difference is "
        "the WCM. This isolation-by-design is what lets us attribute the "
        "REDS4 performance gap directly to the WCM mechanism."
    ))


def slide_metrics(prs):
    s, accent = add_content_slide(prs, section=4)
    set_section_title(s, 4, "Evaluation Metrics",
                       subtitle="9 metrics across 4 categories")

    cats = [
        ("Reconstruction fidelity",   "PSNR ↑ · SSIM ↑",
         "Pixel and structural similarity.",
         SEC[5]),
        ("Full-ref perceptual",       "LPIPS ↓ · DISTS ↓",
         "Deep-feature perceptual distance.",
         SEC[2]),
        ("No-ref perceptual",         "MUSIQ ↑ · CLIP-IQA ↑ · NIQE ↓",
         "Reference-free quality (critical for generative VSR).",
         SEC[3]),
        ("Temporal consistency",      "tLPIPS ↓ · tOF ↓",
         "Inter-frame perceptual / motion stability.",
         SEC[5]),
    ]
    for i, (cat, metrics, body, col) in enumerate(cats):
        row = i // 2
        cidx = i % 2
        x = 0.55 + cidx * 6.3
        y = 1.0 + row * 2.0
        add_card(s, x, y, 6.05, 1.8, fill=WHITE, line=LIGHT_GRAY)
        add_accent_strip(s, x, y, 6.05, col, thickness=0.06)
        tb = add_textbox(s, x + 0.18, y + 0.13, 5.7, 1.6)
        add_para(tb.text_frame, cat, size=13, bold=True, color=col)
        add_para(tb.text_frame, metrics, size=14, bold=True, color=INK,
                  space_before=4)
        add_para(tb.text_frame, body, size=11, color=MUTED_INK, space_before=2)

    # Takeaway
    add_card(s, 0.55, 5.05, 12.3, 1.9, fill=SOFT_BG, line=accent, line_width=1.0)
    add_accent_strip(s, 0.55, 5.05, 12.3, accent, thickness=0.06)
    tb = add_textbox(s, 0.85, 5.25, 11.7, 1.65)
    add_para(tb.text_frame, "Why so many metrics?",
              size=13, bold=True, color=accent)
    add_para(tb.text_frame,
              "Generative VSR sits on the perception–distortion trade-off — "
              "no single metric tells the full story. Pixel metrics under-"
              "credit realistic synthesis; NR metrics complement reference-"
              "based ones; temporal metrics expose flicker that per-frame "
              "metrics miss.",
              size=12, color=MUTED_INK, space_before=4)

    set_notes(s, (
        "Nine metrics across four categories, because generative VSR sits "
        "on the perception–distortion trade-off and no single number "
        "tells the full story.\n\n"
        "Reconstruction fidelity — PSNR and SSIM — measures pixel accuracy. "
        "Full-reference perceptual — LPIPS and DISTS — uses deep features "
        "to capture perceptual similarity. No-reference perceptual — "
        "MUSIQ, CLIP-IQA, and NIQE — is critical for generative methods "
        "because it doesn't penalize realistic synthesis that deviates "
        "from pixel-exact ground truth.\n\n"
        "Temporal consistency — tLPIPS and tOF — captures inter-frame "
        "stability. tLPIPS is especially good at exposing the texture "
        "flicker that pixel metrics miss."
    ))


def slide_results_reds4(prs):
    s, accent = add_content_slide(prs, section=5)
    set_section_title(s, 5, "Results — REDS4",
                       subtitle="Direct comparison vs. StableVSR")

    rows = [
        ("StableVSR",
         "24.04", "0.690", "0.309", "0.164", "42.79", "0.237", "4.38",
         "41.11", "13.962"),
        ("WC-BD-SFT (Ours)",
         "24.48", "0.691", "0.193", "0.088", "65.70", "0.386", "2.77",
         "15.25", "11.118"),
        ("Δ (relative)",
         "+1.8%", "+0.1%", "−37.5%", "−46.3%", "+53.5%", "+62.9%",
         "−36.8%", "−62.9%", "−20.4%"),
    ]
    add_table(s, 0.4, 1.0, 12.55, 1.85,
              headers=["Method",
                       "PSNR↑", "SSIM↑", "LPIPS↓", "DISTS↓",
                       "MUSIQ↑", "CLIP-IQA↑", "NIQE↓", "tLPIPS↓", "tOF↓"],
              rows=rows, header_size=9.5, body_size=10,
              highlight_row_idx=1, highlight_fill=RGBColor(0xE3, 0xF0, 0xFF))

    # Three big-number cards
    add_kpi_card(s, 0.55, 3.15, 4.0, 1.85, label="Perceptual (LPIPS ↓)",
                  value="−37.5 %", sub="0.309 → 0.193", accent=SEC[2])
    add_kpi_card(s, 4.7, 3.15, 4.0, 1.85, label="Perceptual (MUSIQ ↑)",
                  value="+53.5 %", sub="42.79 → 65.70", accent=SEC[3])
    add_kpi_card(s, 8.85, 3.15, 4.0, 1.85, label="Temporal (tLPIPS ↓)",
                  value="−62.9 %", sub="41.11 → 15.25", accent=accent)

    # Interpretation strip
    add_card(s, 0.55, 5.2, 12.3, 1.75, fill=SOFT_BG, line=accent, line_width=1.0)
    add_accent_strip(s, 0.55, 5.2, 12.3, accent, thickness=0.06)
    tb = add_textbox(s, 0.85, 5.4, 11.7, 1.5)
    add_para(tb.text_frame, "Interpretation",
              size=13, bold=True, color=accent)
    add_para(tb.text_frame,
              "Same training data, backbone, and temporal scaffold — only "
              "WCM differs. The gap isolates the wavelet-conditioned "
              "BD-SFT contribution.",
              size=12, color=MUTED_INK, space_before=4)

    set_notes(s, (
        "This is the most direct comparison in the thesis. Same training "
        "data, same Stable Diffusion v2.1 backbone, same temporal scaffold "
        "of ControlNet plus RAFT-based optical flow with bidirectional "
        "sampling. The only difference is the WCM module.\n\n"
        "The proposed method improves on every single metric.\n\n"
        "Most importantly — perceptual: LPIPS drops 37.5 percent, DISTS "
        "drops 46.3 percent, MUSIQ rises 53.5 percent. Temporal "
        "consistency: tLPIPS drops 62.9 percent — from 41 to 15. Pixel "
        "fidelity edges up marginally too.\n\n"
        "Because every other variable is matched, this gap attributes "
        "directly to the wavelet-conditioned BD-SFT mechanism."
    ))


def slide_results_dm_group(prs):
    s, accent = add_content_slide(prs, section=5)
    set_section_title(s, 5, "Results — Within the DM Group",
                       subtitle="vs. DGAF-VSR & cross-paradigm reference")

    rows = [
        ("non-DM", "BasicVSR++",
         "24.88", "0.730", "0.364", "0.179", "41.23", "0.265", "5.90",
         "38.26", "13.739", "—"),
        ("DM", "StableVSR",
         "24.04", "0.690", "0.309", "0.164", "42.79", "0.237", "4.38",
         "41.11", "13.962", "3.00"),
        ("DM", "DGAF-VSR",
         "24.07", "0.694", "0.307", "0.161", "43.41", "0.242", "4.37",
         "40.12", "13.732", "1.89"),
        ("DM", "WC-BD-SFT (Ours)",
         "24.48", "0.691", "0.193", "0.088", "65.70", "0.386", "2.77",
         "15.25", "11.118", "1.11"),
    ]
    add_table(s, 0.3, 1.0, 12.75, 2.5,
              headers=["Paradigm", "Method",
                       "PSNR↑", "SSIM↑", "LPIPS↓", "DISTS↓",
                       "MUSIQ↑", "CLIP-IQA↑", "NIQE↓",
                       "tLPIPS↓", "tOF↓", "Mean rank"],
              rows=rows, header_size=9, body_size=9,
              highlight_row_idx=3, highlight_fill=RGBColor(0xE3, 0xF0, 0xFF))

    # Two takeaway cards
    add_card(s, 0.55, 3.85, 6.0, 3.1, fill=WHITE, line=SEC[3], line_width=1.0)
    add_accent_strip(s, 0.55, 3.85, 6.0, SEC[3], thickness=0.06)
    tb = add_textbox(s, 0.75, 4.05, 5.6, 2.9)
    add_para(tb.text_frame, "vs. DGAF-VSR (strongest DM baseline)",
              size=13, bold=True, color=SEC[3])
    for line, gain in [
        ("LPIPS  0.307 → 0.193", "−37.1 %"),
        ("MUSIQ  43.41 → 65.70", "+51.4 %"),
        ("tLPIPS 40.12 → 15.25", "−62.0 %"),
    ]:
        p = tb.text_frame.add_paragraph()
        p.space_before = Pt(4)
        r1 = p.add_run()
        r1.text = "▸  " + line + "    "
        r1.font.size = Pt(12)
        r1.font.color.rgb = MUTED_INK
        r2 = p.add_run()
        r2.text = gain
        r2.font.size = Pt(12)
        r2.font.bold = True
        r2.font.color.rgb = SEC[3]
    add_para(tb.text_frame, "Mean rank within DM:  1.11  (vs. 1.89 / 3.00)",
              size=12, bold=True, color=INK, space_before=8)

    add_card(s, 6.85, 3.85, 6.0, 3.1, fill=WHITE, line=DEEP, line_width=1.0)
    add_accent_strip(s, 6.85, 3.85, 6.0, DEEP, thickness=0.06)
    tb = add_textbox(s, 7.05, 4.05, 5.6, 2.9)
    add_para(tb.text_frame, "Cross-paradigm reference — BasicVSR++",
              size=13, bold=True, color=DEEP)
    add_para(tb.text_frame,
              "Leads on PSNR / SSIM — regression target.",
              size=12, color=MUTED_INK, space_before=4)
    add_para(tb.text_frame,
              "Loses on all four perceptual metrics — classical "
              "perception–distortion trade-off (Blau & Michaeli, 2018).",
              size=12, color=MUTED_INK, space_before=4)
    add_para(tb.text_frame,
              "Within-DM ranking is the fair comparison axis for the "
              "proposed mechanism.",
              size=12, italic=True, color=GRAY, space_before=6)

    set_notes(s, (
        "Broadening the REDS4 comparison. Within the diffusion-based group, "
        "WC-BD-SFT achieves the best score on eight of nine metrics — only "
        "SSIM is second-place. Mean rank within the DM group is 1.11, "
        "compared to DGAF-VSR's 1.89 and StableVSR's 3.00.\n\n"
        "DGAF-VSR is the strongest existing DM baseline. Versus DGAF-VSR: "
        "LPIPS −37.1%, MUSIQ +51.4%, tLPIPS −62.0%. Both methods adapt a "
        "pre-trained diffusion prior with a small additional module, but "
        "they target different mechanisms — DGAF uses feature-domain "
        "warping; we use frequency-domain conditioning.\n\n"
        "BasicVSR++ I include as a cross-paradigm reference. It leads on "
        "pixel fidelity but loses on every perceptual metric — the "
        "classical perception–distortion trade-off. So within-DM ranking "
        "is the fair axis for comparing my mechanism."
    ))


def slide_results_cross_dataset(prs):
    s, accent = add_content_slide(prs, section=5)
    set_section_title(s, 5, "Results — Out-of-Distribution",
                       subtitle="Vid4 · UDM10 · SPMCS")

    rows = [
        ("Vid4",  "BasicVSR++",  "26.26", "0.828", "0.189", "61.50", "0.341", "5.04", "15.12"),
        ("Vid4",  "StableVSR",   "22.98", "0.674", "0.185", "67.22", "0.454", "3.19", "25.36"),
        ("Vid4",  "DGAF-VSR",    "23.29", "0.690", "0.177", "67.96", "0.470", "3.10", "17.39"),
        ("Vid4",  "Ours",        "22.64", "0.664", "0.194", "66.15", "0.408", "3.32", "28.53"),
        ("UDM10", "BasicVSR++",  "37.48", "0.956", "0.060", "59.36", "0.443", "5.60",  "5.55"),
        ("UDM10", "StableVSR",   "26.71", "0.834", "0.100", "55.69", "0.362", "4.66",  "4.42"),
        ("UDM10", "DGAF-VSR",    "26.71", "0.835", "0.099", "57.15", "0.380", "4.61",  "2.99"),
        ("UDM10", "Ours",        "25.54", "0.811", "0.124", "63.20", "0.447", "4.12", "14.48"),
        ("SPMCS", "BasicVSR++",  "21.94", "0.617", "0.187", "62.48", "0.434", "5.17",  "3.60"),
        ("SPMCS", "StableVSR",   "19.42", "0.478", "0.196", "69.98", "0.582", "3.28", "51.11"),
        ("SPMCS", "DGAF-VSR",    "20.06", "0.511", "0.178", "67.70", "0.533", "3.61", "20.10"),
        ("SPMCS", "Ours",        "19.94", "0.506", "0.183", "67.22", "0.503", "3.70", "31.59"),
    ]
    add_table(s, 0.35, 1.0, 7.4, 5.95,
              headers=["Set", "Method",
                       "PSNR↑", "SSIM↑", "LPIPS↓", "MUSIQ↑",
                       "CLIP-IQA↑", "NIQE↓", "tLPIPS↓"],
              rows=rows, header_size=9, body_size=8.5)

    # Right column — three observation cards
    obs = [
        ("UDM10 — NR perceptual wins",
         "MUSIQ +13.5 %, CLIP-IQA +23.5 %, NIQE −11.6 %.\n"
         "Higher tLPIPS reflects flicker on short static clips — see Discussion.",
         SEC[3]),
        ("SPMCS — temporal gain",
         "tLPIPS 51.11 → 31.59 over StableVSR (−38.2 %).\n"
         "Frequency conditioning helps on longer motion-rich content.",
         SEC[2]),
        ("Vid4 — distribution gap",
         "Compressed SD vs. clean REDS 720p ⇒ smaller gains.\n"
         "DGAF's HR feature alignment is complementary — fusion candidate.",
         DEEP),
    ]
    for i, (h, body, col) in enumerate(obs):
        y = 1.0 + i * 1.95
        add_card(s, 7.95, y, 4.95, 1.85, fill=WHITE, line=LIGHT_GRAY)
        add_accent_strip(s, 7.95, y, 4.95, col, thickness=0.05)
        tb = add_textbox(s, 8.12, y + 0.12, 4.65, 1.7)
        add_para(tb.text_frame, h, size=12, bold=True, color=col)
        add_para(tb.text_frame, body, size=10.5, color=MUTED_INK,
                  space_before=4)

    set_notes(s, (
        "Out-of-distribution evaluation tells a more nuanced story.\n\n"
        "On UDM10, we get the best NR perceptual scores in the DM group — "
        "MUSIQ, CLIP-IQA, NIQE all best — but a higher tLPIPS. UDM10 "
        "sequences are short (32 frames) with limited motion, and "
        "wavelet-driven texture synthesis introduces small per-frame "
        "variations that accumulate as flicker. I'll discuss this in "
        "the next slide.\n\n"
        "On SPMCS — longer sequences with rich motion — tLPIPS improves "
        "substantially over StableVSR, suggesting the frequency "
        "conditioning helps temporal stability when there's enough "
        "motion for bidirectional sampling to absorb stochastic "
        "variation.\n\n"
        "On Vid4 — compressed SD content — the distribution gap from "
        "REDS limits gains. DGAF-VSR does best on Vid4, but its "
        "high-resolution feature warping is orthogonal to my frequency "
        "approach, so combining the two is a natural future direction."
    ))


def slide_ablation(prs):
    s, accent = add_content_slide(prs, section=5)
    set_section_title(s, 5, "Ablation — Frequency-Band Injection",
                       subtitle="Inference-time band disabling on REDS4")

    rows = [
        ("WC-BD-SFT (full)",     "24.48", "0.691", "0.193", "0.088",
         "65.70", "0.386", "2.77", "15.25", "11.118"),
        ("w/o HIGH",             "24.34", "0.689", "0.223", "0.104",
         "64.49", "0.365", "3.00", "22.11", "11.984"),
        ("w/o LOW",              "24.70", "0.701", "0.210", "0.098",
         "63.39", "0.335", "3.00", "16.57", "11.392"),
        ("w/o both ≈ StableVSR", "24.46", "0.695", "0.247", "0.119",
         "59.35", "0.298", "3.29", "25.18", "12.278"),
    ]
    add_table(s, 0.3, 1.0, 12.75, 2.3,
              headers=["Variant", "PSNR↑", "SSIM↑", "LPIPS↓", "DISTS↓",
                       "MUSIQ↑", "CLIP-IQA↑", "NIQE↓",
                       "tLPIPS↓", "tOF↓"],
              rows=rows, header_size=10, body_size=10,
              highlight_row_idx=0,
              highlight_fill=RGBColor(0xE3, 0xF0, 0xFF))

    # Three insight cards
    insights = [
        ("HIGH is critical",
         "tLPIPS  15.25 → 22.11   (+45.0 %)\n"
         "LPIPS   0.193 → 0.223   (+15.5 %)",
         HIGH_BAND),
        ("LOW shapes perception",
         "MUSIQ   65.70 → 63.39\n"
         "CLIP-IQA 0.386 → 0.335\n"
         "(PSNR/SSIM slightly rise — trade-off)",
         LOW_BAND),
        ("Super-additive when both off",
         "LPIPS w/o both = 0.247 — exceeds either single disable.\n"
         "MUSIQ collapses to 59.35.\n"
         "⇒ bands are jointly necessary.",
         DEEP),
    ]
    for i, (h, body, col) in enumerate(insights):
        x = 0.55 + i * 4.16
        add_card(s, x, 3.55, 4.0, 3.4, fill=WHITE, line=col, line_width=1.0)
        add_accent_strip(s, x, 3.55, 4.0, col, thickness=0.06)
        tb = add_textbox(s, x + 0.18, 3.75, 3.65, 3.2)
        add_para(tb.text_frame, h, size=13, bold=True, color=col)
        add_para(tb.text_frame, body, size=11.5, color=MUTED_INK,
                  space_before=6)

    set_notes(s, (
        "Ablation on band injection. Three variants — disabling HIGH, "
        "LOW, or both — at inference time by setting γ=1 and β=0. The "
        "same trained model is used for all variants, so the effects "
        "are not confounded by separate training trajectories.\n\n"
        "Disabling HIGH degrades all perceptual and temporal metrics "
        "substantially — tLPIPS jumps from 15 to 22, LPIPS from 0.193 to "
        "0.223. This confirms HIGH-band guidance at the decoder is "
        "critical for inter-frame texture coherence.\n\n"
        "Disabling LOW shows the opposite pattern — PSNR and SSIM "
        "marginally improve but NR perceptual metrics deteriorate. LOW-"
        "band conditioning shapes the structural representation in a "
        "way that improves perceptual naturalness at the cost of pixel "
        "fidelity — exactly the perception–distortion trade-off in "
        "miniature.\n\n"
        "Critically — when both are off, degradation is super-additive. "
        "LPIPS hits 0.247, worse than either single disable. This means "
        "the two bands carry complementary information that interacts "
        "non-linearly. The asymmetric injection is not just additive — "
        "it's synergistic."
    ))


def slide_discussion(prs):
    s, accent = add_content_slide(prs, section=5)
    set_section_title(s, 5, "Discussion",
                       subtitle="Three trade-offs revealed")

    # Three cards
    cards = [
        ("(1) Perception–distortion",
         "BasicVSR++ → fidelity peak; DM methods → perceptual peak.\n"
         "WC-BD-SFT pushes furthest toward perceptual extreme.",
         SEC[5]),
        ("(2) Specialization–generalization",
         "REDS-only training isolates the WCM contribution.\n"
         "Wavelet priors are most effective on REDS-like statistics; "
         "smaller gains on Vid4.",
         SEC[3]),
        ("(3) Temporal stability vs. motion",
         "UDM10 (32 frames, static) → tLPIPS regression.\n"
         "SPMCS (long motion) → tLPIPS improvement.\n"
         "Hypothesis: bidirectional sampling needs motion to absorb "
         "stochasticity.",
         SEC[2]),
    ]
    for i, (h, body, col) in enumerate(cards):
        y = 1.0 + i * 2.0
        add_card(s, 0.55, y, 12.3, 1.85, fill=WHITE, line=LIGHT_GRAY)
        add_accent_strip(s, 0.55, y, 12.3, col, thickness=0.06)
        nb = slide_number_box(s, 0.75, y + 0.18, 0.5, 0.5,
                               number=i + 1, color=col)
        tb = add_textbox(s, 1.4, y + 0.18, 11.3, 1.6)
        add_para(tb.text_frame, h.split(") ", 1)[1] if ") " in h else h,
                  size=13, bold=True, color=col)
        for line in body.split("\n"):
            add_para(tb.text_frame, line, size=11.5, color=MUTED_INK,
                      space_before=2, space_after=2)

    set_notes(s, (
        "Three trade-offs that the experiments reveal.\n\n"
        "First — perception versus distortion across paradigms. BasicVSR++ "
        "as a regression model maximises pixel fidelity. The DM methods "
        "are biased toward perceptually plausible textures at the cost of "
        "strict pixel accuracy. WC-BD-SFT pushes furthest toward the "
        "perceptual extreme.\n\n"
        "Second — specialization versus generalization. Training is "
        "REDS-only by design — to isolate the WCM contribution under "
        "matched conditions. Frequency priors specialize to REDS's "
        "wavelet statistics, so we see the largest gains on REDS-like "
        "content and smaller gains on Vid4-like content. This is the "
        "specialization–generalization trade-off inherent to "
        "single-distribution training, and it's a separate research "
        "question from the mechanism being proposed.\n\n"
        "Third — temporal stability versus motion. On UDM10's short "
        "32-frame static clips, wavelet-driven texture synthesis "
        "introduces small per-frame variations that accumulate as "
        "flicker. On SPMCS's longer motion-rich content, the opposite "
        "happens — tLPIPS substantially improves. My hypothesis is that "
        "bidirectional sampling needs sufficient motion to absorb "
        "per-frame stochasticity."
    ))


def slide_limitations(prs):
    s, accent = add_content_slide(prs, section=6)
    set_section_title(s, 6, "Limitations")

    items = [
        ("Training-distribution dependence",
         "Best on REDS-like content; smaller gains on compressed SD (Vid4).",
         "Broader training mix or degradation-aware augmentation can "
         "narrow the gap."),
        ("Perception–distortion trade-off",
         "PSNR/SSIM sacrificed for perceptual quality, esp. on UDM10.",
         "Intrinsic to generative VSR — suited to perceptual applications, "
         "not strict pixel accuracy."),
        ("Short-static flicker",
         "tLPIPS higher than DGAF / StableVSR on UDM10 (32-frame clips).",
         "Add lightweight temporal regularization (flow-guided latent "
         "warping) — see Future Work."),
        ("Computational overhead",
         "Four-level DT-CWT adds modest per-frame cost.",
         "Dominated by diffusion denoising; further optimisation possible "
         "via learnable frequency analysis."),
        ("Stochasticity & comparison scope",
         "Single fixed seed (42) for reproducibility; some 2025 methods "
         "(RVRT, MGLD, DiffVSR, …) not benchmarked.",
         "Multi-seed evaluation and broader benchmarking left as future "
         "work."),
    ]
    for i, (h, situation, fix) in enumerate(items):
        col = i % 2
        row = i // 2
        x = 0.55 + col * 6.3
        y = 1.0 + row * 2.0
        if row == 2 and col == 1:
            continue  # only 5 items; skip last grid slot
        add_card(s, x, y, 6.05, 1.85, fill=WHITE, line=LIGHT_GRAY)
        add_accent_strip(s, x, y, 6.05, accent, thickness=0.05)
        tb = add_textbox(s, x + 0.18, y + 0.12, 5.7, 1.65)
        add_para(tb.text_frame, h, size=12.5, bold=True, color=accent)
        add_para(tb.text_frame, situation, size=11, color=MUTED_INK,
                  space_before=3)
        add_para(tb.text_frame, "→ " + fix, size=10.5, italic=True,
                  color=GRAY, space_before=2)

    set_notes(s, (
        "Five limitations I want to explicitly acknowledge.\n\n"
        "One — training-distribution dependence. The wavelet priors are "
        "specialized to REDS-like content. Vid4 with compressed SD "
        "content sees smaller gains.\n\n"
        "Two — the perception–distortion trade-off is real. PSNR is "
        "sacrificed for perceptual quality, especially on UDM10. This is "
        "intrinsic to generative VSR.\n\n"
        "Three — short, quasi-static sequences flicker more than the "
        "baselines, as I showed in the UDM10 results.\n\n"
        "Four — the four-level DT-CWT adds modest computational overhead, "
        "though it's small compared to the diffusion denoising cost.\n\n"
        "Five — single-seed evaluation and limited comparison scope. "
        "Recent 2025 methods like DiffVSR, STAR, SeedVR2 were not "
        "benchmarked due to release-timing and GPU constraints."
    ))


def slide_conclusion(prs):
    s, accent = add_content_slide(prs, section=6)
    set_section_title(s, 6, "Conclusion · Key Findings")

    # Summary card
    add_card(s, 0.55, 1.0, 12.3, 1.8, fill=SOFT_BG, line=accent, line_width=1.0)
    add_accent_strip(s, 0.55, 1.0, 12.3, accent, thickness=0.06)
    tb = add_textbox(s, 0.75, 1.18, 11.9, 1.65)
    add_para(tb.text_frame, "Frequency-domain conditioning is an effective "
              "inductive bias for diffusion-based VSR.",
              size=15, bold=True, color=INK)
    add_para(tb.text_frame,
              "WC-BD-SFT injects DT-CWT priors into a frozen pre-trained "
              "diffusion U-Net via asymmetric encoder-decoder modulation, "
              "with magnitude-phase decoupling and a band-decoupled wavelet "
              "loss — at only 6.22 M added trainable parameters.",
              size=12, color=MUTED_INK, space_before=6)

    # Three KPI cards
    add_kpi_card(s, 0.55, 3.0, 4.0, 3.95,
                  label="REDS4 · tLPIPS",
                  value="−62.9 %",
                  sub="41.11  →  15.25\n\nvs. StableVSR baseline",
                  accent=SEC[2])
    add_kpi_card(s, 4.7, 3.0, 4.0, 3.95,
                  label="REDS4 · HF spectral power",
                  value="0.693",
                  sub="vs. StableVSR's 0.341\n\n+103 % HF retention",
                  accent=SEC[3])
    add_kpi_card(s, 8.85, 3.0, 4.0, 3.95,
                  label="Frequency Encoder",
                  value="6.22 M",
                  sub="≈ 3 % of ControlNet\n\nU-Net + VAE remain frozen",
                  accent=SEC[5])

    set_notes(s, (
        "To summarize. The thesis answers — yes, frequency-domain "
        "conditioning is an effective and parameter-efficient inductive "
        "bias for diffusion-based VSR.\n\n"
        "Three headline numbers tell the story.\n\n"
        "First — temporal consistency. tLPIPS drops 62.9 percent on the "
        "in-domain REDS4 benchmark.\n\n"
        "Second — frequency-domain fidelity. We retain 69.3 percent of "
        "the ground-truth high-frequency power, compared to StableVSR's "
        "34.1 percent — more than double.\n\n"
        "Third — parameter efficiency. The Frequency Encoder is 6.22 M "
        "parameters, about 3 percent of the ControlNet backbone. The "
        "pre-trained U-Net and VAE remain fully frozen — the generative "
        "prior is preserved by construction."
    ))


def slide_future_work(prs):
    s, accent = add_content_slide(prs, section=6)
    set_section_title(s, 6, "Future Work")

    items = [
        (1, "Flow-free pipeline",
         "Replace RAFT-based warping with a frequency-aware temporal "
         "conditioning module — fully remove optical-flow dependency."),
        (2, "Combine with DGAF-VSR",
         "Frequency conditioning is orthogonal to OGWM + FTCM "
         "feature-domain warping — natural fusion for cross-dataset gains."),
        (3, "Broader training mixture",
         "Train on REDS + Vimeo-90K with degradation-aware augmentation "
         "to narrow the OOD gap on Vid4-style content."),
        (4, "Latent-space wavelet analysis",
         "Operate the wavelet loss in latent space; explore adaptive band "
         "partitioning — reduces VAE-decoding overhead."),
        (5, "Short-sequence temporal regularization",
         "Add lightweight modules (flow-guided latent warping / multi-frame "
         "attention) without compromising parameter efficiency."),
    ]
    # 5 cards in 3+2 arrangement
    for i, (n, h, body) in enumerate(items):
        if i < 3:
            x = 0.55 + i * 4.16
            y = 1.0
        else:
            x = 2.65 + (i - 3) * 4.16
            y = 4.0
        w, ht = 4.0, 2.85
        add_card(s, x, y, w, ht, fill=WHITE, line=LIGHT_GRAY)
        add_accent_strip(s, x, y, w, accent, thickness=0.06)
        nb = slide_number_box(s, x + 0.18, y + 0.18, 0.5, 0.5,
                               number=n, color=accent)
        tb = add_textbox(s, x + 0.85, y + 0.18, w - 1.0, 0.55)
        add_para(tb.text_frame, h, size=12.5, bold=True, color=accent)
        body_tb = add_textbox(s, x + 0.18, y + 0.85, w - 0.32, ht - 1.0)
        add_para(body_tb.text_frame, body, size=11, color=MUTED_INK)

    set_notes(s, (
        "Five future directions.\n\n"
        "One — a fully flow-free pipeline. Replace the RAFT-based optical-"
        "flow warping in the ControlNet branch with a frequency-aware "
        "temporal conditioning module, eliminating optical-flow "
        "dependency entirely.\n\n"
        "Two — combine our frequency-domain conditioning with DGAF-VSR's "
        "feature-domain warping. The two are orthogonal — frequency vs. "
        "feature, what-to-inject vs. where-to-inject — so fusion is a "
        "natural candidate for cross-dataset performance.\n\n"
        "Three — broader training data mixture, REDS plus Vimeo-90K, plus "
        "degradation-aware augmentation to narrow the gap on Vid4-style "
        "out-of-distribution content.\n\n"
        "Four — latent-space wavelet analysis to reduce the VAE-decoding "
        "overhead of the wavelet loss.\n\n"
        "Five — lightweight temporal regularization to address the short-"
        "sequence flicker on UDM10 without compromising parameter "
        "efficiency.\n\n"
        "Thank you. I'm happy to take questions."
    ))


# ──────────────────────────────────────────────────────────────────────────
# Speaker notes for the *existing* (template-supplied) content slides
# ──────────────────────────────────────────────────────────────────────────
def notes_existing(prs):
    """Attach speaker notes to the 9 hand-styled template slides too."""
    # After build_v2 the template slides may not all be present in their
    # original positions, so we find them by their distinctive shape pattern.
    # Simpler — apply notes after reordering, by final slide index.
    pass


# ──────────────────────────────────────────────────────────────────────────
# Reorder helper
# ──────────────────────────────────────────────────────────────────────────
def reorder_slides(prs, new_order):
    sldIdLst = prs.slides._sldIdLst
    children = list(sldIdLst)
    assert sorted(new_order) == list(range(len(children)))
    for child in children:
        sldIdLst.remove(child)
    for idx in new_order:
        sldIdLst.append(children[idx])


# Notes for the existing template slides — keyed by *0-indexed* final position.
# These keys correspond to template-supplied slides only; new content slides
# already received their notes via their own builder functions.
#   0  Cover                       (own notes)
#   1  Contents                    (own notes)
#   ...
#   7  Architecture (template)     ← key 7
#   9  SubbandBlock (template)     ← key 9
#  18  Eval overview (template)    ← key 18
#  19  LR subbands (template)      ← key 19
#  20  GT subbands (template)      ← key 20
#  21  Radial spectrum (template)  ← key 21
#  27  Thanks (template)           ← key 27
NOTES_EXISTING_BY_FINAL_IDX = {
    7: (
        "This is the overall architecture of the proposed framework.\n\n"
        "Panel (a) — the full inference pipeline. The LR input goes through "
        "the VAE encoder into latent space, where the U-Net performs "
        "diffusion denoising. The temporal scaffold is inherited from "
        "StableVSR: a ControlNet receives the RAFT-warped x̂_0 prediction "
        "of the previously denoised neighbor frame.\n\n"
        "What's new is the Wavelet Conditioning Module — the WCM — which "
        "consists of DT-CWT decomposition, the Frequency Encoder, and the "
        "BD-SFT injection. HIGH-band modulation (blue) goes into "
        "up_blocks[1] of the U-Net decoder; LOW-band modulation (red) "
        "goes into down_blocks[1] of the encoder.\n\n"
        "Panel (b) — the Frequency Encoder. It contains four "
        "SubbandBlocks, one per DT-CWT scale L=1 through 4. The detailed "
        "internals come in the next slide."
    ),
    9: (
        "Zooming into one SubbandBlock — the PerLevelProcessor.\n\n"
        "The block takes the six high-pass complex coefficients of one "
        "DT-CWT scale. They go through the Magnitude Encoding branch and "
        "the Phase Encoding branch in parallel — each is "
        "DWConv → SiLU → Spatial Attention → Conv-1×1.\n\n"
        "After recombination — where magnitude is multiplied with the "
        "cos and sin halves of the phase encoding — the resulting complex "
        "embeddings are concatenated and passed through two SFT heads "
        "that emit γ and β. Identity-preserving initialization makes the "
        "wrapped U-Net behave identically to the original at step zero."
    ),
    18: (
        "Qualitative results overview. The right side shows the proposed "
        "method's outputs. The left side shows ablation and comparison "
        "results.\n\n"
        "Highlights — sharper textural detail and cleaner edges than the "
        "StableVSR baseline, especially in regions with repeated "
        "patterns and high-contrast structures, where the frequency-"
        "domain conditioning provides explicit guidance that complements "
        "the spatial-domain ControlNet branch."
    ),
    19: (
        "DT-CWT subband visualization for an LR input.\n\n"
        "Each scale produces six directional subbands oriented at "
        "approximately ±15°, ±45°, and ±75°. The visualisation makes the "
        "directional selectivity of DT-CWT explicit — you can see that "
        "each subband responds to edges of a particular orientation. "
        "This is the directional richness that drives the high-frequency "
        "detail synthesis at the decoder."
    ),
    20: (
        "Same DT-CWT subband visualization, this time for the ground-"
        "truth HR frame. The key observation: HR subbands have richer "
        "high-frequency content and clearer directional structure than "
        "LR. The supervision signal in the wavelet loss aligns the "
        "predicted subbands with these GT subbands."
    ),
    21: (
        "Frequency-domain analysis on REDS4. We compute the radial power "
        "spectrum ratio of each method's output relative to the ground "
        "truth.\n\n"
        "Three observations. First — at low frequencies, all methods "
        "match the GT closely. Second — at mid frequencies, our method "
        "maintains ratios close to 1, while BasicVSR++ and StableVSR "
        "fall to around 0.64. Third — at high frequencies, the gap is "
        "largest. We retain 0.69 of the GT high-frequency power versus "
        "StableVSR's 0.34 — more than doubling.\n\n"
        "This directly validates the core hypothesis: routing HIGH-band "
        "DT-CWT coefficients to the U-Net decoder and LOW-band coefficients "
        "to the encoder enables the framework to preserve spectral "
        "content that pixel-level losses fail to enforce."
    ),
    27: (
        "Thank you for your attention. I'm happy to take questions."
    ),
}


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main():
    prs = Presentation(str(TEMPLATE))

    # 1) Cover & contents
    update_cover(prs)
    update_contents(prs)

    # 2) Add new content slides
    slide_motivation(prs)              # 9
    slide_related_work(prs)            # 10
    slide_contributions(prs)           # 11
    slide_background_dtcwt(prs)        # 12
    slide_background_sft(prs)          # 13
    slide_band_partition(prs)          # 14
    slide_subbandblock_detail(prs)     # 15
    slide_bdsft_injection(prs)         # 16
    slide_training_loss(prs)           # 17
    slide_experiments_setup(prs)       # 18
    slide_metrics(prs)                 # 19
    slide_results_reds4(prs)           # 20
    slide_results_dm_group(prs)        # 21
    slide_results_cross_dataset(prs)   # 22
    slide_ablation(prs)                # 23
    slide_discussion(prs)              # 24
    slide_limitations(prs)             # 25
    slide_conclusion(prs)              # 26
    slide_future_work(prs)             # 27

    # 3) Reorder — interleave existing template slides at the right slots.
    # Existing template slide indices: 0 Cover, 1 Contents, 2 Architecture,
    # 3 SubbandBlock, 4 Eval overview, 5 LR subbands, 6 GT subbands,
    # 7 Eval (radial spectrum), 8 Thanks.
    final_order = [
        0,   # 1. Cover
        1,   # 2. Contents
        9,   # 3. Motivation
        10,  # 4. Related Work
        11,  # 5. Contributions
        12,  # 6. Background · DT-CWT
        13,  # 7. Background · SFT
        2,   # 8. Architecture (template)
        14,  # 9. Multi-Scale / band partition
        3,   # 10. SubbandBlock (template)
        15,  # 11. SubbandBlock detail
        16,  # 12. BD-SFT Injection
        17,  # 13. Training Loss
        18,  # 14. Datasets & Setup
        19,  # 15. Metrics
        20,  # 16. REDS4 vs StableVSR
        21,  # 17. REDS4 DM group
        22,  # 18. Cross-dataset
        4,   # 19. Evaluation overview (template)
        5,   # 20. LR subbands (template)
        6,   # 21. GT subbands (template)
        7,   # 22. Radial spectrum (template)
        23,  # 23. Ablation
        24,  # 24. Discussion
        25,  # 25. Limitations
        26,  # 26. Conclusion
        27,  # 27. Future Work
        8,   # 28. Thanks
    ]
    reorder_slides(prs, final_order)

    # 4) Attach speaker notes for the original template slides too.
    for final_idx, note in NOTES_EXISTING_BY_FINAL_IDX.items():
        if final_idx < len(prs.slides):
            set_notes(prs.slides[final_idx], note)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"Saved: {OUT}")
    print(f"Slides: {len(prs.slides)}")


if __name__ == "__main__":
    main()
