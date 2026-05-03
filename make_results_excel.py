"""Generate an Excel file summarizing dual-SFT evaluation results across datasets.

Usage:
    python make_results_excel.py
    # writes results_dualsft.xlsx in cwd

Requires: openpyxl  (pip install openpyxl)
"""

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

results = {
    "REDS4":  {"PSNR": 24.48, "SSIM": 0.691, "LPIPS": 0.193, "DISTS": 0.088,
               "MUSIQ": 65.70, "CLIP-IQA": 0.386, "NIQE": 2.77,
               "tLPIPS": 15.25, "tOF": 11.12},
    "Vid4":   {"PSNR": 22.64, "SSIM": 0.664, "LPIPS": 0.194, "DISTS": 0.114,
               "MUSIQ": 66.15, "CLIP-IQA": 0.408, "NIQE": 3.32,
               "tLPIPS": 28.53, "tOF": 0.92},
    "UDM10":  {"PSNR": 25.54, "SSIM": 0.811, "LPIPS": 0.124, "DISTS": 0.066,
               "MUSIQ": 63.20, "CLIP-IQA": 0.447, "NIQE": 4.12,
               "tLPIPS": 14.48, "tOF": 2.518},
    "SPMCS":  {"PSNR": 19.94, "SSIM": 0.506, "LPIPS": 0.183, "DISTS": 0.106,
               "MUSIQ": 67.22, "CLIP-IQA": 0.503, "NIQE": 3.70,
               "tLPIPS": 31.59, "tOF": 0.576},
}

direction = {
    "PSNR": "↑", "SSIM": "↑",
    "LPIPS": "↓", "DISTS": "↓",
    "MUSIQ": "↑", "CLIP-IQA": "↑",
    "NIQE": "↓",
    "tLPIPS": "↓", "tOF": "↓",
}

groups = [
    ("Reconstruction", ["PSNR", "SSIM"]),
    ("Reference Perceptual", ["LPIPS", "DISTS"]),
    ("No-reference Perceptual", ["MUSIQ", "CLIP-IQA", "NIQE"]),
    ("Temporal", ["tLPIPS", "tOF"]),
]

wb = Workbook()
ws = wb.active
ws.title = "Dual-SFT Results"

header_font = Font(bold=True, color="FFFFFF")
header_fill = PatternFill("solid", fgColor="305496")
group_fill = PatternFill("solid", fgColor="8EA9DB")
metric_font = Font(bold=True)
center = Alignment(horizontal="center", vertical="center")
thin = Side(border_style="thin", color="999999")
border = Border(left=thin, right=thin, top=thin, bottom=thin)

# Row 1: group headers
ws.cell(row=1, column=1, value="Dataset").font = header_font
ws.cell(row=1, column=1).fill = header_fill
ws.cell(row=1, column=1).alignment = center
ws.cell(row=1, column=1).border = border

col = 2
for group_name, metrics in groups:
    span = len(metrics)
    ws.cell(row=1, column=col, value=group_name).font = header_font
    ws.cell(row=1, column=col).fill = header_fill
    ws.cell(row=1, column=col).alignment = center
    if span > 1:
        ws.merge_cells(start_row=1, start_column=col,
                       end_row=1, end_column=col + span - 1)
    for c in range(col, col + span):
        ws.cell(row=1, column=c).border = border
    col += span

# Row 2: metric names with direction arrows
all_metrics = []
for _, metrics in groups:
    all_metrics.extend(metrics)

ws.cell(row=2, column=1, value="").fill = group_fill
ws.cell(row=2, column=1).border = border
for i, m in enumerate(all_metrics, start=2):
    cell = ws.cell(row=2, column=i, value=f"{m} {direction[m]}")
    cell.font = metric_font
    cell.fill = group_fill
    cell.alignment = center
    cell.border = border

# Data rows
for r, (dataset, vals) in enumerate(results.items(), start=3):
    name_cell = ws.cell(row=r, column=1, value=dataset)
    name_cell.font = metric_font
    name_cell.alignment = center
    name_cell.border = border
    for i, m in enumerate(all_metrics, start=2):
        cell = ws.cell(row=r, column=i, value=vals[m])
        cell.alignment = center
        cell.border = border
        # Number format
        if m in ("PSNR", "MUSIQ", "tLPIPS"):
            cell.number_format = "0.00"
        elif m in ("NIQE", "tOF"):
            cell.number_format = "0.000"
        else:
            cell.number_format = "0.000"

# Auto width
for col_idx in range(1, len(all_metrics) + 2):
    letter = get_column_letter(col_idx)
    if col_idx == 1:
        ws.column_dimensions[letter].width = 12
    else:
        ws.column_dimensions[letter].width = 11

# Freeze header
ws.freeze_panes = "B3"

# Notes sheet
ws2 = wb.create_sheet("Notes")
notes = [
    "Dual-SFT (frequency-separated) evaluation results",
    "",
    "Training: REDS only, 20K steps, 2x A6000",
    "Inference: 50 DDPM steps, bidirectional sampling",
    "",
    "Direction: ↑ higher is better, ↓ lower is better",
    "",
    "Metrics:",
    "  PSNR / SSIM        - reconstruction (pixel-level)",
    "  LPIPS / DISTS      - reference perceptual (deep features)",
    "  MUSIQ / CLIP-IQA   - no-reference perceptual (naturalness)",
    "  NIQE               - no-reference natural-image statistics",
    "  tLPIPS             - temporal LPIPS (frame-pair perceptual diff)",
    "  tOF                - temporal optical-flow consistency",
    "",
    "Datasets (all BIx4 degradation):",
    "  REDS4   - 4 clips from REDS train split (000, 011, 015, 020)",
    "  Vid4    - calendar, city, foliage, walk",
    "  UDM10   - 10 indoor/studio clips",
    "  SPMCS   - 30 diverse natural clips (Tao et al. 2017)",
]
for i, line in enumerate(notes, start=1):
    cell = ws2.cell(row=i, column=1, value=line)
    if i == 1:
        cell.font = Font(bold=True, size=12)
ws2.column_dimensions["A"].width = 80

out = "results_dualsft.xlsx"
wb.save(out)
print(f"Wrote {out}")
