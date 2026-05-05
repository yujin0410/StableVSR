"""Frequency-band summary table across REDS4 sequences.

Computes radial-averaged 2D-FFT power ratio (method/GT) on each REDS4
sequence (000, 011, 015, 020), averages it across sequences, then
groups frequencies into Low / Mid / High bands and prints a summary
table suitable for a paper.

Usage:
    python make_freq_summary_table.py
"""

import os
import numpy as np
from PIL import Image


PATHS = {
    'LR (bicubic up)': '/mnt/HDD_raid1/yjcho/data/REDS/test/bicubic/{seq}/{idx:08d}.png',
    'BasicVSR++':      '/mnt/HDD_raid1/yjcho/BasicVSR_PlusPlus/reds/{seq}/{idx:08d}.png',
    'StableVSR':       '/mnt/HDD_raid1/yjcho/stablevsr_reds/{seq}/{idx:08d}.png',
    'Ours':            '/mnt/HDD_raid1/yjcho/20260430/reds/{seq}/{idx:08d}.png',
    'GT':              '/mnt/HDD_raid1/yjcho/data/REDS/test/gt/{seq}/{idx:08d}.png',
}

SEQUENCES = ['000', '011', '015', '020']
NUM_FRAMES = 100

BANDS = [
    ('Low (0-0.05)',   0.0,  0.05),
    ('Mid (0.05-0.15)', 0.05, 0.15),
    ('High (0.15-0.5)', 0.15, 0.5),
]


def load_y(path, upscale4=False):
    img = Image.open(path).convert('RGB')
    if upscale4:
        img = img.resize((img.width * 4, img.height * 4), Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def radial_profile(power_2d):
    H, W = power_2d.shape
    cy, cx = H / 2.0, W / 2.0
    y, x = np.indices((H, W))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(np.int32)
    tbin = np.bincount(r.ravel(), power_2d.ravel())
    nr = np.bincount(r.ravel())
    profile = tbin / np.maximum(nr, 1)
    rmax = int(min(H, W) // 2)
    return profile[:rmax]


def compute_method_spectrum(seq, name, tmpl, num_frames):
    upscale4 = (name == 'LR (bicubic up)')
    profiles = []
    for t in range(num_frames):
        path = tmpl.format(seq=seq, idx=t)
        if not os.path.isfile(path):
            continue
        y = load_y(path, upscale4=upscale4)
        F = np.fft.fftshift(np.fft.fft2(y))
        profiles.append(radial_profile(np.abs(F) ** 2))
    if not profiles:
        return None
    L = min(len(p) for p in profiles)
    return np.stack([p[:L] for p in profiles], axis=0).mean(axis=0)


def main():
    methods = list(PATHS.keys())
    methods_no_gt = [m for m in methods if m != 'GT']

    # collect per-sequence ratio profiles
    per_seq_ratio = {m: [] for m in methods_no_gt}
    common_freq = None

    for seq in SEQUENCES:
        spectra = {}
        for name, tmpl in PATHS.items():
            sp = compute_method_spectrum(seq, name, tmpl, NUM_FRAMES)
            if sp is not None:
                spectra[name] = sp
        L = min(len(s) for s in spectra.values())
        gt = spectra['GT'][:L]
        freq = np.arange(L) / (2 * L)
        if common_freq is None or len(freq) < len(common_freq):
            common_freq = freq
        for m in methods_no_gt:
            r = np.maximum(spectra[m][:L], 1e-12) / np.maximum(gt, 1e-12)
            per_seq_ratio[m].append((freq, r))

    # average across sequences (interpolate to common_freq)
    avg_ratio = {}
    for m in methods_no_gt:
        rs = []
        for f, r in per_seq_ratio[m]:
            r_interp = np.interp(common_freq, f, r)
            rs.append(r_interp)
        avg_ratio[m] = np.mean(np.stack(rs, axis=0), axis=0)

    # banded mean
    print("\n=== Frequency-band summary (mean Power_method / Power_GT) ===")
    print(f"Averaged over {len(SEQUENCES)} REDS4 sequences x {NUM_FRAMES} frames each.")
    print()
    header = f"{'Method':<18} | " + " | ".join(f"{name:>16}" for name, _, _ in BANDS)
    print(header)
    print("-" * len(header))
    for m in methods_no_gt:
        cells = []
        for _, lo, hi in BANDS:
            mask = (common_freq >= lo) & (common_freq < hi)
            v = avg_ratio[m][mask].mean() if mask.any() else float('nan')
            cells.append(f"{v:>16.3f}")
        print(f"{m:<18} | " + " | ".join(cells))

    # also LaTeX-friendly version
    print("\n=== LaTeX table snippet ===")
    print(r"\begin{tabular}{l" + "c" * len(BANDS) + "}")
    print(r"\toprule")
    print(r"Method & " + " & ".join(name for name, _, _ in BANDS) + r" \\")
    print(r"\midrule")
    for m in methods_no_gt:
        cells = []
        for _, lo, hi in BANDS:
            mask = (common_freq >= lo) & (common_freq < hi)
            v = avg_ratio[m][mask].mean() if mask.any() else float('nan')
            cells.append(f"{v:.3f}")
        # bold the highest in each band? (we know Ours is best at mid/high)
        print(f"{m} & " + " & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == '__main__':
    main()
