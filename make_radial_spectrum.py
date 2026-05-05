"""Radial-averaged 2D FFT spectrum for VSR comparison.

For each method, compute the 2D FFT magnitude of every frame in the
sequence, radially average it, then average across frames. Plots all
methods on a single log-log graph so high-frequency reconstruction
quality can be compared directly against GT.

Usage:
    python make_radial_spectrum.py --seq 000 \
        --output figures/radial_spectrum_000.pdf
"""

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


PATHS = {
    'LR (bicubic up)': '/mnt/HDD_raid1/yjcho/data/REDS/test/bicubic/{seq}/{idx:08d}.png',
    'BasicVSR++':      '/mnt/HDD_raid1/yjcho/BasicVSR_PlusPlus/reds/{seq}/{idx:08d}.png',
    'StableVSR':       '/mnt/HDD_raid1/yjcho/stablevsr_reds/{seq}/{idx:08d}.png',
    'Ours':            '/mnt/HDD_raid1/yjcho/20260430/reds/{seq}/{idx:08d}.png',
    'GT':              '/mnt/HDD_raid1/yjcho/data/REDS/test/gt/{seq}/{idx:08d}.png',
}

COLORS = {
    'LR (bicubic up)': '#bbbbbb',
    'BasicVSR++':      '#1f77b4',
    'StableVSR':       '#2ca02c',
    'Ours':            '#d62728',
    'GT':              'black',
}

LINESTYLES = {
    'LR (bicubic up)': ':',
    'BasicVSR++':      '-',
    'StableVSR':       '-',
    'Ours':            '-',
    'GT':              '--',
}


def load_grayscale(path, upscale4=False):
    img = Image.open(path).convert('RGB')
    if upscale4:
        img = img.resize((img.width * 4, img.height * 4), Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0
    # luminance (Y channel)
    y = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return y


def radial_profile(power_2d):
    """Average 2D power spectrum over concentric radii."""
    H, W = power_2d.shape
    cy, cx = H / 2.0, W / 2.0
    y, x = np.indices((H, W))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_int = r.astype(np.int32)
    tbin = np.bincount(r_int.ravel(), power_2d.ravel())
    nr = np.bincount(r_int.ravel())
    profile = tbin / np.maximum(nr, 1)
    # truncate to valid range (radius up to min(H, W) / 2)
    rmax = int(min(H, W) // 2)
    return profile[:rmax]


def compute_method_spectrum(seq, name, tmpl, num_frames):
    profiles = []
    upscale4 = (name == 'LR (bicubic up)')
    for t in range(num_frames):
        path = tmpl.format(seq=seq, idx=t)
        if not os.path.isfile(path):
            continue
        y = load_grayscale(path, upscale4=upscale4)
        F = np.fft.fft2(y)
        F = np.fft.fftshift(F)
        power = np.abs(F) ** 2
        profiles.append(radial_profile(power))
    if not profiles:
        return None
    # truncate all profiles to common length, then average
    L = min(len(p) for p in profiles)
    profiles = np.stack([p[:L] for p in profiles], axis=0)
    return profiles.mean(axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seq', default='000')
    p.add_argument('--num_frames', type=int, default=100)
    p.add_argument('--output', default='figures/radial_spectrum.pdf')
    p.add_argument('--ratio', action='store_true',
                   help="plot power ratio (method/GT) instead of raw spectrum")
    p.add_argument('--combined', action='store_true',
                   help="2-panel: top=raw spectrum, bottom=ratio")
    p.add_argument('--xmin', type=float, default=None,
                   help="optional lower x-limit for highlighting high-freq")
    args = p.parse_args()

    spectra = {}
    for name, tmpl in PATHS.items():
        prof = compute_method_spectrum(args.seq, name, tmpl, args.num_frames)
        if prof is None:
            print(f"  Skipping {name}: no frames found.")
            continue
        spectra[name] = prof

    # align lengths
    L = min(len(p) for p in spectra.values())
    spectra = {k: v[:L] for k, v in spectra.items()}
    freq = np.arange(L) / (2 * L)

    if args.combined:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
        for name, prof in spectra.items():
            prof = np.maximum(prof, 1e-12)
            ax1.plot(freq, prof, label=name,
                     color=COLORS[name], linestyle=LINESTYLES[name], linewidth=2)
        ax1.set_xscale('log'); ax1.set_yscale('log')
        ax1.set_ylabel('Radial-averaged power', fontsize=14)
        ax1.set_title(f'Radial 2D-FFT power spectrum (REDS4 seq {args.seq})', fontsize=12)
        ax1.legend(fontsize=11, loc='lower left')
        ax1.grid(True, which='both', linestyle='--', alpha=0.4)

        gt = spectra.get('GT')
        if gt is not None:
            for name, prof in spectra.items():
                if name == 'GT':
                    continue
                ratio = np.maximum(prof, 1e-12) / np.maximum(gt, 1e-12)
                ax2.plot(freq, ratio, label=name,
                         color=COLORS[name], linestyle=LINESTYLES[name], linewidth=2)
            ax2.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.6, label='GT (= 1.0)')
        ax2.set_xscale('log')
        ax2.set_ylabel('Power / Power(GT)', fontsize=14)
        ax2.set_xlabel('Spatial frequency (cycles/pixel)', fontsize=14)
        ax2.set_ylim(0, 1.1)
        ax2.legend(fontsize=11, loc='lower left')
        ax2.grid(True, which='both', linestyle='--', alpha=0.4)

        if args.xmin is not None:
            ax1.set_xlim(left=args.xmin)
            ax2.set_xlim(left=args.xmin)

        plt.tight_layout()
    elif args.ratio:
        plt.figure(figsize=(8, 5))
        gt = spectra['GT']
        for name, prof in spectra.items():
            if name == 'GT':
                continue
            ratio = np.maximum(prof, 1e-12) / np.maximum(gt, 1e-12)
            plt.plot(freq, ratio, label=name,
                     color=COLORS[name], linestyle=LINESTYLES[name], linewidth=2)
        plt.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.6, label='GT (= 1.0)')
        plt.xscale('log')
        plt.xlabel('Spatial frequency (cycles/pixel)', fontsize=14)
        plt.ylabel('Power / Power(GT)', fontsize=14)
        plt.title(f'Power ratio to GT (REDS4 seq {args.seq})', fontsize=12)
        plt.ylim(0, 1.1)
        plt.legend(fontsize=11, loc='lower left')
        plt.grid(True, which='both', linestyle='--', alpha=0.4)
        if args.xmin is not None:
            plt.xlim(left=args.xmin)
        plt.tight_layout()
    else:
        plt.figure(figsize=(8, 6))
        for name, prof in spectra.items():
            prof = np.maximum(prof, 1e-12)
            plt.plot(freq, prof, label=name,
                     color=COLORS[name], linestyle=LINESTYLES[name], linewidth=2)
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('Spatial frequency (cycles/pixel)', fontsize=14)
        plt.ylabel('Radial-averaged power', fontsize=14)
        plt.title(f'Radial 2D-FFT power spectrum (REDS4 seq {args.seq}, '
                  f'avg over {args.num_frames} frames)', fontsize=12)
        plt.legend(fontsize=12, loc='lower left')
        plt.grid(True, which='both', linestyle='--', alpha=0.4)
        if args.xmin is not None:
            plt.xlim(left=args.xmin)

    out_dir = os.path.dirname(args.output) or '.'
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    png_path = os.path.splitext(args.output)[0] + '.png'
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {args.output}")
    print(f"Saved: {png_path}")


if __name__ == '__main__':
    main()
