"""Placement-regularization evidence for dual-SFT, from existing outputs.

Quantifies the claim "dual-SFT regularizes WHERE the prior places high-frequency
detail (placement), to the input structure" -- no retraining.

Analyses:
  (1) structure alignment   : gradient correlation (out vs GT, out vs LR) and
                              Canny-edge Chamfer distance (out <-> GT). If the
                              regularized model aligns to GT structure better
                              than a reference -> placement is regularized.
  (2) radial power spectrum  : 1D radially-averaged power of out / GT / ref.
                              Shows in which band the output moved toward GT
                              ("LF locked, HF free").
  (3) seed-variance map      : --seed_dirs d1,d2,...  per-pixel std across
                              outputs generated from the SAME LR with different
                              diffusion seeds. Low std on edges (placement fixed)
                              + high std on texture (content free) is the single
                              strongest evidence for C1. (Needs you to generate
                              the seed runs first -- see note at bottom.)

Requires: numpy, opencv-python, scipy, matplotlib

Usage (1)(2):
    python scripts/analyze_placement.py \
        --out_path /path/SR_out --gt_path /path/GT \
        [--ref_path /path/StableVSR_or_bicubic] [--lr_path /path/LR_bicubic] \
        --tag dualsft_reds

Usage (3):
    python scripts/analyze_placement.py --seed_dirs run42,run7,run123 \
        --gt_path /path/GT --tag dualsft_seedvar
"""

import os
import glob
import argparse

import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_IMG = ('.png', '.jpg', '.jpeg', '.bmp')


def seqs_of(root):
    return [d for d in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, d))
            and any(f.lower().endswith(_IMG) for f in os.listdir(os.path.join(root, d)))]


def frames_of(d):
    return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.lower().endswith(_IMG)]


def luma(path, like=None):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if like is not None and im.shape[:2] != like:
        im = cv2.resize(im, (like[1], like[0]), interpolation=cv2.INTER_CUBIC)
    y = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return y


def grad_mag(y):
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def pearson(a, b):
    a, b = a.ravel(), b.ravel()
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum()) + 1e-8
    return float((a * b).sum() / d)


def chamfer_edges(yo, yg):
    eo = cv2.Canny((yo * 255).astype(np.uint8), 80, 160) > 0
    eg = cv2.Canny((yg * 255).astype(np.uint8), 80, 160) > 0
    if eo.sum() == 0 or eg.sum() == 0:
        return float('nan')
    dt_g = distance_transform_edt(~eg)   # dist to nearest GT edge
    dt_o = distance_transform_edt(~eo)
    return 0.5 * (dt_g[eo].mean() + dt_o[eg].mean())   # symmetric, lower=better


def rapsd(y):
    F = np.fft.fftshift(np.fft.fft2(y))
    P = np.abs(F) ** 2
    h, w = y.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
    tbin = np.bincount(r.ravel(), P.ravel())
    nr = np.bincount(r.ravel())
    prof = tbin / (nr + 1e-8)
    return prof[:min(cy, cx)]


def analyze_align_spectrum(args):
    oseqs = seqs_of(args.out_path)
    if args.max_seqs:
        oseqs = oseqs[:args.max_seqs]
    gcorr_gt, gcorr_lr, gcorr_gt_ref, cham, cham_ref = [], [], [], [], []
    sp_out, sp_gt, sp_ref = [], [], []
    for seq in oseqs:
        ofs = frames_of(os.path.join(args.out_path, seq))
        gdir = os.path.join(args.gt_path, seq) if args.gt_path else None
        gfs = frames_of(gdir) if gdir and os.path.isdir(gdir) else None
        rfs = frames_of(os.path.join(args.ref_path, seq)) if args.ref_path and os.path.isdir(os.path.join(args.ref_path, seq)) else None
        lfs = frames_of(os.path.join(args.lr_path, seq)) if args.lr_path and os.path.isdir(os.path.join(args.lr_path, seq)) else None
        for i, of in enumerate(ofs):
            yo = luma(of)
            sp_out.append(rapsd(yo))
            if gfs and i < len(gfs):
                yg = luma(gfs[i], like=yo.shape)
                go, gg = grad_mag(yo), grad_mag(yg)
                gcorr_gt.append(pearson(go, gg))
                cham.append(chamfer_edges(yo, yg))
                sp_gt.append(rapsd(yg))
                if lfs and i < len(lfs):
                    yl = luma(lfs[i], like=yo.shape)
                    gcorr_lr.append(pearson(go, grad_mag(yl)))
                if rfs and i < len(rfs):
                    yr = luma(rfs[i], like=yo.shape)
                    gcorr_gt_ref.append(pearson(grad_mag(yr), gg))
                    cham_ref.append(chamfer_edges(yr, yg))
                    sp_ref.append(rapsd(yr))

    print('\n===============  placement / structure alignment  ===============')
    print(f'tag={args.tag}  frames={len(sp_out)}')
    if gcorr_gt:
        print(f'(1) gradient corr  out vs GT : {np.nanmean(gcorr_gt):.4f}'
              + (f'   |  ref vs GT : {np.nanmean(gcorr_gt_ref):.4f}' if gcorr_gt_ref else ''))
    if gcorr_lr:
        print(f'    gradient corr  out vs LR : {np.nanmean(gcorr_lr):.4f}')
    if cham:
        print(f'(1) edge Chamfer   out<->GT : {np.nanmean(cham):.3f} px (lower=better)'
              + (f'   |  ref<->GT : {np.nanmean(cham_ref):.3f} px' if cham_ref else ''))
        print('    -> if out is closer to GT structure than ref: placement is regularized')

    # (2) power spectrum plot
    if sp_out:
        L = min(len(s) for s in sp_out)
        po = np.mean([s[:L] for s in sp_out], 0)
        plt.figure(figsize=(6, 4))
        freqs = np.arange(L) / L
        plt.semilogy(freqs, po, label=f'out ({args.tag})')
        if sp_gt:
            Lg = min(L, min(len(s) for s in sp_gt))
            plt.semilogy(np.arange(Lg) / Lg, np.mean([s[:Lg] for s in sp_gt], 0), '--', label='GT')
        if sp_ref:
            Lr = min(L, min(len(s) for s in sp_ref))
            plt.semilogy(np.arange(Lr) / Lr, np.mean([s[:Lr] for s in sp_ref], 0), ':', label='ref')
        plt.xlabel('normalized spatial frequency (0=LF, 0.5=HF)')
        plt.ylabel('radially-averaged power')
        plt.title('(2) power spectrum: where output sits vs GT')
        plt.legend(); plt.tight_layout()
        out_png = f'{args.tag}_rapsd.png'
        plt.savefig(out_png, dpi=130)
        print(f'(2) power spectrum saved -> {out_png}')
    print('=================================================================\n')


def analyze_seed_variance(args):
    dirs = args.seed_dirs.split(',')
    seqs = seqs_of(dirs[0])
    if args.max_seqs:
        seqs = seqs[:args.max_seqs]
    var_edge, var_flat = [], []
    saved = 0
    for seq in seqs:
        fl = [frames_of(os.path.join(d, seq)) for d in dirs]
        n = min(len(x) for x in fl)
        for i in range(n):
            stack = []
            for k in range(len(dirs)):
                im = cv2.imread(fl[k][i], cv2.IMREAD_COLOR).astype(np.float32) / 255.0
                stack.append(im)
            stack = np.stack(stack, 0)                 # (K,H,W,3)
            std = stack.std(0).mean(-1)                # (H,W) per-pixel std across seeds
            # edge vs flat split (edges from the seed-mean image)
            mean_im = stack.mean(0)
            yl = cv2.cvtColor((mean_im * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            edge = cv2.dilate((cv2.Canny(yl, 80, 160) > 0).astype(np.uint8), np.ones((3, 3)), 1) > 0
            var_edge.append(std[edge].mean() if edge.sum() else np.nan)
            var_flat.append(std[~edge].mean() if (~edge).sum() else np.nan)
            if saved < args.swap_examples:
                hm = (std / (std.max() + 1e-8) * 255).astype(np.uint8)
                cv2.imwrite(f'{args.tag}_seedvar_{seq}_{i}.png', cv2.applyColorMap(hm, cv2.COLORMAP_JET))
                cv2.imwrite(f'{args.tag}_seedmean_{seq}_{i}.png', (mean_im * 255).astype(np.uint8))
                saved += 1

    print('\n===============  (3) seed-variance (placement vs content)  ===============')
    print(f'tag={args.tag}  seeds={len(dirs)}  frames={len(var_edge)}')
    print(f'  std on EDGES (structure) : {np.nanmean(var_edge):.4f}   <- low = placement FIXED')
    print(f'  std on FLAT  (texture)   : {np.nanmean(var_flat):.4f}   <- high = content FREE')
    ratio = np.nanmean(var_flat) / (np.nanmean(var_edge) + 1e-8)
    print(f'  flat/edge ratio          : {ratio:.2f}   (>1 supports "placement fixed, content free")')
    print(f'  heatmaps saved: {args.tag}_seedvar_*.png')
    print('==========================================================================\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_path', default=None)
    ap.add_argument('--gt_path', default=None)
    ap.add_argument('--ref_path', default=None, help='2nd model (StableVSR/bicubic) for ON-vs-OFF')
    ap.add_argument('--lr_path', default=None, help='LR (bicubic upsampled) for out-vs-LR gradient corr')
    ap.add_argument('--seed_dirs', default=None, help='comma list of output dirs from different seeds (method 3)')
    ap.add_argument('--tag', default='model')
    ap.add_argument('--max_seqs', type=int, default=0)
    ap.add_argument('--swap_examples', type=int, default=3)
    args = ap.parse_args()

    if args.seed_dirs:
        analyze_seed_variance(args)
    else:
        assert args.out_path, '--out_path required for methods (1)(2)'
        analyze_align_spectrum(args)


if __name__ == '__main__':
    main()
