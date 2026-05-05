"""Find best frame for qualitative comparison using PSNR/SSIM (CPU only).

Computes per-frame PSNR for each method on REDS4 and ranks frames by:
- Top: where Ours has highest absolute PSNR (fidelity highlight)
- Top: where Ours wins by largest gap vs competitors

Usage:
    python find_best_frame_psnr.py [--top 5]
"""

import os
import argparse
import numpy as np
from PIL import Image


GT_ROOT = '/mnt/HDD_raid1/yjcho/data/REDS/test/gt'

METHODS = {
    'BasicVSR++': '/mnt/HDD_raid1/yjcho/BasicVSR_PlusPlus/reds',
    'StableVSR':  '/mnt/HDD_raid1/yjcho/stablevsr_reds',
    'Ours':       '/mnt/HDD_raid1/yjcho/20260430/reds',
}

SEQUENCES = ['000', '011', '015', '020']


def psnr(a, b, max_val=1.0):
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(max_val / np.sqrt(mse))


def load_aligned(gt_path, pred_path):
    gt = np.array(Image.open(gt_path).convert('RGB')).astype(np.float32) / 255.0
    pr = np.array(Image.open(pred_path).convert('RGB')).astype(np.float32) / 255.0
    if gt.shape != pr.shape:
        h = min(gt.shape[0], pr.shape[0])
        w = min(gt.shape[1], pr.shape[1])
        gh = (gt.shape[0] - h) // 2
        gw = (gt.shape[1] - w) // 2
        ph = (pr.shape[0] - h) // 2
        pw = (pr.shape[1] - w) // 2
        gt = gt[gh:gh+h, gw:gw+w]
        pr = pr[ph:ph+h, pw:pw+w]
    return gt, pr


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--top', type=int, default=5)
    args = p.parse_args()

    for seq in SEQUENCES:
        gt_dir = os.path.join(GT_ROOT, seq)
        frames = sorted(os.listdir(gt_dir))

        per_frame = {m: [] for m in METHODS}
        for fr in frames:
            gt_path = os.path.join(gt_dir, fr)
            for m, root in METHODS.items():
                pred_path = os.path.join(root, seq, fr)
                if not os.path.isfile(pred_path):
                    per_frame[m].append(np.nan)
                    continue
                gt, pr = load_aligned(gt_path, pred_path)
                per_frame[m].append(psnr(gt, pr))

        arr = {m: np.array(v) for m, v in per_frame.items()}

        # gap = Ours - (BasicVSR++ + StableVSR)/2  (higher = Ours fidelity better)
        gap = arr['Ours'] - (arr['BasicVSR++'] + arr['StableVSR']) / 2

        order_abs = np.argsort(-arr['Ours'])  # Ours highest absolute PSNR
        order_gap = np.argsort(-gap)          # Ours wins by largest margin

        order_abs = [i for i in order_abs if not np.isnan(arr['Ours'][i])]
        order_gap = [i for i in order_gap if not np.isnan(gap[i])]

        print(f"\n=== Sequence {seq} (top {args.top} highest absolute Ours PSNR) ===")
        print(f"{'rank':>4} {'frame':>10}  {'Ours PSNR':>10}  {'BasicVSR++':>11} {'StableVSR':>10}")
        for rank, idx in enumerate(order_abs[:args.top], 1):
            print(f"  {rank:>2}.  {frames[idx]:>10}  {arr['Ours'][idx]:>10.3f}  "
                  f"{arr['BasicVSR++'][idx]:>11.3f}  {arr['StableVSR'][idx]:>10.3f}")

        print(f"\n=== Sequence {seq} (top {args.top} largest fidelity gap, Ours wins) ===")
        print(f"{'rank':>4} {'frame':>10}  {'gap':>8}  {'Ours':>8}  {'BasicVSR++':>11} {'StableVSR':>10}")
        for rank, idx in enumerate(order_gap[:args.top], 1):
            print(f"  {rank:>2}.  {frames[idx]:>10}  {gap[idx]:+.3f}   "
                  f"{arr['Ours'][idx]:>8.3f}  {arr['BasicVSR++'][idx]:>11.3f}  "
                  f"{arr['StableVSR'][idx]:>10.3f}")


if __name__ == '__main__':
    main()
