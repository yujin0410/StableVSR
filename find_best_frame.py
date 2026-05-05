"""Find the best frame for qualitative comparison on REDS4.

For each REDS4 sequence, compute per-frame LPIPS for BasicVSR++,
StableVSR baseline, and Ours, and rank frames by the perceptual gap
(competitors' LPIPS minus Ours' LPIPS). The top frames are the most
demonstrative for showing Ours' perceptual advantage.

Usage:
    python find_best_frame.py [--top 5]
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, CenterCrop
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS


GT_ROOT = '/mnt/HDD_raid1/yjcho/data/REDS/test/gt'

METHODS = {
    'BasicVSR++': '/mnt/HDD_raid1/yjcho/BasicVSR_PlusPlus/reds',
    'StableVSR':  '/mnt/HDD_raid1/yjcho/stablevsr_reds',
    'Ours':       '/mnt/HDD_raid1/yjcho/20260430/reds',
}

SEQUENCES = ['000', '011', '015', '020']


def load_aligned(gt_path, pred_path, tt, device):
    gt = tt(Image.open(gt_path).convert('RGB')).unsqueeze(0).to(device)
    pr = tt(Image.open(pred_path).convert('RGB')).unsqueeze(0).to(device)
    if gt.shape[-2:] != pr.shape[-2:]:
        h = min(gt.shape[-2], pr.shape[-2])
        w = min(gt.shape[-1], pr.shape[-1])
        gt = CenterCrop((h, w))(gt)
        pr = CenterCrop((h, w))(pr)
    return gt, pr


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--top', type=int, default=5,
                   help="How many top frames to print per sequence")
    args = p.parse_args()

    device = 'cuda'
    lpips = LPIPS(normalize=True).to(device)
    tt = ToTensor()

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
                with torch.no_grad():
                    gt, pr = load_aligned(gt_path, pred_path, tt, device)
                    per_frame[m].append(lpips(gt, pr).item())

        arr = {m: np.array(v) for m, v in per_frame.items()}

        # gap = (BasicVSR++ + StableVSR) / 2 - Ours
        # higher gap means Ours is much better (lower LPIPS) than competitors
        gap = (arr['BasicVSR++'] + arr['StableVSR']) / 2 - arr['Ours']

        order = np.argsort(-gap)  # descending: largest gap first
        order = [i for i in order if not np.isnan(gap[i])]

        print(f"\n=== Sequence {seq} (top {args.top} frames where Ours wins most) ===")
        print(f"{'rank':>4} {'frame':>10} {'gap':>8}  {'BasicVSR++':>12} {'StableVSR':>12} {'Ours':>8}")
        for rank, idx in enumerate(order[:args.top], 1):
            fr_name = frames[idx]
            print(f"  {rank:>2}.  {fr_name:>10} {gap[idx]:+.4f}    "
                  f"{arr['BasicVSR++'][idx]:.4f}      {arr['StableVSR'][idx]:.4f}    "
                  f"{arr['Ours'][idx]:.4f}")

        # also show 5 worst (where Ours is worst)
        print(f"\n  (worst {args.top} where Ours loses)")
        worst_order = order[::-1]
        for rank, idx in enumerate(worst_order[:args.top], 1):
            fr_name = frames[idx]
            print(f"  {rank:>2}.  {fr_name:>10} {gap[idx]:+.4f}    "
                  f"{arr['BasicVSR++'][idx]:.4f}      {arr['StableVSR'][idx]:.4f}    "
                  f"{arr['Ours'][idx]:.4f}")


if __name__ == '__main__':
    main()
