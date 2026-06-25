"""Standalone evaluator = eval.py + per-sequence tLPIPS/tOF breakdown.

Identical metric definitions to the repo's eval.py (so numbers match the
paper tables), but additionally prints a per-sequence breakdown of the
temporal metrics. This answers a diagnostic question the aggregate mean
cannot: is a bad OOD tLPIPS spread evenly across sequences, or driven by
a few outlier clips? (The latter means the fix is local and much easier.)

This is a NEW file; it does not modify the repo's eval.py.

Usage (same args as eval.py):
    python ablation_tools/eval_per_seq.py \
        --out_path ./udm10_no_high --gt_path /path/to/UDM10/gt
"""

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
import pyiqa
from DISTS_pytorch import DISTS
from torchvision.models.optical_flow import raft_large as raft
import os
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from util.flow_utils import get_flow
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Per-sequence evaluator for StableVSR.")
parser.add_argument("--out_path", type=str, default='./StableVSR_results/',
                    help="Path to output folder containing the upscaled frames.")
parser.add_argument("--gt_path", type=str, required=True, help="Path to folder with GT frames.")
args = parser.parse_args()

print("Run with arguments:")
for arg, value in vars(args).items():
    print(f"  {arg}: {value}")

gt_path = args.gt_path
rec_path = args.out_path
seqs = sorted(os.listdir(rec_path))

device = torch.device('cuda')
of_model = raft(pretrained=True).to(device)
lpips = LPIPS(normalize=True).to(device)
dists = DISTS().to(device)
psnr = PSNR(data_range=1).to(device)
ssim = SSIM(data_range=1).to(device)
musiq = pyiqa.create_metric('musiq', device='cuda', as_loss=False)
niqe = pyiqa.create_metric('niqe', device='cuda', as_loss=False)
clip = pyiqa.create_metric('clipiqa', device='cuda', as_loss=False)

lpips_dict, psnr_dict, ssim_dict, dists_dict = {}, {}, {}, {}
musiq_dict, niqe_dict, clip_dict, tlpips_dict, tof_dict = {}, {}, {}, {}, {}
tt = ToTensor()

total = 0
for root, dirs, files in os.walk(gt_path):
    total += len(files)
pbar = tqdm(total=total, ncols=100)

for seq in seqs:
    ims_rec = sorted(os.listdir(os.path.join(rec_path, seq)))
    ims_gt = sorted(os.listdir(os.path.join(gt_path, seq)))

    for d in (lpips_dict, psnr_dict, ssim_dict, dists_dict, musiq_dict,
              niqe_dict, clip_dict, tlpips_dict, tof_dict):
        d[seq] = []

    for i, (im_rec, im_gt) in enumerate(zip(ims_rec, ims_gt)):
        with torch.no_grad():
            gt = tt(Image.open(os.path.join(gt_path, seq, im_gt))).unsqueeze(0).to(device)
            rec = tt(Image.open(os.path.join(rec_path, seq, im_rec))).unsqueeze(0).to(device)

            psnr_value = psnr(gt, rec)
            ssim_value = ssim(gt, rec)
            lpips_value = lpips(gt, rec)
            dists_value = dists(gt, rec)
            musiq_value = musiq(rec)
            niqe_value = niqe(rec)
            clip_value = clip(rec)
            if i > 0:
                tlpips_value = (lpips(gt, prev_gt) - lpips(rec, prev_rec)).abs()
                tlpips_dict[seq].append(tlpips_value.item())
                tof_value = (get_flow(of_model, rec, prev_rec) - get_flow(of_model, gt, prev_gt)).abs().mean()
                tof_dict[seq].append(tof_value.item())

        psnr_dict[seq].append(psnr_value.item())
        ssim_dict[seq].append(ssim_value.item())
        lpips_dict[seq].append(lpips_value.item())
        dists_dict[seq].append(dists_value.item())
        musiq_dict[seq].append(musiq_value.item())
        niqe_dict[seq].append(niqe_value.item())
        clip_dict[seq].append(clip_value.item())

        prev_rec = rec
        prev_gt = gt
        pbar.update()

pbar.close()

mean_lpips = np.round(np.mean([np.mean(lpips_dict[k]) for k in lpips_dict]), 3)
mean_dists = np.round(np.mean([np.mean(dists_dict[k]) for k in dists_dict]), 3)
mean_psnr = np.round(np.mean([np.mean(psnr_dict[k]) for k in psnr_dict]), 2)
mean_ssim = np.round(np.mean([np.mean(ssim_dict[k]) for k in ssim_dict]), 3)
mean_musiq = np.round(np.mean([np.mean(musiq_dict[k]) for k in musiq_dict]), 2)
mean_niqe = np.round(np.mean([np.mean(niqe_dict[k]) for k in niqe_dict]), 2)
mean_clip = np.round(np.mean([np.mean(clip_dict[k]) for k in clip_dict]), 3)
mean_tlpips = np.round(np.mean([np.mean(tlpips_dict[k]) for k in tlpips_dict]) * 1e3, 2)
mean_tof = np.round(np.mean([np.mean(tof_dict[k]) for k in tof_dict]) * 1e1, 3)

# ---- per-sequence temporal breakdown (the diagnostic addition) ----
print("\n=== Per-sequence tLPIPS (x1e3, desc) | tOF (x1e1) ===")
per_seq_tl = {k: np.mean(v) * 1e3 for k, v in tlpips_dict.items() if len(v)}
per_seq_to = {k: np.mean(tof_dict[k]) * 1e1 for k in per_seq_tl}
for k, v in sorted(per_seq_tl.items(), key=lambda kv: -kv[1]):
    print(f"  {k:24s} tLPIPS={v:8.2f}   tOF={per_seq_to[k]:7.3f}   (n={len(tlpips_dict[k])})")
_v = np.array(list(per_seq_tl.values()))
print(f"  --> tLPIPS  mean={_v.mean():.2f}  median={np.median(_v):.2f}  "
      f"std={_v.std():.2f}  min={_v.min():.2f}  max={_v.max():.2f}")
print("=" * 60)

print(f'PSNR: {mean_psnr}, SSIM: {mean_ssim}, LPIPS: {mean_lpips}, DISTS: {mean_dists}, '
      f'MUSIQ: {mean_musiq}, CLIP: {mean_clip}, NIQE: {mean_niqe}, '
      f'tLPIPS: {mean_tlpips}, tOF: {mean_tof}')
