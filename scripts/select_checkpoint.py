#!/usr/bin/env python
# coding=utf-8
"""Validation-driven checkpoint selection / optimal early stopping for StableVSR.

WHY THIS EXISTS
---------------
For a diffusion model the *training loss* (noise MSE averaged over random
timesteps) is essentially flat and does NOT track VSR quality, so you cannot use
"loss converged" as a stopping signal. The reliable signal is the *validation
metric* curve on held-out frames. This script sweeps checkpoints, runs inference
on a small validation subset across several datasets, computes GT-referenced
metrics (LPIPS / DISTS / tLPIPS / tOF -- the same definitions as eval.py), and
applies patience-based early stopping to recommend the optimal checkpoint.

The composite objective encodes the goal discussed for REDS->(REDS+Vimeo)
fine-tuning:
    * recover the clean out-of-domain datasets (UDM10 / SPMCS / Vid4 = "target")
    * without letting the in-domain dataset (REDS4 = "guardrail") regress too far

Metrics are stored to a JSON cache keyed by (checkpoint, dataset), so you can
re-tune the composite weights or resume a sweep without re-running inference.

Usage
-----
    python scripts/select_checkpoint.py \
        --exp_dir experiments/vimeo_finetune \
        --val_config scripts/val_select_config.yaml \
        --recon_dir /tmp/select_recon \
        --num_inference_steps 50 \
        --patience 3

The first evaluated checkpoint is used as the reference (set it to the
fine-tuning start, e.g. checkpoint-20000) so every score is relative to it.
"""

import argparse
import gc
import json
import os
import re
import time
from glob import glob

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import ToTensor

# --- metric backends (mirror eval.py) -------------------------------------
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

from util.flow_utils import get_flow

try:
    from DISTS_pytorch import DISTS
    _HAS_DISTS = True
except Exception:
    _HAS_DISTS = False

# --- StableVSR inference (mirror test.py) ---------------------------------
from pipeline.stablevsr_pipeline import StableVSRPipeline
from diffusers import DDPMScheduler, ControlNetModel

MODEL_ID = 'claudiom4sir/StableVSR'

# Lower-is-better, GT-referenced metrics used for the stopping objective.
OBJECTIVE_METRICS = ['lpips', 'dists', 'tlpips', 'tof']


# ----------------------------------------------------------------------------
# checkpoint discovery
# ----------------------------------------------------------------------------
def discover_checkpoints(exp_dir):
    """Return [(step, controlnet_dir), ...] sorted by step.

    Accepts both accelerate-style `checkpoint-XXXX/controlnet/` and a plain
    diffusers controlnet folder (config.json + safetensors) at top level.
    """
    found = []
    for d in glob(os.path.join(exp_dir, 'checkpoint-*')):
        m = re.search(r'checkpoint-(\d+)', os.path.basename(d))
        if not m:
            continue
        step = int(m.group(1))
        cn = os.path.join(d, 'controlnet')
        cn = cn if os.path.isdir(cn) else d  # fall back to the dir itself
        if os.path.exists(os.path.join(cn, 'config.json')):
            found.append((step, cn))
    found.sort(key=lambda x: x[0])
    return found


# ----------------------------------------------------------------------------
# inference
# ----------------------------------------------------------------------------
def build_pipeline(controlnet_dir, device):
    controlnet = ControlNetModel.from_pretrained(controlnet_dir)
    pipeline = StableVSRPipeline.from_pretrained(MODEL_ID, controlnet=controlnet)
    pipeline.scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder='scheduler')
    pipeline = pipeline.to(device)
    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    return pipeline


def run_inference_sequence(pipeline, of_model, lr_seq_dir, out_seq_dir, steps):
    os.makedirs(out_seq_dir, exist_ok=True)
    frame_names = sorted(os.listdir(lr_seq_dir))
    frames = [Image.open(os.path.join(lr_seq_dir, n)).convert('RGB') for n in frame_names]
    with torch.no_grad():
        out = pipeline('', frames, num_inference_steps=steps,
                       guidance_scale=0, of_model=of_model).images
    out = [f[0] for f in out]
    for name, img in zip(frame_names, out):
        img.save(os.path.join(out_seq_dir, name))


# ----------------------------------------------------------------------------
# metrics (definitions identical to eval.py)
# ----------------------------------------------------------------------------
class MetricBank:
    def __init__(self, device):
        self.device = device
        self.tt = ToTensor()
        self.of_model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()
        self.lpips = LPIPS(normalize=True).to(device)
        self.psnr = PSNR(data_range=1).to(device)
        self.ssim = SSIM(data_range=1).to(device)
        self.dists = DISTS().to(device) if _HAS_DISTS else None

    def eval_sequence(self, rec_seq_dir, gt_seq_dir):
        ims_rec = sorted(os.listdir(rec_seq_dir))
        ims_gt = sorted(os.listdir(gt_seq_dir))
        acc = {k: [] for k in ['psnr', 'ssim', 'lpips', 'dists', 'tlpips', 'tof']}
        prev_rec = prev_gt = None
        for i, (r, g) in enumerate(zip(ims_rec, ims_gt)):
            with torch.no_grad():
                gt = self.tt(Image.open(os.path.join(gt_seq_dir, g)).convert('RGB')).unsqueeze(0).to(self.device)
                rec = self.tt(Image.open(os.path.join(rec_seq_dir, r)).convert('RGB')).unsqueeze(0).to(self.device)
                acc['psnr'].append(self.psnr(gt, rec).item())
                acc['ssim'].append(self.ssim(gt, rec).item())
                acc['lpips'].append(self.lpips(gt, rec).item())
                if self.dists is not None:
                    acc['dists'].append(self.dists(gt, rec).item())
                if i > 0:
                    tlpips = (self.lpips(gt, prev_gt) - self.lpips(rec, prev_rec)).abs()
                    acc['tlpips'].append(tlpips.item())
                    tof = (get_flow(self.of_model, rec, prev_rec)
                           - get_flow(self.of_model, gt, prev_gt)).abs().mean()
                    acc['tof'].append(tof.item())
                prev_rec, prev_gt = rec, gt
        return acc


def aggregate(seq_accs):
    """Mean-over-sequences, applying eval.py's scaling for tLPIPS (*1e3) and tOF (*1e1)."""
    out = {}
    for m in ['psnr', 'ssim', 'lpips', 'dists', 'tlpips', 'tof']:
        per_seq = [np.mean(a[m]) for a in seq_accs if len(a[m]) > 0]
        if not per_seq:
            continue
        val = float(np.mean(per_seq))
        if m == 'tlpips':
            val *= 1e3
        elif m == 'tof':
            val *= 1e1
        out[m] = val
    return out


# ----------------------------------------------------------------------------
# composite scoring + early stopping
# ----------------------------------------------------------------------------
def dataset_relative_score(metrics, ref_metrics, metric_weights):
    """Weighted mean of (metric / reference_metric) over OBJECTIVE_METRICS.

    < 1.0 means improved vs the reference checkpoint (all are lower-is-better).
    """
    num = den = 0.0
    for m in OBJECTIVE_METRICS:
        if m not in metrics or m not in ref_metrics or ref_metrics[m] == 0:
            continue
        w = metric_weights.get(m, 1.0)
        num += w * (metrics[m] / ref_metrics[m])
        den += w
    return num / den if den else float('nan')


def composite(per_dataset_metrics, ref_per_dataset, ds_roles, ds_weights, metric_weights):
    """Return (target_score, guardrail_score) relative to the reference checkpoint."""
    def weighted(role):
        num = den = 0.0
        for ds, role_d in ds_roles.items():
            if role_d != role or ds not in per_dataset_metrics or ds not in ref_per_dataset:
                continue
            s = dataset_relative_score(per_dataset_metrics[ds], ref_per_dataset[ds], metric_weights)
            if np.isnan(s):
                continue
            w = ds_weights.get(ds, 1.0)
            num += w * s
            den += w
        return num / den if den else float('nan')
    return weighted('target'), weighted('guardrail')


# ----------------------------------------------------------------------------
# main sweep
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exp_dir', required=True, help='Dir containing checkpoint-* folders.')
    ap.add_argument('--val_config', required=True, help='YAML describing validation datasets.')
    ap.add_argument('--recon_dir', default='/tmp/select_recon', help='Where to write reconstructed frames.')
    ap.add_argument('--cache', default=None, help='Metrics cache JSON (default: <exp_dir>/select_cache.json).')
    ap.add_argument('--num_inference_steps', type=int, default=50)
    ap.add_argument('--start_step', type=int, default=None, help='Skip checkpoints below this step.')
    ap.add_argument('--max_step', type=int, default=None, help='Skip checkpoints above this step.')
    ap.add_argument('--patience', type=int, default=3, help='Stop after N evals with no target improvement.')
    ap.add_argument('--min_delta', type=float, default=0.002, help='Min target-score drop counted as improvement.')
    ap.add_argument('--guardrail_tol', type=float, default=0.03,
                    help='Max allowed guardrail (REDS4) regression vs reference, e.g. 0.03 = +3%%.')
    ap.add_argument('--watch', action='store_true', help='Poll exp_dir for new checkpoints instead of exiting.')
    ap.add_argument('--watch_interval', type=int, default=300, help='Polling seconds in --watch mode.')
    args = ap.parse_args()

    cfg = OmegaConf.load(args.val_config)
    datasets = cfg['datasets']
    ds_roles = {k: v.get('role', 'target') for k, v in datasets.items()}
    ds_weights = {k: float(v.get('weight', 1.0)) for k, v in datasets.items()}
    metric_weights = {m: float(OmegaConf.select(cfg, f'metric_weights.{m}') or 1.0) for m in OBJECTIVE_METRICS}

    cache_path = args.cache or os.path.join(args.exp_dir, 'select_cache.json')
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    device = torch.device('cuda')
    bank = MetricBank(device)

    def evaluate_checkpoint(step, cn_dir):
        key = str(step)
        cache.setdefault(key, {})
        need = [ds for ds in datasets if ds not in cache[key]]
        if need:
            pipeline = build_pipeline(cn_dir, device)
            for ds in need:
                d = datasets[ds]
                seqs = d.get('seqs') or sorted(os.listdir(d['lr']))
                seq_accs = []
                for seq in seqs:
                    out_seq = os.path.join(args.recon_dir, key, ds, seq)
                    run_inference_sequence(pipeline, bank.of_model,
                                           os.path.join(d['lr'], seq), out_seq,
                                           args.num_inference_steps)
                    seq_accs.append(bank.eval_sequence(out_seq, os.path.join(d['gt'], seq)))
                cache[key][ds] = aggregate(seq_accs)
                json.dump(cache, open(cache_path, 'w'), indent=2)
            del pipeline
            gc.collect()
            torch.cuda.empty_cache()
        return {ds: cache[key][ds] for ds in datasets}

    def in_range(step):
        return (args.start_step is None or step >= args.start_step) and \
               (args.max_step is None or step <= args.max_step)

    print(f"[select] cache: {cache_path}")
    print(f"[select] objective metrics: {OBJECTIVE_METRICS}  weights: {metric_weights}")
    print(f"[select] guardrail tolerance: +{args.guardrail_tol:.0%}, patience: {args.patience}\n")

    reference = None
    best = {'step': None, 'target': float('inf')}
    no_improve = 0
    processed = set()

    while True:
        ckpts = [(s, d) for s, d in discover_checkpoints(args.exp_dir) if in_range(s)]
        new = [(s, d) for s, d in ckpts if s not in processed]

        for step, cn_dir in new:
            pdm = evaluate_checkpoint(step, cn_dir)
            processed.add(step)
            if reference is None:
                reference = pdm  # first checkpoint is the relative baseline
                ref_step = step
                print(f"checkpoint-{step}: REFERENCE (all scores relative to this)\n")
                continue

            target, guardrail = composite(pdm, reference, ds_roles, ds_weights, metric_weights)
            valid = (not np.isnan(guardrail)) and (guardrail <= 1.0 + args.guardrail_tol)
            flag = 'OK' if valid else f'REJECT(guardrail {guardrail:.3f} > {1 + args.guardrail_tol:.3f})'
            improved = valid and (target < best['target'] - args.min_delta)

            print(f"checkpoint-{step}: target={target:.4f}  guardrail={guardrail:.4f}  [{flag}]"
                  + ("  <-- new best" if improved else ""))

            if improved:
                best = {'step': step, 'target': target, 'guardrail': guardrail}
                no_improve = 0
            elif valid:
                no_improve += 1

            if no_improve >= args.patience:
                print(f"\n[select] EARLY STOP: no target improvement for {args.patience} evals.")
                break
        else:
            if args.watch:
                print(f"[select] waiting {args.watch_interval}s for new checkpoints...")
                time.sleep(args.watch_interval)
                continue
            # exhausted all checkpoints without triggering patience
        break

    print("\n================= RECOMMENDATION =================")
    if best['step'] is None:
        print("No checkpoint beat the reference within the guardrail. Reference is best so far.")
        print(f"  -> use checkpoint-{ref_step}")
    else:
        print(f"  -> use checkpoint-{best['step']}")
        print(f"     target score {best['target']:.4f} ({(1 - best['target']) * 100:+.1f}% vs reference)")
        print(f"     guardrail    {best['guardrail']:.4f}")
    print(f"\nFull metrics cached at: {cache_path}")


if __name__ == '__main__':
    main()
