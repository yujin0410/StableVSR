"""Dump train/val scalar curves from experiment TensorBoard logs.

Diagnostic for the tfl+Vimeo perceptual collapse: was it driven by the
DATA (metrics drift gradually as training sees Vimeo) or the LOSS
(metrics jump immediately from step ~0)? Compares the REDS dual-SFT
baseline against the Vimeo / mixed / tft variants without any retraining.

Reads each experiment's TensorBoard event file and prints the
start / mid / end value of every relevant scalar so the trajectory
direction is visible at a glance.

Usage:
    python ablation_tools/dump_tb_curves.py                 # default set
    python ablation_tools/dump_tb_curves.py <logdir> ...    # custom dirs
"""
import sys
import glob
import os

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    sys.exit("tensorboard not installed: pip install tensorboard")

DEFAULT = {
    "dualsft_REDS":   "experiments/20260430_dualsft/logs/train_controlnet",
    "reds_vimeo":     "experiments/20260602_reds_vimeo/logs/train_controlnet",
    "mixed_finetune": "experiments/20260602_mixed_finetune/logs/train_controlnet",
    "tft":            "experiments/20260602_tft/logs/train_controlnet",
    "tft_loss":       "experiments/20260602_tft_loss/logs/train_controlnet",
    "vimeo_from30k":  "experiments/20260602_vimeo_from30k/logs/train_controlnet",
    "loss_init":      "experiments/20260602_loss_init/logs/train_controlnet",
}

KEYS = ("musiq", "lpips", "clip", "niqe", "psnr", "ssim", "val", "loss")


def summarize(name, d):
    evs = sorted(glob.glob(os.path.join(d, "events.*")))
    if not evs:
        print(f"\n=== {name} ===  (no event files in {d})")
        return
    ea = EventAccumulator(evs[-1], size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    print(f"\n=== {name} ===  ({len(tags)} scalar tags)")
    if not tags:
        print("  (no scalars logged)")
        return
    pri = [t for t in tags if any(k in t.lower() for k in KEYS)]
    show = pri if pri else tags
    # val/perceptual tags first, then losses
    show = sorted(show, key=lambda t: (0 if any(k in t.lower() for k in
                  ("musiq", "lpips", "clip", "niqe", "val")) else 1, t))
    for tag in show:
        v = ea.Scalars(tag)
        if not v:
            continue
        steps = [s.step for s in v]
        vals = [s.value for s in v]
        n = len(vals)
        print(f"  {tag:30s} step {steps[0]:>6}->{steps[-1]:<7} "
              f"start={vals[0]:.4f}  mid={vals[n//2]:.4f}  end={vals[-1]:.4f}  (n={n})")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for d in sys.argv[1:]:
            summarize(d, d)
    else:
        for name, d in DEFAULT.items():
            summarize(name, d)
