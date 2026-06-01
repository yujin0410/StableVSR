# Lipschitz-anchored conditioning — empirical verification

`scripts/verify_lipschitz.py` tests the composed-bound theory used in the
defense narrative:

  **||Δx̂||_HR  ≤  L · K' · ||ΔLR||**

with three measured slopes:

| Slope | What it tests | Code path |
|-------|---------------|-----------|
| `K_input_DTCWT_magnitude` | Pillar B at the input (no encoder) | `dtcwt_magnitude_diff` |
| `K_input_FFT_magnitude`   | Pillar B counter-control          | `fft_magnitude_diff`   |
| `K_encoder_DTCWT_full`    | Pillar A (encoder Lipschitz)      | `encoder_cond_diff`    |
| `K_e2e_x_hat` (Phase 2)   | End-to-end bound                  | `hr_pair_diff`         |

All ΔLR values are motion-compensated via RAFT (LR_n − warp(LR_{n−1})).

## Phase 1 — fast, no inference (recommended first run)

```bash
python scripts/verify_lipschitz.py \
    --ckpt          /home/yjcho/StableVSR/experiments/20260430_dualsft/checkpoint-20000/ \
    --reds_lr_root  /mnt/HDD_raid1/yjcho/20260430/reds/ \
    --clips         000 011 015 020 \
    --max_frames    30 \
    --crop          128 \
    --output_dir    results_lipschitz/
```

Outputs:
- `results_lipschitz/lipschitz_records.json` — per-pair records + summary
- `results_lipschitz/lipschitz_scatter.png`  — 3-panel scatter

Expected runtime: ~10 min on one A100 for 4 clips × 30 frames.

## Phase 2 — end-to-end (slow, optional)

```bash
python scripts/verify_lipschitz.py \
    --ckpt /home/yjcho/StableVSR/experiments/20260430_dualsft/checkpoint-20000/ \
    --reds_lr_root /mnt/HDD_raid1/yjcho/20260430/reds/ \
    --clips 000 011 \
    --max_frames 10 \
    --run_inference \
    --num_inference_steps 50 \
    --pretrained_model stabilityai/stable-diffusion-2-1
```

For end-to-end, restrict to fewer clips/frames — each clip runs the full
StableVSR bidirectional sweep (≈ several minutes per clip on A100).

## How to interpret the result

### If the theory is **supported**

- All three Phase 1 scatters show a clear linear envelope (finite slope,
  not a cloud).
- `K_encoder_DTCWT_full` median is **finite and bounded** — typically
  on the order of `K_input_DTCWT_magnitude × (network gain)`.
- In Phase 2, the `K_e2e_x_hat` slope tracks `L_unet · K_encoder` within
  a factor of ~2–5.
- **Defense statement**: "The composed Lipschitz bound is empirically
  tight to within a constant factor; per-frame deterministic conditioning
  is the mechanism by which spatial improvements translate to temporal
  metric gains."

### If the theory is **partially supported**

- Phase 1 shows linear envelopes per transform but slopes do **not**
  rank as DT-CWT < FFT < Pixel < DWT (e.g. FFT slope is lower than
  DT-CWT, which matches the tLPIPS ablation result).
- **Defense statement**: "Pillar B (DT-CWT-specific shift invariance) is
  not the dominant contributor in our regime; Pillar A (Lipschitz
  anchoring by *any* deterministic encoder) carries the temporal gain.
  This is consistent with the cond_mode ablation, where FFT slightly
  beats DT-CWT on tLPIPS."

### If the theory is **refuted**

- Phase 1 scatter is a cloud with no linear envelope.
- `K_e2e_x_hat` does not track `K_encoder`.
- **Defense statement**: "The Lipschitz-anchored framing does not fit our
  data; the temporal mechanism must be (a) cascade amplification through
  ControlNet bidirectional propagation and/or (b) spectrum-locked weights
  imposed by the frequency-separated loss during training, neither of
  which is captured by a single-frame Lipschitz bound. We treat this as
  open theoretical work and report only the *empirical* temporal gain."

## Channel-count notes

`--inject_high` / `--inject_low` must match
`unet.config.block_out_channels[1] // 2` for the base model the
checkpoint was trained on:

| Base                                 | block_out_channels[1] | inject |
|--------------------------------------|-----------------------|--------|
| stabilityai/stable-diffusion-2-1     | 640                   | 320    |
| claudiom4sir/StableVSR (256-first)   | 512                   | 256    |

If `[load] missing=`/`unexpected=` in the load log are large, the
channel count is wrong.
