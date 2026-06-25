# ablation_tools

Additive helpers for the UDM10 (OOD) band-disabling diagnostic. **None of
these files modify existing repo code** — they are safe to pull alongside
your local dual-SFT working tree.

## Why
On REDS the HIGH-frequency SFT band drives temporal consistency
(slide-24 band toggle). OOD (Vid4/UDM10/SPMCS) the temporal metrics
collapse. This kit reruns the *same* band-disabling ablation on UDM10 to
decide the fix:

- **`no_high` barely changes UDM10 tLPIPS** → HIGH anchoring is inactive
  OOD → fix = generalize the anchoring (degradation diversity / less
  REDS overfit).
- **`no_high` worsens UDM10 tLPIPS a lot** (like REDS) → anchoring works
  but is insufficient → fix = strengthen it / inference-time band boost.

## Files
- `flow_cache.py` — monkey-patch that caches optical flow to disk. Flows
  depend only on the input, so the 4 band variants share them (≈3–4×
  speedup). Enable with one import + `FLOW_CACHE_DIR`.
- `eval_per_seq.py` — `eval.py` clone that also prints a per-sequence
  tLPIPS/tOF breakdown (systematic vs a few outlier clips). Same metric
  definitions, so numbers match the paper tables.
- `run_udm10_ablation.sh` — runs full / no_high / no_low / no_both on
  UDM10 (2-GPU sharded) then evaluates each.

## Use
1. Enable the flow cache — add to the top of `test.py` (after the
   pipeline import):
   ```python
   import ablation_tools.flow_cache  # noqa: F401
   ```
   (No effect unless `FLOW_CACHE_DIR` is set, so it is safe to leave in.)
2. Edit the 4 paths at the top of `run_udm10_ablation.sh`.
3. `bash ablation_tools/run_udm10_ablation.sh`
4. Send back `eval_udm10_full.txt` and `eval_udm10_no_high.txt`.
