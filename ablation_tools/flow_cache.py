"""Flow-cache monkey-patch for StableVSRPipeline.

Optical flow in the pipeline depends ONLY on the (upscaled) input LR
frames, never on the SFT bands. So across the band-disabling variants
(full / no_high / no_low / no_both) the forward/backward flows are
identical -- but the stock pipeline recomputes them on every run, which
is the dominant wall-clock cost (~3h+ per run in practice).

This module wraps ``StableVSRPipeline.compute_flows`` with a content-
keyed disk cache. The first run (e.g. ``full``) computes and stores the
flows; the remaining variants load them. No edits to the pipeline file
are required.

Usage
-----
1. Set an env var pointing at a cache dir:
       export FLOW_CACHE_DIR=/path/to/flow_cache
2. Add ONE line near the top of your inference script (after the
   pipeline import), e.g. in test.py:
       import ablation_tools.flow_cache  # noqa: F401  (enables flow cache)
3. Run the variants as usual. With FLOW_CACHE_DIR unset, behaviour is
   identical to the stock pipeline (cache disabled), so it is safe to
   leave the import in permanently.

The cache key is a hash of (#frames, frame shape, rescale_factor) plus a
sparse content sample of the first/middle/last frame, so it is stable
across processes and collides correctly across the band variants while
distinguishing different sequences.
"""

import os
import hashlib

import torch

from pipeline.stablevsr_pipeline import StableVSRPipeline

_orig_compute_flows = StableVSRPipeline.compute_flows


def _cache_key(cache_dir, images, rescale_factor):
    h = hashlib.md5()
    h.update(f"{len(images)}_{tuple(images[0].shape)}_{rescale_factor}".encode())
    idxs = sorted({0, len(images) // 2, len(images) - 1})
    for idx in idxs:
        sample = images[idx].detach().float().flatten()[::1031].cpu().numpy()
        h.update(sample.tobytes())
    return os.path.join(cache_dir, h.hexdigest() + ".pt")


def _cached_compute_flows(self, of_model, images, rescale_factor=1):
    cache_dir = os.environ.get("FLOW_CACHE_DIR")
    if not cache_dir or len(images) < 2:
        return _orig_compute_flows(self, of_model, images, rescale_factor)

    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(cache_dir, images, rescale_factor)

    if os.path.exists(key):
        print(f"[flow_cache] hit  {key}")
        d = torch.load(key, map_location=images[0].device)
        return d["fwd"], d["bwd"]

    fwd, bwd = _orig_compute_flows(self, of_model, images, rescale_factor)
    try:
        torch.save({"fwd": fwd, "bwd": bwd}, key)
        print(f"[flow_cache] save {key}")
    except Exception as e:  # never let caching break inference
        print(f"[flow_cache] WARNING: failed to save {key}: {e}")
    return fwd, bwd


StableVSRPipeline.compute_flows = _cached_compute_flows
print("[flow_cache] StableVSRPipeline.compute_flows patched "
      f"(FLOW_CACHE_DIR={'set' if os.environ.get('FLOW_CACHE_DIR') else 'unset -> disabled'})")
