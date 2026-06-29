"""Override the diffusion sampling seed at inference via an env var.

For the seed-variation experiment: generate the SAME frames K times with
different initial noise to measure how much the output detail depends on the
random seed. Strong spatial conditioning (anchoring) should make the output
nearly seed-invariant; weak conditioning lets the seed move the detail around.

Monkey-patches StableVSRPipeline.__call__ to inject a fresh
generator seeded by GEN_SEED, so no edits to test.py are needed beyond one
import. With GEN_SEED unset, behaviour is unchanged.

Usage:
    export GEN_SEED=0           # then 1, 2, 3, 4 for the other runs
    # in test.py, after the pipeline import:
    import ablation_tools.seed_override  # noqa: F401
"""
import os

import torch

from pipeline.stablevsr_pipeline import StableVSRPipeline

_orig_call = StableVSRPipeline.__call__


def _call(self, *args, **kwargs):
    s = os.environ.get("GEN_SEED")
    if s is not None:
        kwargs["generator"] = torch.Generator(
            device=self._execution_device).manual_seed(int(s))
    return _orig_call(self, *args, **kwargs)


StableVSRPipeline.__call__ = _call
print("[seed_override] StableVSRPipeline.__call__ patched "
      f"(GEN_SEED={os.environ.get('GEN_SEED', 'unset')})")
