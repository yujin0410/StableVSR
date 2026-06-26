"""Disable Temporal Texture Guidance (TTG) at inference via an env var.

TTG = the StableVSR ControlNet path that injects the flow-warped
previous-frame prediction (`warped_prev_est`) as `controlnet_cond`.
The pipeline multiplies the ControlNet residuals by
`controlnet_conditioning_scale`; setting it to 0 makes those residuals
zero, which is equivalent to removing TTG (the U-Net then receives zero
additional residuals -- identical to the first-frame path).

This monkey-patches StableVSRPipeline.__call__ to force
`controlnet_conditioning_scale=0.0` whenever NO_TTG=1, so no edits to the
pipeline or to the test.py call site are needed beyond one import.

Usage
-----
    export NO_TTG=1
    # in test.py, after the pipeline import:
    import ablation_tools.no_ttg  # noqa: F401

With NO_TTG unset or 0, behaviour is unchanged (TTG on). Safe to leave
the import in permanently.
"""
import os

from pipeline.stablevsr_pipeline import StableVSRPipeline

_orig_call = StableVSRPipeline.__call__


def _call(self, *args, **kwargs):
    if os.environ.get("NO_TTG") == "1":
        kwargs["controlnet_conditioning_scale"] = 0.0
    return _orig_call(self, *args, **kwargs)


StableVSRPipeline.__call__ = _call
print("[no_ttg] StableVSRPipeline.__call__ patched "
      f"(NO_TTG={'1 -> TTG OFF' if os.environ.get('NO_TTG') == '1' else 'unset -> TTG ON'})")
