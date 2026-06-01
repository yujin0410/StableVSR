"""Count parameters in arbitrary model checkpoints.

Generic counter that handles the formats VSR papers ship in:
  - diffusers directory  (model_index.json + subfolder/{unet,vae,...})
  - single .bin / .pth / .pt / .ckpt file (state_dict or full checkpoint)
  - single .safetensors file
  - directory of .bin / .safetensors files (sum across all)

Each entry given as NAME=PATH on the command line. Reports total params,
trainable params (if known), and on-disk size.

Usage
-----
    python scripts/count_params.py \\
        BasicVSR++=/path/to/basicvsr_plusplus.pth \\
        StableVSR=/path/to/StableVSR \\
        STAR=/path/to/star_ckpt \\
        DLoRAL=/path/to/dloral_ckpt \\
        VividVR=/path/to/vivid_ckpt \\
        FlashVSR=/path/to/flashvsr_ckpt \\
        Ours=/home/yjcho/StableVSR/experiments/20260430_dualsft/checkpoint-20000/

Outputs a markdown table to stdout and JSON to --output (default
param_counts.json).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


# ---------------------------------------------------------------------------
# State-dict loaders
# ---------------------------------------------------------------------------

def _load_torch(path: Path) -> Dict:
    """Load torch checkpoint, normalize to a flat state_dict."""
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=False)
    # Common wrapper shapes
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict", "model", "params", "params_ema"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        # Already a flat state_dict (tensor values)
        if any(hasattr(v, "shape") for v in obj.values()):
            return obj
    # Last resort: empty
    return {}


def _load_safetensors(path: Path) -> Dict:
    from safetensors.torch import load_file
    return load_file(str(path))


def _scan_files(root: Path) -> List[Path]:
    """Recursively scan for weight files, ignoring optimizer / scheduler dumps."""
    bad_substrings = ("optimizer", "scheduler", "rng", "random_states")
    bad_suffixes = (".json", ".txt", ".md", ".yaml", ".yml", ".log")
    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        name = p.name.lower()
        if any(s in name for s in bad_substrings):
            continue
        if name.endswith(bad_suffixes):
            continue
        if name.endswith((".bin", ".safetensors", ".pth", ".pt", ".ckpt")):
            out.append(p)
    return out


def _state_dict_from(path: Path) -> Dict:
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        return _load_safetensors(path)
    if suffix in (".bin", ".pth", ".pt", ".ckpt"):
        return _load_torch(path)
    return {}


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------

def count_path(path: Path) -> Tuple[int, int, List[Path]]:
    """Return (n_params, on_disk_bytes, files_used).

    For a directory, sums all weight files found recursively.
    """
    files: List[Path] = []
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = _scan_files(path)
    else:
        print(f"[warn] {path} not found", file=sys.stderr)
        return 0, 0, []

    total_params = 0
    total_bytes = 0
    keys_seen: set = set()

    for f in files:
        try:
            sd = _state_dict_from(f)
        except Exception as e:
            print(f"[warn] could not load {f}: {e}", file=sys.stderr)
            continue
        if not sd:
            continue
        total_bytes += f.stat().st_size
        for k, v in sd.items():
            # Avoid double-counting identical keys (e.g. when both .bin and
            # .safetensors are present for the same component).
            if k in keys_seen:
                continue
            if hasattr(v, "numel"):
                total_params += int(v.numel())
                keys_seen.add(k)
    return total_params, total_bytes, files


def fmt_M(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    return f"{n / 1e6:.2f}M"


def fmt_size(b: int) -> str:
    if b >= 1024 ** 3:
        return f"{b / 1024 ** 3:.2f} GB"
    if b >= 1024 ** 2:
        return f"{b / 1024 ** 2:.1f} MB"
    return f"{b / 1024:.1f} KB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "models", nargs="+",
        help="NAME=PATH entries. Paths may be files or directories.",
    )
    ap.add_argument("--output", default="param_counts.json")
    ap.add_argument("--verbose", action="store_true",
                    help="List every weight file scanned per model")
    args = ap.parse_args()

    entries = []
    for spec in args.models:
        if "=" not in spec:
            print(f"[err] bad spec {spec!r}, expected NAME=PATH", file=sys.stderr)
            continue
        name, _, raw = spec.partition("=")
        entries.append((name, Path(raw).expanduser().resolve()))

    print(f"\nCounting parameters in {len(entries)} models ...\n")
    results = []
    for name, path in entries:
        n, b, files = count_path(path)
        if args.verbose:
            print(f"[{name}] {len(files)} file(s):")
            for f in files:
                print(f"    {f}")
        results.append({
            "name": name,
            "path": str(path),
            "n_params": n,
            "params_human": fmt_M(n),
            "bytes": b,
            "size_human": fmt_size(b),
            "n_files": len(files),
        })

    # Markdown table.
    print(f"{'Method':<20s}  {'Params':>10s}   {'Size':>10s}   files")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<20s}  {r['params_human']:>10s}   "
              f"{r['size_human']:>10s}   {r['n_files']}")
    print()

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[save] {args.output}")


if __name__ == "__main__":
    main()
