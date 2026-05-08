"""Video inference for StableVSR.

Accepts a single .mp4 file or a folder containing .mp4 files (e.g. REDS4 LR
sequences encoded as 000.mp4, 011.mp4, 015.mp4, 020.mp4) and produces 4x
super-resolved videos. Optionally also saves frames as PNGs in the layout
expected by eval.py:

    out_path/frames/<seq>/0000000.png
    out_path/<seq>.mp4

Requires `imageio` and `imageio-ffmpeg` (pip install imageio imageio-ffmpeg).
"""

from pipeline.stablevsr_pipeline import StableVSRPipeline
from diffusers import DDPMScheduler, ControlNetModel
from accelerate.utils import set_seed
from PIL import Image
import os
import argparse
import numpy as np
from pathlib import Path
import imageio.v2 as imageio
import torch
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights


def read_video_frames(path):
    reader = imageio.get_reader(str(path))
    meta = reader.get_meta_data()
    fps = meta.get('fps', 24)
    frames = [Image.fromarray(f) for f in reader]
    reader.close()
    return frames, fps


def write_video(path, frames, fps):
    writer = imageio.get_writer(str(path), fps=fps, codec='libx264',
                                quality=8, macro_block_size=1)
    for f in frames:
        writer.append_data(np.array(f))
    writer.close()


def list_videos(in_path):
    p = Path(in_path)
    if p.is_file():
        return [p]
    return sorted([q for q in p.iterdir() if q.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv'}])


def main():
    parser = argparse.ArgumentParser(description="Video inference for StableVSR.")
    parser.add_argument("--in_path", type=str, required=True,
                        help="Path to an mp4 file or a folder of mp4 files.")
    parser.add_argument("--out_path", type=str, default='./StableVSR_results/',
                        help="Output folder.")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--controlnet_ckpt", type=str, default=None,
                        help="Path to a local controlnet checkpoint folder. "
                             "If None, fetches claudiom4sir/StableVSR.")
    parser.add_argument("--fps", type=float, default=None,
                        help="Override output fps. Default: source fps.")
    parser.add_argument("--save_frames", action='store_true',
                        help="Also dump SR frames as PNGs in out_path/frames/<seq>/ "
                             "so eval.py can be run directly.")
    parser.add_argument("--no_video", action='store_true',
                        help="Skip writing the .mp4 (e.g. when you only need frames).")
    args = parser.parse_args()

    print("Run with arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    set_seed(42)
    device = torch.device('cuda')
    model_id = 'claudiom4sir/StableVSR'
    controlnet_model = ControlNetModel.from_pretrained(
        args.controlnet_ckpt if args.controlnet_ckpt is not None else model_id,
        subfolder='controlnet')
    pipeline = StableVSRPipeline.from_pretrained(model_id, controlnet=controlnet_model)
    pipeline.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder='scheduler')
    pipeline = pipeline.to(device)
    pipeline.enable_xformers_memory_efficient_attention()
    of_model = raft_large(weights=Raft_Large_Weights.DEFAULT)
    of_model.requires_grad_(False)
    of_model = of_model.to(device)

    videos = list_videos(args.in_path)
    if not videos:
        raise RuntimeError(f"No video files found at {args.in_path}")

    os.makedirs(args.out_path, exist_ok=True)

    for vid in videos:
        seq_name = vid.stem
        print(f"\n=== Processing {vid.name} (seq={seq_name}) ===")
        frames, src_fps = read_video_frames(vid)
        out_fps = args.fps if args.fps is not None else src_fps
        print(f"  frames: {len(frames)}, src fps: {src_fps}, "
              f"resolution: {frames[0].size[0]}x{frames[0].size[1]}")

        sr = pipeline('', frames,
                      num_inference_steps=args.num_inference_steps,
                      guidance_scale=0,
                      of_model=of_model).images
        sr = [f[0] for f in sr]

        if args.save_frames:
            frame_dir = Path(args.out_path) / 'frames' / seq_name
            frame_dir.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(sr):
                f.save(frame_dir / f"{i:08d}.png")
            print(f"  saved frames -> {frame_dir}")

        if not args.no_video:
            out_video = Path(args.out_path) / f"{seq_name}.mp4"
            write_video(out_video, sr, out_fps)
            print(f"  saved video  -> {out_video}")


if __name__ == '__main__':
    main()
