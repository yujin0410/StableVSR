from pipeline.stablevsr_pipeline import StableVSRPipeline
from diffusers import DDPMScheduler, ControlNetModel
from accelerate.utils import set_seed
from PIL import Image
import os
import argparse
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from pathlib import Path
import torch

from util.sft_utils import (
    SFTAdapter,
    UNetWithSFT,
    FrequencyConditioningEncoder,
    UNetWithDualSFT,
)
from util.frequency_utils import DTCWTForward


def center_crop(im, size=128):
    width, height = im.size   # Get dimensions
    left = (width - size)/2
    top = (height - size)/2
    right = (width + size)/2
    bottom = (height + size)/2
    return im.crop((left, top, right, bottom))


# get arguments
parser = argparse.ArgumentParser(description="Test code for StableVSR.")
parser.add_argument("--out_path", default='./StableVSR_results/', type=str, help="Path to output folder.")
parser.add_argument("--in_path", type=str, required=True, help="Path to input folder (containing sets of LR images).")
parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of sampling steps")
parser.add_argument(
    "--controlnet_ckpt",
    type=str,
    default=None,
    help="Path to the directory that contains a 'controlnet' subfolder (e.g. an "
         "experiments/.../checkpoint-N folder). Defaults to the base StableVSR repo.",
)
parser.add_argument(
    "--sft_ckpt",
    type=str,
    default=None,
    help="Optional path to sft_adapter.bin. When provided, the SFT adapter is "
         "loaded and DT-CWT frequency conditioning is enabled in the pipeline.",
)
parser.add_argument(
    "--dual_sft",
    action="store_true",
    help="Use the frequency-separated dual-SFT design "
         "(FrequencyConditioningEncoder + UNetWithDualSFT, DT-CWT J=4) "
         "matching --dual_sft training. Requires --sft_ckpt.",
)
parser.add_argument(
    "--num_shards",
    type=int,
    default=1,
    help="Total number of shards to split the input sequences across. "
         "Combine with --shard_id to run on multiple GPUs in parallel.",
)
parser.add_argument(
    "--shard_id",
    type=int,
    default=0,
    help="0-indexed shard id for this process; processes "
         "seqs[shard_id::num_shards].",
)
parser.add_argument(
    "--cond_mode",
    type=str,
    default="dtcwt",
    choices=["dtcwt", "pixel", "fft", "dwt"],
    help="Conditioning input mode (must match training). 'pixel' / 'fft' / "
         "'dwt' attach the corresponding conditioner as child of the encoder "
         "so the saved sft_adapter.bin checkpoint loads cleanly.",
)
parser.add_argument(
    "--no_high_sft",
    action="store_true",
    help="Ablation: replace high_gamma with ones and high_beta with zeros "
         "after the encoder forward, so up_blocks[1] passes through unchanged. "
         "Requires --dual_sft.",
)
parser.add_argument(
    "--no_low_sft",
    action="store_true",
    help="Ablation: replace low_gamma with ones and low_beta with zeros "
         "after the encoder forward, so down_blocks[1] passes through unchanged. "
         "Requires --dual_sft.",
)
args = parser.parse_args()

print("Run with arguments:")
for arg, value in vars(args).items():
    print(f"  {arg}: {value}")

# set parameters
set_seed(42)
device = torch.device('cuda')
model_id = 'claudiom4sir/StableVSR'

controlnet_root = args.controlnet_ckpt if args.controlnet_ckpt is not None else model_id
controlnet_model = ControlNetModel.from_pretrained(controlnet_root, subfolder='controlnet')

pipeline = StableVSRPipeline.from_pretrained(model_id, controlnet=controlnet_model)
scheduler = DDPMScheduler.from_pretrained(model_id, subfolder='scheduler')
pipeline.scheduler = scheduler
pipeline = pipeline.to(device)
pipeline.enable_xformers_memory_efficient_attention()

of_model = raft_large(weights=Raft_Large_Weights.DEFAULT)
of_model.requires_grad_(False)
of_model = of_model.to(device)

# SFT + DT-CWT (optional, enabled when --sft_ckpt is given)
unet_with_sft = None
dtcwt_model = None
if args.sft_ckpt is not None:
    if not os.path.isfile(args.sft_ckpt):
        raise FileNotFoundError(f"--sft_ckpt does not exist: {args.sft_ckpt}")

    if args.dual_sft:
        inject_channels = pipeline.unet.config.block_out_channels[1]
        per_level = inject_channels // 2
        print(f"  Dual-SFT: inject channels = {inject_channels}, per-level = {per_level}")
        sft_adapter = FrequencyConditioningEncoder(
            in_channels=18, mid_channels=64,
            sft_out_channels_high=per_level,
            sft_out_channels_low=per_level,
            high_target=None, low_target=None,
        )
        if args.cond_mode == "pixel":
            from util.sft_utils import PixelPyramidConditioner
            sft_adapter.pixel_cond_model = PixelPyramidConditioner(
                in_channels=3, num_subbands=6, num_levels=4,
            )
            print("  --cond_mode pixel: PixelPyramidConditioner attached "
                  "(DT-CWT bypassed in conditioning path).")
        elif args.cond_mode == "fft":
            from util.sft_utils import FFTPyramidConditioner
            sft_adapter.pixel_cond_model = FFTPyramidConditioner(
                num_subbands=6, num_levels=4,
            )
            print("  --cond_mode fft: FFTPyramidConditioner attached "
                  "(DT-CWT bypassed; fixed FFT directional decomposition).")
        elif args.cond_mode == "dwt":
            from util.sft_utils import DWTPyramidConditioner
            sft_adapter.pixel_cond_model = DWTPyramidConditioner(
                num_levels=4, wave="db4", mode="zero",
            )
            print("  --cond_mode dwt: DWTPyramidConditioner attached "
                  "(DT-CWT bypassed; real DWT db4, 3 subbands -> 6 zero-pad).")
        sft_adapter.load_state_dict(torch.load(args.sft_ckpt, map_location='cpu'))
        sft_adapter = sft_adapter.to(device)
        sft_adapter.eval()
        sft_adapter.requires_grad_(False)

        if args.no_high_sft or args.no_low_sft:
            base_encoder = sft_adapter

            class AblatedEncoder(torch.nn.Module):
                def __init__(self, base, no_high, no_low):
                    super().__init__()
                    self.base = base
                    self.no_high = no_high
                    self.no_low = no_low

                def forward(self, yh, yl=None):
                    out = self.base(yh, yl)
                    if self.no_high:
                        out['high_gamma'] = torch.ones_like(out['high_gamma'])
                        out['high_beta'] = torch.zeros_like(out['high_beta'])
                    if self.no_low:
                        out['low_gamma'] = torch.ones_like(out['low_gamma'])
                        out['low_beta'] = torch.zeros_like(out['low_beta'])
                    return out

            sft_adapter = AblatedEncoder(base_encoder, args.no_high_sft, args.no_low_sft).to(device)
            sft_adapter.eval()
            sft_adapter.requires_grad_(False)
            print(f"  Ablation: no_high_sft={args.no_high_sft}, no_low_sft={args.no_low_sft}")

        unet_with_sft = UNetWithDualSFT(pipeline.unet)
        unet_with_sft.cond_encoder = sft_adapter

        dtcwt_model = DTCWTForward(J=4, biort='near_sym_a', qshift='qshift_a').to(device)
        dtcwt_model.requires_grad_(False)
        print(f"  Loaded dual-SFT FrequencyConditioningEncoder from {args.sft_ckpt}")
    else:
        feature_channels = pipeline.unet.config.block_out_channels[0]
        print(f"  SFT feature_channels = {feature_channels} (from unet.config.block_out_channels[0])")

        sft_adapter = SFTAdapter(cond_channels=36, feature_channels=feature_channels)
        sft_adapter.load_state_dict(torch.load(args.sft_ckpt, map_location='cpu'))
        sft_adapter = sft_adapter.to(device)
        sft_adapter.eval()
        sft_adapter.requires_grad_(False)

        # Wrap pipeline.unet so the forward hook on up_blocks[3] becomes active.
        unet_with_sft = UNetWithSFT(pipeline.unet, sft_adapter)

        dtcwt_model = DTCWTForward(J=3, biort='near_sym_a', qshift='qshift_a').to(device)
        dtcwt_model.requires_grad_(False)
        print(f"  Loaded SFT adapter from {args.sft_ckpt}")
else:
    if args.dual_sft:
        raise ValueError("--dual_sft requires --sft_ckpt to load the trained encoder.")
    print("  SFT disabled (no --sft_ckpt provided); inference uses ControlNet only.")

# iterate for every video sequence in the input folder
seqs = sorted(os.listdir(args.in_path))
if args.num_shards > 1:
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError(f"--shard_id must be in [0, {args.num_shards}); got {args.shard_id}")
    total = len(seqs)
    seqs = seqs[args.shard_id::args.num_shards]
    print(f"  Sharding: shard {args.shard_id}/{args.num_shards} -> {len(seqs)}/{total} sequences")
for seq in seqs:
    frame_names = sorted(os.listdir(os.path.join(args.in_path, seq)))
    frames = []
    for frame_name in frame_names:
        frame = Path(os.path.join(args.in_path, seq, frame_name))
        frame = Image.open(frame)
        # frame = center_crop(frame)
        frames.append(frame)

    # upscale frames using StableVSR (with optional SFT/DTCWT conditioning)
    frames = pipeline(
        '',
        frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=0,
        of_model=of_model,
        dtcwt_model=dtcwt_model,
        unet_with_sft=unet_with_sft,
    ).images
    frames = [frame[0] for frame in frames]

    # save upscaled sequences
    seq = Path(seq)
    target_path = os.path.join(args.out_path, seq.parent.name, seq.name)
    os.makedirs(target_path, exist_ok=True)
    for frame, name in zip(frames, frame_names):
        frame.save(os.path.join(target_path, name))
