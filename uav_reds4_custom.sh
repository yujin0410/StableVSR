#!/usr/bin/env bash
# Upscale-A-Video custom inference wrapper.
#
# UAV writes PNGs to:
#   <out>/frame/<video_name>_n<n>_g<g>_s<s>{prop}{suffix}/<frame>.png
# We loop per sequence and consolidate into our comparison root:
#   /mnt/HDD_raid1/yjcho/Comparison/Upscale-A-Video/<DATASET>/<seq>/<frame>.png
#
# Drop into the Upscale-A-Video repo root and run:
#   conda activate UAV
#   bash uav_reds4_custom.sh
#
# Args (env vars):
#   DATASET=reds4|vid4   (default: reds4)
#   N_STEP=30            (inference_steps)
#   NOISE=120            (noise_level)
#   GS=6                 (guidance_scale)
#   PROP="24,26,28"      (propagation steps; "" to disable)
#   NO_LLAVA=1           (skip LLaVA prompting)
set -e

DATASET="${DATASET:-reds4}"
N_STEP="${N_STEP:-30}"
NOISE="${NOISE:-120}"
GS="${GS:-6}"
PROP="${PROP:-24,26,28}"
NO_LLAVA="${NO_LLAVA:-1}"

case "$DATASET" in
  reds4)
    LR_ROOT="/mnt/HDD_raid1/yjcho/data/REDS/test/bicubic"
    SEQS="000 011 015 020"
    DST_ROOT="/mnt/HDD_raid1/yjcho/Comparison/Upscale-A-Video/REDS"
    ;;
  vid4)
    LR_ROOT="/mnt/HDD_raid1/yjcho/data/Vid4/BIx4"
    SEQS="calendar city foliage walk"
    DST_ROOT="/mnt/HDD_raid1/yjcho/Comparison/Upscale-A-Video/Vid4"
    ;;
  *) echo "unknown DATASET=$DATASET"; exit 1 ;;
esac

EXTRA=""
[ -n "$PROP" ]   && EXTRA="$EXTRA -p $PROP"
[ "$NO_LLAVA" = "1" ] && EXTRA="$EXTRA --no_llava"

mkdir -p "$DST_ROOT"
TMP_OUT=$(mktemp -d)
echo "[*] Temp out: $TMP_OUT"
echo "[*] Dst    : $DST_ROOT"

for seq in $SEQS; do
  SRC="$LR_ROOT/$seq"
  echo "[*] === seq $seq ==="
  python inference_upscale_a_video.py \
    -i "$SRC" \
    -o "$TMP_OUT/$seq" \
    --save_image \
    -n "$NOISE" -g "$GS" -s "$N_STEP" \
    $EXTRA

  # find produced frame folder (single subdir under <tmp>/<seq>/frame/)
  PROD=$(find "$TMP_OUT/$seq/frame" -mindepth 1 -maxdepth 1 -type d | head -n1)
  if [ -z "$PROD" ]; then
    echo "[!] No output frames found for $seq under $TMP_OUT/$seq/frame"
    exit 1
  fi
  echo "    produced: $PROD"
  rm -rf "$DST_ROOT/$seq"
  mkdir -p "$DST_ROOT/$seq"
  cp "$PROD"/*.png "$DST_ROOT/$seq/"
  echo "    -> $DST_ROOT/$seq ($(ls "$DST_ROOT/$seq" | wc -l) frames)"
done

echo "[+] Done. Eval with:"
echo "    cd /home/user/StableVSR && python eval.py \\"
echo "      --out_path $DST_ROOT/ --gt_path /mnt/HDD_raid1/yjcho/data/REDS/test/gt/"
