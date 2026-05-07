#!/usr/bin/env bash
# Upscale-A-Video custom inference wrapper (multi-server friendly).
#
# UAV writes PNGs to:
#   <out>/frame/<video_name>_n<n>_g<g>_s<s>{prop}{suffix}/<frame>.png
# This wrapper consolidates each sequence into:
#   $DST_ROOT/<DATASET-uppercase>/<seq>/<frame>.png
#
# Drop into the Upscale-A-Video repo root and run:
#   conda activate UAV
#   bash uav_reds4_custom.sh
#
# Env vars:
#   DATASET   reds4|vid4   (default: reds4)
#   LR_ROOT   override LR data root
#   DST_ROOT  override comparison output root (default per host below)
#   SEQS      space-separated sequence ids; overrides default
#   SHARD     "id/total" e.g. "0/2" or "1/2" — split sequences across servers
#   N_STEP    inference_steps (default: 30)
#   NOISE     noise_level     (default: 120)
#   GS        guidance_scale  (default: 6)
#   PROP      propagation     (default: "24,26,28"; "" to disable)
#   NO_LLAVA  1=skip LLaVA    (default: 1)
#
# Example — split REDS4 over 2 servers:
#   server A:  SHARD=0/2 DATASET=reds4 bash uav_reds4_custom.sh
#   server B:  SHARD=1/2 DATASET=reds4 bash uav_reds4_custom.sh
set -e

DATASET="${DATASET:-reds4}"
N_STEP="${N_STEP:-30}"
NOISE="${NOISE:-120}"
GS="${GS:-6}"
PROP="${PROP:-24,26,28}"
NO_LLAVA="${NO_LLAVA:-1}"

# Default roots — choose by which mount actually exists on this host.
if   [ -d /mnt/data/dataset ];     then HOST_PRESET=3090ti
elif [ -d /mnt/HDD_raid1/yjcho ];  then HOST_PRESET=main
else                                    HOST_PRESET=unknown
fi
echo "[*] Detected host preset: $HOST_PRESET"

case "$DATASET" in
  reds4)
    DEFAULT_SEQS="000 011 015 020"
    DEFAULT_GT="$([ "$HOST_PRESET" = "3090ti" ] && echo /mnt/data/dataset/REDS/test/gt || echo /mnt/HDD_raid1/yjcho/data/REDS/test/gt)"
    DEFAULT_LR="$([ "$HOST_PRESET" = "3090ti" ] && echo /mnt/data/dataset/REDS/test/bicubic || echo /mnt/HDD_raid1/yjcho/data/REDS/test/bicubic)"
    DST_SUB="REDS"
    ;;
  vid4)
    DEFAULT_SEQS="calendar city foliage walk"
    DEFAULT_GT="$([ "$HOST_PRESET" = "3090ti" ] && echo /mnt/data/dataset/Vid4/GT || echo /mnt/HDD_raid1/yjcho/data/Vid4/GT)"
    DEFAULT_LR="$([ "$HOST_PRESET" = "3090ti" ] && echo /mnt/data/dataset/Vid4/BIx4 || echo /mnt/HDD_raid1/yjcho/data/Vid4/BIx4)"
    DST_SUB="Vid4"
    ;;
  *) echo "unknown DATASET=$DATASET"; exit 1 ;;
esac

DEFAULT_DST="$([ "$HOST_PRESET" = "3090ti" ] \
  && echo /mnt/data/dataset/Comparison/Upscale-A-Video \
  || echo /mnt/HDD_raid1/yjcho/Comparison/Upscale-A-Video)"

LR_ROOT="${LR_ROOT:-$DEFAULT_LR}"
DST_ROOT="${DST_ROOT:-$DEFAULT_DST}/$DST_SUB"
SEQS="${SEQS:-$DEFAULT_SEQS}"

# Apply SHARD if given (id/total)
if [ -n "$SHARD" ]; then
  ID="${SHARD%/*}"
  TOT="${SHARD#*/}"
  PICKED=""
  i=0
  for s in $SEQS; do
    if [ "$((i % TOT))" = "$ID" ]; then PICKED="$PICKED $s"; fi
    i=$((i + 1))
  done
  SEQS="$(echo $PICKED | xargs)"
  echo "[*] Shard $ID/$TOT -> SEQS='$SEQS'"
fi

if [ ! -d "$LR_ROOT" ]; then echo "[!] LR_ROOT not found: $LR_ROOT"; exit 1; fi

EXTRA=""
[ -n "$PROP" ]        && EXTRA="$EXTRA -p $PROP"
[ "$NO_LLAVA" = "1" ] && EXTRA="$EXTRA --no_llava"

mkdir -p "$DST_ROOT"
TMP_OUT=$(mktemp -d)
echo "[*] LR     : $LR_ROOT"
echo "[*] Dst    : $DST_ROOT"
echo "[*] SEQS   : $SEQS"
echo "[*] Tmp    : $TMP_OUT"

for seq in $SEQS; do
  SRC="$LR_ROOT/$seq"
  if [ ! -d "$SRC" ]; then echo "[!] missing seq dir: $SRC"; exit 1; fi
  echo "[*] === seq $seq ==="
  python inference_upscale_a_video.py \
    -i "$SRC" \
    -o "$TMP_OUT/$seq" \
    --save_image \
    -n "$NOISE" -g "$GS" -s "$N_STEP" \
    $EXTRA

  PROD=$(find "$TMP_OUT/$seq/frame" -mindepth 1 -maxdepth 1 -type d | head -n1)
  if [ -z "$PROD" ]; then
    echo "[!] No output frames found for $seq under $TMP_OUT/$seq/frame"
    exit 1
  fi
  rm -rf "$DST_ROOT/$seq"
  mkdir -p "$DST_ROOT/$seq"
  cp "$PROD"/*.png "$DST_ROOT/$seq/"
  echo "    -> $DST_ROOT/$seq ($(ls "$DST_ROOT/$seq" | wc -l) frames)"
done

echo "[+] Done."
