#!/bin/bash
# Run qualitative comparison for ALL FRAMES of ONE chosen sequence per dataset.
#
# Usage:
#   bash run_qualitative_one_seq.sh
#
# Datasets and their target sequences:
#   - Vid4   -> calendar (frames 0..40)
#   - UDM10  -> 000      (frames 0..31)
#   - SPMCS  -> AMVTG_004 (auto-detected frame range)
#
# Outputs go to figures/seq_{dataset}_{seq}/qual_f{N}.pdf (and .png).

set -e
mkdir -p figures

CROP="600 350 120 120"

############### Vid4: calendar (all frames) ###############
echo "===== Vid4 / calendar ====="
DATASET=vid4
SEQ=calendar
mkdir -p "figures/seq_${DATASET}_${SEQ}"
LR_DIR="/mnt/HDD_raid1/yjcho/data/Vid4/BIx4/${SEQ}"
TOTAL=$(ls $LR_DIR | wc -l)
echo "Total frames: $TOTAL"
for f in $(ls $LR_DIR | sort); do
  idx=$(echo "$f" | sed 's/\..*//' | awk '{print $0+0}')
  out="figures/seq_${DATASET}_${SEQ}/qual_f$(printf "%03d" $idx).pdf"
  python make_qualitative_multidataset.py --dataset $DATASET --seq $SEQ \
    --frame $idx --crop $CROP --output "$out"
done

############### UDM10: 000 (all frames) ###############
echo "===== UDM10 / 000 ====="
DATASET=udm10
SEQ=000
mkdir -p "figures/seq_${DATASET}_${SEQ}"
LR_DIR="/mnt/HDD_raid1/yjcho/data/UDM10/BIx4/${SEQ}"
TOTAL=$(ls $LR_DIR | wc -l)
echo "Total frames: $TOTAL"
for f in $(ls $LR_DIR | sort); do
  idx=$(echo "$f" | sed 's/\..*//' | awk '{print $0+0}')
  out="figures/seq_${DATASET}_${SEQ}/qual_f$(printf "%03d" $idx).pdf"
  python make_qualitative_multidataset.py --dataset $DATASET --seq $SEQ \
    --frame $idx --crop $CROP --output "$out"
done

############### SPMCS: AMVTG_004 (all frames) ###############
echo "===== SPMCS / AMVTG_004 ====="
DATASET=spmcs
SEQ=AMVTG_004
mkdir -p "figures/seq_${DATASET}_${SEQ}"
LR_DIR="/mnt/HDD_raid1/yjcho/data/SPMCS_BIx4/${SEQ}"
TOTAL=$(ls $LR_DIR | wc -l)
echo "Total frames: $TOTAL"
for f in $(ls $LR_DIR | sort); do
  idx=$(echo "$f" | sed 's/\..*//' | awk '{print $0+0}')
  out="figures/seq_${DATASET}_${SEQ}/qual_f$(printf "%04d" $idx).pdf"
  python make_qualitative_multidataset.py --dataset $DATASET --seq $SEQ \
    --frame $idx --crop $CROP --output "$out"
done

echo
echo "===== ALL DONE ====="
for d in vid4_calendar udm10_000 spmcs_AMVTG_004; do
  echo "$d: $(ls figures/seq_$d/ 2>/dev/null | grep -c .pdf) figures"
done
