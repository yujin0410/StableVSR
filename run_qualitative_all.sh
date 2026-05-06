#!/bin/bash
# Run qualitative comparison for ALL sequences in each dataset.
# For SPMCS, automatically detects each sequence's first frame index.
#
# Usage:
#   bash run_qualitative_all.sh
#
# Outputs go to figures/qual_{dataset}/{seq}_f{N}.pdf (and .png).

set -e
mkdir -p figures

# Default crop and a representative frame index per dataset
# (you can edit these per-sequence later for the best visual)

CROP="600 350 120 120"

############### Vid4 ###############
echo "===== Vid4 ====="
for seq in calendar city foliage walk; do
  # Vid4 frames typically 0..40, use middle
  frame=20
  out="figures/qual_vid4/${seq}_f${frame}.pdf"
  python make_qualitative_multidataset.py --dataset vid4 --seq "$seq" \
    --frame $frame --crop $CROP --output "$out"
done

############### UDM10 ###############
echo "===== UDM10 ====="
for seq in 000 001 002 003 004 005 006 007 008 009; do
  frame=10
  out="figures/qual_udm10/${seq}_f${frame}.pdf"
  python make_qualitative_multidataset.py --dataset udm10 --seq "$seq" \
    --frame $frame --crop $CROP --output "$out"
done

############### SPMCS ###############
# SPMCS has variable per-sequence start frames; auto-detect first frame.
echo "===== SPMCS ====="
for seq in $(ls /mnt/HDD_raid1/yjcho/data/SPMCS_BIx4/ | sort); do
  first=$(ls /mnt/HDD_raid1/yjcho/data/SPMCS_BIx4/$seq | sort | head -1)
  start=$(echo "$first" | sed 's/\..*//' | awk '{print $0+0}')
  # pick a frame ~10 in the sequence
  frame=$((start + 10))
  out="figures/qual_spmcs/${seq}_f${frame}.pdf"
  python make_qualitative_multidataset.py --dataset spmcs --seq "$seq" \
    --frame $frame --crop $CROP --output "$out"
done

echo "===== ALL DONE ====="
echo "Vid4:  $(ls figures/qual_vid4/  2>/dev/null | grep -c .pdf) figures"
echo "UDM10: $(ls figures/qual_udm10/ 2>/dev/null | grep -c .pdf) figures"
echo "SPMCS: $(ls figures/qual_spmcs/ 2>/dev/null | grep -c .pdf) figures"
