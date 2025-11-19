#!/bin/bash

set -e  # stop if any command fails

OUTDIR="./testing_3000_steps/results"
SEED=1
SUBSCALE="1e-4"

echo "Running sweeps... results → $OUTDIR"
mkdir -p "$OUTDIR"

module load pytorch
# -------------------------
# BF16 on a100
# -------------------------
python sweep_lr_real_vendor_job.py \
    --vendor bf16_a100 \
    --precision bf16 \
    --lr 0.003 \
    --sub-scale $SUBSCALE \
    --seed $SEED \
    --max-steps 3000 \
    --output-dir $OUTDIR

python sweep_lr_real_vendor_job.py \
    --vendor bf16_a100 \
    --precision bf16 \
    --lr 0.001 \
    --sub-scale $SUBSCALE \
    --seed $SEED \
    --max-steps 3000 \
    --output-dir $OUTDIR

python sweep_lr_real_vendor_job.py \
    --vendor bf16_a100 \
    --precision bf16 \
    --lr 0.01 \
    --sub-scale $SUBSCALE \
    --seed $SEED \
    --max-steps 3000 \
    --output-dir $OUTDIR

# # -------------------------
# # FP16 on a100
# # -------------------------
python sweep_lr_real_vendor_job.py \
    --vendor fp16_a100 \
    --precision fp16 \
    --lr 0.003 \
    --sub-scale $SUBSCALE \
    --seed $SEED \
    --max-steps 3000 \
    --output-dir $OUTDIR

python sweep_lr_real_vendor_job.py \
    --vendor fp16_a100 \
    --precision fp16 \
    --lr 0.001 \
    --sub-scale $SUBSCALE \
    --seed $SEED \
    --max-steps 3000 \
    --output-dir $OUTDIR

python sweep_lr_real_vendor_job.py \
    --vendor fp16_a100 \
    --precision fp16 \
    --lr 0.01 \
    --sub-scale $SUBSCALE \
    --seed $SEED \
    --max-steps 3000 \
    --output-dir $OUTDIR

echo "All sweeps completed!"

