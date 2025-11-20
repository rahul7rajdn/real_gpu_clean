#!/bin/bash

set -e  # stop if any command fails

# -----------------------------
# Read max steps from command line
# Default = 3000
# -----------------------------
MAX_STEPS=${1:-3000}

OUTDIR="./test_results_${MAX_STEPS}_steps/results"
SEED=1
SUBSCALE="1e-4"

echo "Running sweeps..... max_steps=${MAX_STEPS}, results ==> $OUTDIR"
mkdir -p "$OUTDIR"

source /tmp/pytorch_cuda_env/bin/activate 
which python

nvidia-smi -L

echo "*****************************************************************"

# module load pytorch
# -------------------------
# BF16 on A100
# -------------------------

python sweep_lr_real_vendor_job.py \
    --vendor bf16_a100 \
    --precision bf16 \
    --lr 0.003 \
    --sub-scale $SUBSCALE \
    --seed $SEED \
    --max-steps $MAX_STEPS \
    --output-dir $OUTDIR

python sweep_lr_real_vendor_job.py \
    --vendor bf16_a100 \
    --precision bf16 \
    --lr 0.001 \
    --sub-scale $SUBSCALE \
    --seed $SEED \
    --max-steps $MAX_STEPS \
    --output-dir $OUTDIR

python sweep_lr_real_vendor_job.py \
    --vendor bf16_a100 \
    --precision bf16 \
    --lr 0.01 \
    --sub-scale $SUBSCALE \
    --seed $SEED \
    --max-steps $MAX_STEPS \
    --output-dir $OUTDIR

# -------------------------
# FP16 on A100
# -------------------------
python sweep_lr_real_vendor_job.py \
    --vendor fp16_a100 \
    --precision fp16 \
    --lr 0.003 \
    --sub-scale $SUBSCALE \
    --seed $SEED \
    --max-steps $MAX_STEPS \
    --output-dir $OUTDIR

python sweep_lr_real_vendor_job.py \
    --vendor fp16_a100 \
    --precision fp16 \
    --lr 0.001 \
    --sub-scale $SUBSCALE \
    --seed $SEED \
    --max-steps $MAX_STEPS \
    --output-dir $OUTDIR

python sweep_lr_real_vendor_job.py \
    --vendor fp16_a100 \
    --precision fp16 \
    --lr 0.01 \
    --sub-scale $SUBSCALE \
    --seed $SEED \
    --max-steps $MAX_STEPS \
    --output-dir $OUTDIR

echo "================All sweeps completed on A100!======================"
