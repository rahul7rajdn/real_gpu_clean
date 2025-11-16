#!/bin/bash
#SBATCH --job-name=real_v100_fp16_lr3e-3_sub1e-3_s3
#SBATCH --output=real_gpu_results_20251116_122625/job_logs/real_v100_fp16_lr3e-3_sub1e-3_s3_%j.out
#SBATCH --error=real_gpu_results_20251116_122625/job_logs/real_v100_fp16_lr3e-3_sub1e-3_s3_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:v100:1

# Verify GPU
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Run training
cd "/home/hice1/rn84/GPU_HW_SW_8803/real_gpu_clean/real_gpu_clean"
python sweep_lr_real_vendor_job.py   --vendor "fp16_a100"   --precision "fp16"   --lr 3e-3   --sub-scale 1e-3   --seed 3   --max-steps 3000   --batch-size 32   --output-dir "real_gpu_results_20251116_122625/results"

