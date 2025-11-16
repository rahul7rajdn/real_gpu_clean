# Real GPU Training Experiments

Minimal codebase for running floating-point format experiments on real GPU hardware (NVIDIA and AMD).

## Files

- `sweep_lr_real_vendor_job.py` - Main training script using PyTorch AMP
- `real_tiny_gpt_model.py` - Tiny GPT model definition
- `merge_real_gpu_results.py` - Script to aggregate results from all jobs
- `submit_real_gpu_batch.sh` - SLURM batch job submission script
- `README.md` - This file

## Requirements

- PyTorch with CUDA support
- SLURM workload manager
- Access to GPU nodes

## Quick Start

1. **Configure for your cluster:**
   Edit `submit_real_gpu_batch.sh` and adjust:
   - `PARTITION` - Your cluster's GPU partition
   - `QOS` - Your account's QOS
   - `GPU_TYPES` - Which GPUs to test (e.g., `("v100" "a100" "mi210")`)

2. **Submit jobs:**
   ```bash
   chmod +x submit_real_gpu_batch.sh
   ./submit_real_gpu_batch.sh
   ```

3. **Monitor jobs:**
   ```bash
   squeue -u $USER
   ```

4. **Merge results after completion:**
   ```bash
   python merge_real_gpu_results.py
   # Or specify directory:
   python merge_real_gpu_results.py --output-dir real_gpu_results_YYYYMMDD_HHMMSS
   ```

## What It Does

Trains a tiny GPT model on different GPU types (NVIDIA V100, A100, AMD MI210) using:
- **Precisions**: FP16, BF16
- **Learning rates**: 1e-3, 3e-3, 1e-2
- **Sub-scales**: 1e-3, 1e-4, 1e-5 (to stress subnormal handling)
- **Seeds**: 3 seeds per configuration

Each job saves a result pickle file with:
- Training loss curve
- Final/best loss
- Divergence status
- Device name

## Output Structure

```
real_gpu_results_YYYYMMDD_HHMMSS/
├── config.txt                    # Job configuration
├── merged_results.pkl            # All results merged (after running merge script)
├── job_logs/                     # SLURM job scripts and logs
│   ├── real_v100_fp16_*.sh
│   ├── real_v100_fp16_*.out
│   └── real_v100_fp16_*.err
├── results/                      # Individual result pickle files
│   ├── result_vendor=*.pkl
│   └── ...
└── source_code/                  # Copy of all source files for reproducibility
```

## Customization

### Change hyperparameters:
```bash
LRS="1e-4,1e-3,3e-3" SUB_SCALES="1e-4,1e-5" ./submit_real_gpu_batch.sh
```

### Test specific GPUs:
Edit `GPU_TYPES` in `submit_real_gpu_batch.sh`:
```bash
GPU_TYPES=("a100")  # Only test A100
```

### Run single job manually:
```bash
python sweep_lr_real_vendor_job.py \
  --vendor fp16_a100 \
  --precision fp16 \
  --lr 0.01 \
  --sub-scale 1e-4 \
  --seed 1 \
  --output-dir ./test_output
```

## Notes

- The code uses PyTorch's Automatic Mixed Precision (AMP) to control precision
- Vendor names (e.g., `fp16_a100`, `fp16_mi250x`) are just labels for tracking
- The model uses `sub_scale` to push activations into subnormal ranges, stressing floating-point formats
- Jobs will queue if GPUs are not immediately available

