# Real GPU Training Experiments

Minimal codebase for running floating-point format experiments on real GPU hardware (NVIDIA and AMD).

## Files

- `sweep_lr_real_vendor_job.py` - Main training script using PyTorch AMP
- `real_tiny_gpt_model.py` - Tiny GPT model definition
- `merge_real_gpu_results.py` - Script to aggregate results from all jobs
- `plot_graphs.py` - Plots loss saturation
- `README.md` - This file

## Steps

### MI210 GPU 

```
salloc --gres=gpu:MI210:1 --ntasks-per-node=4

./run_script_mi210.sh
```

##### Note:
 - there are 6 runs in the above script (3 fp16 + 3 bf16);
 - Comment the runs if required
 - replace the 3000 (i.e, steps) as necessary
 - results written to testing_3000_steps/results/


### A100 GPU 

```
salloc --gres=gpu:A100:1 --ntasks-per-node=4

./run_script_a100.sh
```

### Merge Results:

```
module load pytorch

python merge_real_gpu_results.py --output-dir  testing_3000_steps/
```

### Plot graphs
```
python plot_graphs.py --result-file testing_3000_steps/merged_results.pkl --output-dir plots_3000_steps
```