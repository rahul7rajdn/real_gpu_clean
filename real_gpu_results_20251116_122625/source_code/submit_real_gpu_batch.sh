#!/bin/bash
# submit_real_gpu_batch.sh
#
# Submit SLURM jobs to run experiments on different real GPU vendors in parallel.
# Each GPU type × precision × hyperparameter combination gets its own job.
#
# Usage:
#   ./submit_real_gpu_batch.sh
#   PARTITION=your-partition QOS=your-qos ./submit_real_gpu_batch.sh
#
# Configuration:
#   - Edit GPU_TYPES array to select which GPUs to test
#   - Edit LRS and SUB_SCALES to change hyperparameter sweep
#   - Adjust PARTITION and QOS based on your cluster

set -e

# Default values - ADJUST THESE FOR YOUR CLUSTER
PARTITION="${PARTITION:-coc-gpu}"  # Change to your partition (e.g., coc-gpu, pace-gpu, ice-gpu, coe-gpu)
QOS="${QOS:-coc-ice}"  # Change to your QOS (e.g., coc-ice, coc-students, coc-grade)
JOB_TIME="${JOB_TIME:-02:00:00}"
MAX_STEPS="${MAX_STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
N_SEEDS="${N_SEEDS:-3}"

# GPU types to test - EDIT THIS LIST
GPU_TYPES=("v100" "a100" "mi210")
# Available options: v100, a100, h100, h200, mi210

# Precisions to test for each GPU
declare -A GPU_PRECISIONS
GPU_PRECISIONS["v100"]="fp16"
GPU_PRECISIONS["a100"]="fp16 bf16"
GPU_PRECISIONS["h100"]="bf16 fp16"
GPU_PRECISIONS["h200"]="bf16 fp16"
GPU_PRECISIONS["mi210"]="fp16 bf16"

# Hyperparameters to sweep - EDIT THESE
LRS="${LRS:-1e-3,3e-3,1e-2}"
SUB_SCALES="${SUB_SCALES:-1e-3,1e-4,1e-5}"

# Parse comma-separated lists
IFS=',' read -ra LR_ARRAY <<< "$LRS"
IFS=',' read -ra SUB_SCALE_ARRAY <<< "$SUB_SCALES"

# Create timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="real_gpu_results_${TIMESTAMP}"
JOB_OUTPUT_DIR="${OUTPUT_DIR}/job_logs"
RESULTS_DIR="${OUTPUT_DIR}/results"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${JOB_OUTPUT_DIR}"
mkdir -p "${RESULTS_DIR}"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Copy all source code for reproducibility
SOURCE_CODE_DIR="${OUTPUT_DIR}/source_code"
mkdir -p "${SOURCE_CODE_DIR}"

echo "=== Copying source code to ${SOURCE_CODE_DIR} ==="
SOURCE_FILES=(
    "sweep_lr_real_vendor_job.py"
    "real_tiny_gpt_model.py"
    "merge_real_gpu_results.py"
)

for file in "${SOURCE_FILES[@]}"; do
    if [ -f "${SCRIPT_DIR}/${file}" ]; then
        cp "${SCRIPT_DIR}/${file}" "${SOURCE_CODE_DIR}/"
        echo "  Copied ${file}"
    fi
done

# Copy this script
cp "${BASH_SOURCE[0]}" "${SOURCE_CODE_DIR}/"

# Save configuration
CONFIG_FILE="${OUTPUT_DIR}/config.txt"
cat > "${CONFIG_FILE}" <<EOF
Timestamp: ${TIMESTAMP}
GPU Types: ${GPU_TYPES[@]}
Learning Rates: ${LRS}
Sub Scales: ${SUB_SCALES}
Max Steps: ${MAX_STEPS}
Batch Size: ${BATCH_SIZE}
Seeds per config: ${N_SEEDS}
Partition: ${PARTITION}
QOS: ${QOS}
Job Time: ${JOB_TIME}
EOF

echo ""
echo "=== Configuration ==="
cat "${CONFIG_FILE}"
echo ""

# Count total jobs
TOTAL_JOBS=0
for gpu_type in "${GPU_TYPES[@]}"; do
    PRECISIONS=(${GPU_PRECISIONS[$gpu_type]})
    for precision in "${PRECISIONS[@]}"; do
        for lr in "${LR_ARRAY[@]}"; do
            for sub_scale in "${SUB_SCALE_ARRAY[@]}"; do
                for seed in $(seq 1 ${N_SEEDS}); do
                    TOTAL_JOBS=$((TOTAL_JOBS + 1))
                done
            done
        done
    done
done

echo "=== Submitting ${TOTAL_JOBS} jobs ==="
echo ""

JOB_COUNT=0

# Submit jobs
for gpu_type in "${GPU_TYPES[@]}"; do
    PRECISIONS=(${GPU_PRECISIONS[$gpu_type]})
    
    for precision in "${PRECISIONS[@]}"; do
        # Determine vendor name based on GPU type and precision
        case "$gpu_type" in
            v100|a100|h100|h200)
                # NVIDIA GPUs
                if [ "$precision" = "bf16" ]; then
                    VENDOR="bf16_a100"
                else
                    VENDOR="fp16_a100"
                fi
                ;;
            mi210)
                # AMD MI210
                if [ "$precision" = "bf16" ]; then
                    VENDOR="bf16_mi250x"
                else
                    VENDOR="fp16_mi250x"
                fi
                ;;
            *)
                echo "Unknown GPU type for vendor mapping: $gpu_type"
                continue
                ;;
        esac
        
        # Determine GRES specification
        case "$gpu_type" in
            v100)
                GRES="gpu:v100:1"
                ;;
            a100)
                GRES="gpu:a100:1"
                ;;
            h100)
                GRES="gpu:h100:1"
                ;;
            h200)
                GRES="gpu:h200:1"
                ;;
            mi210)
                GRES="gpu:mi210:1"
                ;;
            *)
                echo "Unknown GPU type: $gpu_type"
                continue
                ;;
        esac
        
        for lr in "${LR_ARRAY[@]}"; do
            for sub_scale in "${SUB_SCALE_ARRAY[@]}"; do
                for seed in $(seq 1 ${N_SEEDS}); do
                    JOB_COUNT=$((JOB_COUNT + 1))
                    
                    JOB_NAME="real_${gpu_type}_${precision}_lr${lr}_sub${sub_scale}_s${seed}"
                    
                    # Create job script
                    JOB_SCRIPT="${JOB_OUTPUT_DIR}/${JOB_NAME}.sh"
                    cat > "${JOB_SCRIPT}" <<JOBEOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${JOB_OUTPUT_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${JOB_OUTPUT_DIR}/${JOB_NAME}_%j.err
#SBATCH --time=${JOB_TIME}
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=${GRES}

# Verify GPU
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Run training
cd "${SCRIPT_DIR}"
python sweep_lr_real_vendor_job.py \
  --vendor "${VENDOR}" \
  --precision "${precision}" \
  --lr ${lr} \
  --sub-scale ${sub_scale} \
  --seed ${seed} \
  --max-steps ${MAX_STEPS} \
  --batch-size ${BATCH_SIZE} \
  --output-dir "${RESULTS_DIR}"

JOBEOF
                    
                    # Submit job
                    SUBMIT_OUTPUT=$(sbatch "${JOB_SCRIPT}" 2>&1)
                    EXIT_CODE=$?
                    
                    if [ $EXIT_CODE -ne 0 ] || echo "$SUBMIT_OUTPUT" | grep -qi "error"; then
                        echo "ERROR: Failed to submit ${JOB_NAME}"
                        echo "  GPU type: ${gpu_type}, GRES: ${GRES}"
                        echo "  Error: $SUBMIT_OUTPUT"
                        continue
                    else
                        if [ $((JOB_COUNT % 10)) -eq 0 ]; then
                            echo "  Submitted ${JOB_COUNT}/${TOTAL_JOBS} jobs..."
                        fi
                        sleep 0.1
                    fi
                done
            done
        done
    done
done

echo ""
echo "=== All ${TOTAL_JOBS} jobs submitted ==="
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo "Job logs: ${JOB_OUTPUT_DIR}"
echo "Results: ${RESULTS_DIR}"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "After jobs complete, merge results with:"
echo "  python merge_real_gpu_results.py --output-dir ${OUTPUT_DIR}"

