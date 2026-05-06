#!/bin/bash
#SBATCH --job-name=full-ft-stage2
#SBATCH --output=/blue/cis6930/d.li2/pubmedqa_project/logs/lora_finetune_stage1_%j.out
#SBATCH --error=/blue/cis6930/d.li2/pubmedqa_project/logs/lora_finetune_stage1_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=hpg-b200
#SBATCH --account=cis6930
#SBATCH --qos=cis6930

echo "============================================================"
echo "  Lora Fine-tuning Stage 1: PQA-A Balanced"
echo "============================================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Job Name:      $SLURM_JOB_NAME"
echo "Node:          $SLURM_NODELIST"
echo "Partition:     $SLURM_JOB_PARTITION"
echo "GPU:           B200 (192GB VRAM)"
echo "CPUs:          4"
echo "Memory:        32GB"
echo "Start Time:    $(date)"
echo "============================================================"

# Environment setup
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-/blue/cis6930/d.li2/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}

mkdir -p "$HF_HOME"
mkdir -p /blue/cis6930/d.li2/pubmedqa_project/logs

# Load modules
module load conda
module load cuda/12.8.1

# Activate environment
conda activate qwen_finetune

# Paths
DATA_DIR="/blue/cis6930/d.li2/pubmedqa_project/processed_data/pqaa_balanced/hf_dataset"
MODEL_PATH="/blue/cis6930/d.li2/pubmedqa_project/model/Qwen2.5-7B"
OUTPUT_DIR="/blue/cis6930/d.li2/pubmedqa_project/results/lora_stage1"

echo ""
echo "Configuration:"
echo "------------------------------------------------------------"
echo "Data:          $DATA_DIR"
echo "Model:         $MODEL_PATH"
echo "Output:        $OUTPUT_DIR"
echo "============================================================"

# Environment info
echo ""
echo "Environment Information:"
echo "------------------------------------------------------------"
echo "Python:        $(which python)"
echo "PyTorch:       $(python -c 'import torch; print(torch.__version__)')"
echo "Transformers:  $(python -c 'import transformers; print(transformers.__version__)')"
echo "CUDA:          $(python -c 'import torch; print(torch.version.cuda)')"
echo "Device count:  $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "Device name:   $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "BF16 support:  $(python -c 'import torch; print(torch.cuda.is_bf16_supported() if torch.cuda.is_available() else "N/A")')"

if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

echo "============================================================"

# Dataset sanity check
echo ""
echo "Dataset Check:"
python -c "
from datasets import load_from_disk
ds = load_from_disk('$DATA_DIR')
print(f'Splits: {list(ds.keys())}')
for k in ds.keys():
    print(f'  {k}: {len(ds[k])} samples, columns: {ds[k].column_names}')
"

echo "============================================================"

# Run training
echo ""
echo "Starting Full Fine-tuning Stage 1..."
echo "Expected duration: ~6-8 hours"
echo "============================================================"

python /blue/cis6930/d.li2/pubmedqa_project/scripts/lora_v1_train.py \
  --data "$DATA_DIR" \
  --model "$MODEL_PATH" \
  --out "$OUTPUT_DIR" \
  --epochs 2 \
  --bsz 4 \
  --eval-bsz 4 \
  --grad-acc 8 \
  --lr 2e-4 \
  --warmup 0.08 \
  --max-len 1024 \
  --eval-steps 200 \
  --save-steps 200 \
  --logging-steps 100 \
  --oversample \
  --gc \
  --early-stop \
  --early-metric eval_loss \
  --early-patience 2 \
  --early-mode min

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Stage 1 Training finished with exit code: $EXIT_CODE"
echo "End Time:      $(date)"
echo "============================================================"

# Final GPU status
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "Final GPU Status:"
    nvidia-smi
fi

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Stage 1 Complete!"
    echo "  Model saved to: $OUTPUT_DIR/final_model"
    echo ""
    echo "Next: Run Stage 2"
    echo "  sbatch /blue/cis6930/d.li2/pubmedqa_project/jobs/lora_train_stage.sh"
fi

exit $EXIT_CODE