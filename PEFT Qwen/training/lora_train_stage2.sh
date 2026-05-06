#!/bin/bash
#SBATCH --job-name=qwen_lora_stage2_arr
#SBATCH --output=/blue/cis6930/d.li2/pubmedqa_project/logs/lora_stage2_fold%a_%j.out
#SBATCH --error=/blue/cis6930/d.li2/pubmedqa_project/logs/lora_stage2_fold%a_%j.err

#SBATCH --array=0-9          # 关键：10个fold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=24:00:00


#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1

#SBATCH --account=cis6930
#SBATCH --qos=cis6930

module load conda
module load cuda/12.8.1
conda activate qwen_finetune

export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=0

PROJECT_DIR=/blue/cis6930/d.li2/pubmedqa_project
SCRIPT_DIR=${PROJECT_DIR}/scripts
MODEL_DIR=${PROJECT_DIR}/model/Qwen2.5-7B
INIT_ADAPTER=${PROJECT_DIR}/outputs/stage1/final

FOLD_ID=${SLURM_ARRAY_TASK_ID}
DATASET_DIR_STAGE2=${PROJECT_DIR}/processed_data/pqal/fold${FOLD_ID}/hf_dataset

OUT_DIR=${PROJECT_DIR}/outputs/stage2_fold${FOLD_ID}
mkdir -p ${PROJECT_DIR}/logs
mkdir -p ${OUT_DIR}

cd ${SCRIPT_DIR}
echo "=== Stage-2 LoRA Fold ${FOLD_ID} ==="

python lora_v2_train.py \
  --data   "${DATASET_DIR_STAGE2}" \
  --model  "${MODEL_DIR}" \
  --out    "${OUT_DIR}" \
  --epochs 1 \
  --bsz 1 \
  --eval-bsz 16 \
  --grad-acc 8 \
  --lr 1e-4 \
  --warmup 0.03 \
  --max-len 256 \
  --eval-steps 100000000 \
  --save-steps 100000000 \
  --logging-steps 50 \
  --oversample \
  --gc \
  --init-adapter "${INIT_ADAPTER}"
