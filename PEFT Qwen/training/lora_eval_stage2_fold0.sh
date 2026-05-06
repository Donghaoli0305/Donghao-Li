#!/bin/bash
#SBATCH --job-name=calib_lora_fold0
#SBATCH --output=/blue/cis6930/d.li2/pubmedqa_project/logs/calib_lora_fold0_%j.out
#SBATCH --error=/blue/cis6930/d.li2/pubmedqa_project/logs/calib_lora_fold0_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1
#SBATCH --account=cis6930
#SBATCH --qos=cis6930

module load conda
module load cuda/12.8.1
conda activate qwen_finetune

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_TORCHVISION=1

PROJECT_DIR=/blue/cis6930/d.li2/pubmedqa_project
SCRIPT_DIR=${PROJECT_DIR}/scripts
BASE_MODEL=${PROJECT_DIR}/model/Qwen2.5-7B
ADAPTER_DIR=${PROJECT_DIR}/outputs/stage2/fold0/final
VAL_DATA_DIR=${PROJECT_DIR}/processed_data/pqal/fold0/hf_dataset

cd ${SCRIPT_DIR}

python lora_evaluation.py \
  --data "${VAL_DATA_DIR}" \
  --base-model "${BASE_MODEL}" \
  --adapter "${ADAPTER_DIR}" \
  --split validation \
  --bsz 32 \
  --max-len 1024 \
  --answer-tag "Answer:" \
  --calibrate \
  --grid-yes="0,0.1,0.2" \
  --grid-no="0.5,1.0,1.5,1.8,2,2.5" \
  --grid-maybe="-0.4,-0.2,0,0.2,0.4" \
  --grid-margin="0.3,0.5,0.7" \
  --calib-metric accuracy
  
EXIT_CODE=$?