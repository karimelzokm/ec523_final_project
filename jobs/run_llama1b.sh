#!/bin/bash
# run_llama1b.sh - Llama-3.2-1B-Instruct sweep
# 9 array tasks: {gsm8k, math, both} × {binary, format_bonus, length_penalty}
# submit with: qsub jobs/run_llama1b.sh

#$ -P vkolagrp
#$ -t 1-3
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_type=L40S
#$ -l h_rt=48:00:00
#$ -N RLVR_llama1b
#$ -j y
#$ -m ea
#$ -o /projectnb/vkolagrp/kelzokm/RLVR_EC523/qsub_runs

source /share/pkg.8/miniconda/24.5.0/install/etc/profile.d/conda.sh
conda activate /projectnb/vkolagrp/kelzokm/conda_envs/grpo
export PYTHONNOUSERSITE=1
export HF_HOME="/projectnb/vkolagrp/kelzokm/.cache/huggingface"

cd /projectnb/vkolagrp/kelzokm/RLVR_EC523
mkdir -p qsub_runs

case "$SGE_TASK_ID" in
  1) DATASET="gsm8k" ; REWARD="binary"         ;;
  2) DATASET="gsm8k" ; REWARD="format_bonus"   ;;
  3) DATASET="gsm8k" ; REWARD="length_penalty" ;;
  4) DATASET="math"  ; REWARD="binary"         ;;
  5) DATASET="math"  ; REWARD="format_bonus"   ;;
  6) DATASET="math"  ; REWARD="length_penalty" ;;
  7) DATASET="both"  ; REWARD="binary"         ;;
  8) DATASET="both"  ; REWARD="format_bonus"   ;;
  9) DATASET="both"  ; REWARD="length_penalty" ;;
  *) echo "Unexpected SGE_TASK_ID=$SGE_TASK_ID"; exit 1 ;;
esac

MODEL="meta-llama/Llama-3.2-1B-Instruct"
MODEL_SHORT="${MODEL##*/}"
SFT_CKPT="checkpoints/${MODEL_SHORT}/${DATASET}_${REWARD}/sft_lora"
GRPO_CKPT="checkpoints/${MODEL_SHORT}/${DATASET}_${REWARD}/grpo_lora"
RESULTS="results/${MODEL_SHORT}/${DATASET}_${REWARD}"

echo "============================================================"
echo "  RLVR sweep  task=$SGE_TASK_ID  model=$MODEL_SHORT"
echo "  dataset=$DATASET  reward=$REWARD"
echo "  Results   : $RESULTS"
echo "============================================================"

python pilot_gsm8k_sft_grpo.py --mode all \
  --model_name   "$MODEL" \
  --dataset      "$DATASET" \
  --K            4 \
  --batch_prompts 4 \
  --sft_steps    400 \
  --grpo_steps   300 \
  --max_new_tokens 256 \
  --beta_kl      0.1 \
  --pass_at_k    4 \
  --seed         42 \
  --sft_ckpt     "$SFT_CKPT" \
  --grpo_ckpt    "$GRPO_CKPT" \
  --results_dir  "$RESULTS" \
  --reward_type  "$REWARD"

echo ""
echo "============================================================"
echo "  RESULTS  [model=$MODEL_SHORT  dataset=$DATASET  reward=$REWARD]"
echo "============================================================"
python - <<PYEOF
import json, os
r = "$RESULTS"
for p in ["base", "sft", "grpo"]:
    path = f"{r}/{p}_metrics.json"
    if not os.path.exists(path): continue
    d = json.load(open(path))
    pak = f"  pass@{d.get('k','?')}={d.get('pass_at_k','n/a'):.3f}" if d.get('pass_at_k') is not None else ""
    print(f"  {p.upper():8s}  acc={d['accuracy']:.3f}  fmt={d['format_success']:.3f}  len={d['avg_output_length']:.1f}w{pak}")
PYEOF
echo "============================================================"
