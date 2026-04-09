#! /bin/bash
# run.sh - full experiment sweep (base/SFT/GRPO) on GSM8K + MATH
# each array task runs one (dataset, reward) combo
# submit with: qsub run.sh

# SGE directives
#$ -P vkolagrp
#$ -t 1-6
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l h_rt=48:00:00
#$ -N RLVR_sweep
#$ -j y
#$ -m ea
#$ -o /projectnb/vkolagrp/kelzokm/RLVR_EC523/qsub_runs

# environment setup
source /share/pkg.8/miniconda/24.5.0/install/etc/profile.d/conda.sh
conda activate /projectnb/vkolagrp/kelzokm/conda_envs/grpo
export PYTHONNOUSERSITE=1
export HF_HOME="/projectnb/vkolagrp/kelzokm/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd /projectnb/vkolagrp/kelzokm/RLVR_EC523
mkdir -p qsub_runs

# map array task to (dataset, reward)
case "$SGE_TASK_ID" in
  1) DATASET="gsm8k" ; REWARD="binary"         ;;
  2) DATASET="gsm8k" ; REWARD="format_bonus"   ;;
  3) DATASET="gsm8k" ; REWARD="length_penalty" ;;
  4) DATASET="math"  ; REWARD="binary"         ;;
  5) DATASET="math"  ; REWARD="format_bonus"   ;;
  6) DATASET="math"  ; REWARD="length_penalty" ;;
  *) echo "Unexpected SGE_TASK_ID=$SGE_TASK_ID"; exit 1 ;;
esac

MODEL="meta-llama/Llama-3.2-1B-Instruct"
MODEL_SHORT="${MODEL##*/}"
SFT_CKPT="checkpoints/${MODEL_SHORT}/${DATASET}_${REWARD}/sft_lora"
GRPO_CKPT="checkpoints/${MODEL_SHORT}/${DATASET}_${REWARD}/grpo_lora"
RESULTS="results/${MODEL_SHORT}/${DATASET}_${REWARD}"

echo "============================================================"
echo "  RLVR sweep  task=$SGE_TASK_ID  dataset=$DATASET  reward=$REWARD"
echo "  SFT ckpt  : $SFT_CKPT"
echo "  GRPO ckpt : $GRPO_CKPT"
echo "  Results   : $RESULTS"
echo "============================================================"

# shared hyperparameters (train_size/eval_size -1 = full dataset)
COMMON_ARGS=(
  --model_name   "$MODEL"
  --dataset      "$DATASET"
  --K            4
  --batch_prompts 4
  --sft_steps    400
  --grpo_steps   300
  --max_new_tokens 256
  --beta_kl      0.1
  --pass_at_k    4
  --seed         42
  --sft_ckpt     "$SFT_CKPT"
  --grpo_ckpt    "$GRPO_CKPT"
  --results_dir  "$RESULTS"
  --reward_type  "$REWARD"
)

# run it
python pilot_gsm8k_sft_grpo.py --mode all "${COMMON_ARGS[@]}"

# print summary
echo ""
echo "============================================================"
echo "  RESULTS  [dataset=$DATASET  reward=$REWARD]"
echo "============================================================"
python - <<PYEOF
import json, os
r = "$RESULTS"
phases = ["base", "sft", "grpo"]
for p in phases:
    path = f"{r}/{p}_metrics.json"
    if not os.path.exists(path):
        continue
    d = json.load(open(path))
    pak = f"  pass@{d.get('k','?')}={d.get('pass_at_k','n/a'):.3f}" if d.get('pass_at_k') is not None else ""
    print(f"  {p.upper():8s}  acc={d['accuracy']:.3f}  "
          f"fmt={d['format_success']:.3f}  "
          f"len={d['avg_output_length']:.1f}w{pak}")
PYEOF
echo "============================================================"
echo "  Full metrics → $RESULTS/"
echo "  System info  → $RESULTS/system_info.json"
echo "============================================================"
