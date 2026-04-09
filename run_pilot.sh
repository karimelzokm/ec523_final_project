#!/usr/bin/env bash
# run_pilot.sh - quick local run (no qsub)
# usage: bash run_pilot.sh
#        DATASET=gsm8k REWARD=binary bash run_pilot.sh
set -euo pipefail

PYTHON=${PYTHON:-python}
MODEL=${MODEL:-"Qwen/Qwen2.5-3B-Instruct"}
DATASET=${DATASET:-"both"}          # gsm8k | math | both
REWARD=${REWARD:-"binary"}          # binary | format_bonus | length_penalty
K=${K:-4}
SFT_STEPS=${SFT_STEPS:-400}
GRPO_STEPS=${GRPO_STEPS:-300}
TRAIN_SIZE=${TRAIN_SIZE:--1}        # -1 = full dataset
EVAL_SIZE=${EVAL_SIZE:--1}
PASS_AT_K=${PASS_AT_K:-4}
SEED=${SEED:-42}

MODEL_SHORT="${MODEL##*/}"
SFT_CKPT="checkpoints/${MODEL_SHORT}/${REWARD}/sft_lora"
GRPO_CKPT="checkpoints/${MODEL_SHORT}/${REWARD}/grpo_lora"
RESULTS="results/${MODEL_SHORT}/${REWARD}"

echo "============================================================"
echo "  Pilot run"
echo "  Dataset   : $DATASET"
echo "  Reward    : $REWARD"
echo "  K         : $K"
echo "  SFT steps : $SFT_STEPS"
echo "  GRPO steps: $GRPO_STEPS"
echo "  pass@k    : $PASS_AT_K"
echo "  Results → : $RESULTS"
echo "============================================================"

$PYTHON pilot_gsm8k_sft_grpo.py \
    --mode         all \
    --model_name   "$MODEL" \
    --dataset      "$DATASET" \
    --reward_type  "$REWARD" \
    --K            "$K" \
    --sft_steps    "$SFT_STEPS" \
    --grpo_steps   "$GRPO_STEPS" \
    --train_size   "$TRAIN_SIZE" \
    --eval_size    "$EVAL_SIZE" \
    --pass_at_k    "$PASS_AT_K" \
    --seed         "$SEED" \
    --sft_ckpt     "$SFT_CKPT" \
    --grpo_ckpt    "$GRPO_CKPT" \
    --results_dir  "$RESULTS"

echo ""
echo "============================================================"
echo "  RESULTS  [dataset=$DATASET  reward=$REWARD]"
echo "============================================================"
python - <<PYEOF
import json, os
r = "$RESULTS"
for p in ["base", "sft", "grpo"]:
    path = f"{r}/{p}_metrics.json"
    if not os.path.exists(path): continue
    d = json.load(open(path))
    pak = f"  pass@{d.get('k','?')}={d.get('pass_at_k'):.3f}" if d.get('pass_at_k') is not None else ""
    print(f"  {p.upper():8s}  acc={d['accuracy']:.3f}  "
          f"fmt={d['format_success']:.3f}  "
          f"len={d['avg_output_length']:.1f}w{pak}")
PYEOF
echo "============================================================"
