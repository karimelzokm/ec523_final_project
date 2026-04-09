# RLVR Pilot – SFT vs GRPO on GSM8K / MATH

Testing whether GRPO with a verifiable reward (exact-match) beats SFT on math reasoning.

Runs on a single NVIDIA L40S (48 GB) using QLoRA (4-bit + LoRA).

---

## Files

```
.
├── pilot_gsm8k_sft_grpo.py       # main script, all the logic
├── run.sh                         # SGE array job for full sweep
├── run_pilot.sh                   # quick local run
├── setup_env.sh                   # conda env setup
├── requirements.txt
├── results/
│   ├── Qwen2.5-3B-Instruct/
│   │   ├── gsm8k_binary/         # base/sft/grpo metrics + qualitative
│   │   ├── gsm8k_format_bonus/
│   │   ├── gsm8k_length_penalty/
│   │   ├── math_binary/
│   │   ├── math_format_bonus/
│   │   └── math_length_penalty/
│   ├── Qwen2.5-1.5B-Instruct/
│   └── Llama-3.2-1B-Instruct/
└── checkpoints/                   # LoRA adapters (not in repo, too large)
```

## Setup

```bash
pip install -r requirements.txt
```

---

## Running

```bash
# full pipeline on both datasets (SFT -> eval -> GRPO -> eval)
python pilot_gsm8k_sft_grpo.py --mode all --dataset both
```

You can also change the reward type (`--reward_type binary|format_bonus|length_penalty`), model (`--model_name`), etc. See the argparser in the script for all flags.

---

## Method

**SFT**: Standard fine-tuning on (prompt, answer) pairs. Prompt tokens are masked so only the answer contributes to the loss.

**GRPO**: For each prompt, sample K completions, score them with a reward function (exact-match), compute advantages relative to the group mean, then do a PPO-clip update with a KL penalty against the frozen SFT model. No value network needed.

**Reward types** (for ablation):
- `binary` — 1 if correct, 0 otherwise
- `format_bonus` — adds a small bonus for using the right answer format
- `length_penalty` — penalizes long outputs

