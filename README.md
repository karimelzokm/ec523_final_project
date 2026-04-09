# RLVR Pilot – SFT vs GRPO on GSM8K / MATH

Testing whether GRPO with a verifiable reward (exact-match) beats SFT on math reasoning.

Runs on a single NVIDIA L40S (48 GB) using QLoRA (4-bit + LoRA).

---

## Files

- `pilot_gsm8k_sft_grpo.py` — main script, everything is in here
- `run_pilot.sh` / `run.sh` — shell scripts to launch runs
- `setup_env.sh` — conda env setup
- `results/` — saved metrics and qualitative examples
- `checkpoints/` — LoRA adapters (not in repo, too large)

## Setup

```bash
pip install -r requirements.txt
```

---

## Running

```bash
# full pipeline (SFT -> eval -> GRPO -> eval)
bash run_pilot.sh

# or run steps separately
python pilot_gsm8k_sft_grpo.py --mode train_sft
python pilot_gsm8k_sft_grpo.py --mode eval_sft
python pilot_gsm8k_sft_grpo.py --mode train_grpo
python pilot_gsm8k_sft_grpo.py --mode eval_grpo

# run everything at once
python pilot_gsm8k_sft_grpo.py --mode all
```

You can also change the dataset (`--dataset gsm8k|math|both`), reward type (`--reward_type binary|format_bonus|length_penalty`), model (`--model_name`), etc. Check the argparser in the script for all flags.

---

## Method

**SFT**: Standard fine-tuning on (prompt, answer) pairs. Prompt tokens are masked so only the answer contributes to the loss.

**GRPO**: For each prompt, sample K completions, score them with a reward function (exact-match), compute advantages relative to the group mean, then do a PPO-clip update with a KL penalty against the frozen SFT model. No value network needed.

**Reward types** (for ablation):
- `binary` — 1 if correct, 0 otherwise
- `format_bonus` — adds a small bonus for using the right answer format
- `length_penalty` — penalizes long outputs

