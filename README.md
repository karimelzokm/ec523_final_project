# RLVR Pilot – SFT vs GRPO on GSM8K / MATH

Testing whether GRPO with a verifiable reward (exact-match) beats SFT on math reasoning.

Runs on a single NVIDIA L40S (48 GB) using QLoRA (4-bit + LoRA).

---

## Files

```
.
├── pilot_gsm8k_sft_grpo.py       # main script, all the logic
├── setup_env.sh                   # conda env setup (recommended)
├── requirements.txt               # pinned deps (used by setup_env.sh)
├── jobs/
│   ├── download_models.sh        # pre-cache models & datasets
│   ├── run_qwen3b.sh             # Qwen2.5-3B-Instruct sweep   (9-task array)
│   ├── run_qwen1.5b.sh           # Qwen2.5-1.5B-Instruct sweep (9-task array)
│   ├── run_qwen7b.sh             # Qwen2.5-7B-Instruct sweep   (3-task array, GSM8K only)
│   ├── run_llama3b.sh            # Llama-3.2-3B-Instruct sweep (9-task array)
│   └── run_llama1b.sh            # Llama-3.2-1B-Instruct sweep (9-task array)
├── results/
│   └── <ModelShort>/<dataset>_<reward>/   # base/sft/grpo metrics
└── checkpoints/                   # LoRA adapters (not in repo, too large)
```

## Setup

Recommended: build the pinned conda env via the provided script. This installs PyTorch 2.5.1+cu121 with the exact transformers/peft/trl/bitsandbytes versions used for every result in the repo.

```bash
bash setup_env.sh
conda activate /projectnb/vkolagrp/kelzokm/conda_envs/grpo   # adjust path as needed
```

If you only need to install on top of an existing environment:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes [`math-verify`](https://github.com/huggingface/Math-Verify), which the layered MATH grader uses to recognize equivalent LaTeX answers (e.g. `\dfrac{1}{2}` vs `\frac{1}{2}`, ordered tuples). If `math-verify` is missing the script falls through to a hand-rolled normalizer; this is silent at runtime but will cost ~2pp on full-MATH accuracy.

---

## Running

### SGE array jobs

Each 1.5B / 3B job script in `jobs/` runs one model across 9 array tasks (`{gsm8k, math, both} × {binary, format_bonus, length_penalty}`):

| Task | Dataset | Reward |
|------|---------|--------|
| 1 | gsm8k | binary |
| 2 | gsm8k | format_bonus |
| 3 | gsm8k | length_penalty |
| 4 | math | binary |
| 5 | math | format_bonus |
| 6 | math | length_penalty |
| 7 | both | binary |
| 8 | both | format_bonus |
| 9 | both | length_penalty |

`run_qwen7b.sh` is a reduced 3-task array (`{gsm8k} × {binary, format_bonus, length_penalty}`) with `--K 2 --batch_prompts 2` to fit on a single L40S.

Pre-cache models and datasets first:

```bash
bash jobs/download_models.sh
```

Submit all sweeps:

```bash
qsub jobs/run_qwen3b.sh
qsub jobs/run_qwen1.5b.sh
qsub jobs/run_qwen7b.sh
qsub jobs/run_llama3b.sh
qsub jobs/run_llama1b.sh
```

To run a subset of tasks, use `-t`:

```bash
qsub -t 4-6 jobs/run_llama1b.sh   # only MATH runs
qsub -t 7-9 jobs/run_qwen3b.sh    # only "both" runs
```

### Local run

```bash
python pilot_gsm8k_sft_grpo.py --mode all --dataset both
```

See the argparser in the script for all flags (`--reward_type`, `--model_name`, etc.).

---

## Method

**SFT**: Standard fine-tuning on (prompt, answer) pairs. Prompt tokens are masked so only the answer contributes to the loss.

**GRPO**: For each prompt, sample K completions, score them with a reward function (exact-match), compute advantages relative to the group mean, then do a PPO-clip update with a KL penalty against the frozen SFT model. No value network needed.

**Reward types** (for ablation):
- `binary` — 1 if correct, 0 otherwise
- `format_bonus` — adds a small bonus for using the right answer format
- `length_penalty` — penalizes long outputs

