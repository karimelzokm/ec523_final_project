# RLVR Pilot – SFT vs GRPO on GSM8K / MATH

Testing whether GRPO with a verifiable reward (exact-match) beats SFT on math reasoning.

Runs on a single NVIDIA L40S (48 GB) using QLoRA (4-bit + LoRA).

---

## Repo layout

```
.
├── pilot_gsm8k_sft_grpo.py   # single script – all logic lives here
├── requirements.txt
├── run_pilot.sh               # orchestration shell script
├── README.md
├── checkpoints/
│   ├── sft_lora/             # saved after SFT
│   └── grpo_lora/            # saved after GRPO
└── results/
    ├── sft_metrics.json
    ├── grpo_metrics.json
    ├── sft_train.jsonl        # per-step loss
    ├── grpo_train.jsonl       # per-step reward + KL
    ├── qualitative.jsonl      # 10 examples per phase
    └── summary.json           # side-by-side comparison
```

---

## Setup

```bash
# 1. Create and activate a virtual environment (or use an existing one)
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Log in to HuggingFace if the model is gated
huggingface-cli login
```

> **Note on bitsandbytes**: `bitsandbytes>=0.43` ships CUDA 12 wheels.
> If you see import errors on CUDA 11.x, try `pip install bitsandbytes==0.41.3`.

---

## Running

### Full pipeline (recommended)

```bash
bash run_pilot.sh
```

This runs four steps in sequence:
1. `train_sft`  – fine-tune on 1 000 GSM8K examples (≈ 400 gradient steps)
2. `eval_sft`   – greedy-decode accuracy on 200 test examples
3. `train_grpo` – GRPO starting from the SFT adapter (≈ 300 updates)
4. `eval_grpo`  – greedy-decode accuracy on the same 200 examples

### Run steps individually

```bash
# SFT only
python pilot_gsm8k_sft_grpo.py --mode train_sft

# Evaluate SFT
python pilot_gsm8k_sft_grpo.py --mode eval_sft

# GRPO (requires SFT checkpoint)
python pilot_gsm8k_sft_grpo.py --mode train_grpo

# Evaluate GRPO
python pilot_gsm8k_sft_grpo.py --mode eval_grpo

# All-in-one
python pilot_gsm8k_sft_grpo.py --mode all
```

### Switch to K = 4 samples per prompt

```bash
bash run_pilot.sh K=4
# or directly:
python pilot_gsm8k_sft_grpo.py --mode train_grpo --K 4
```

### Quick smoke test (< 10 min)

```bash
python pilot_gsm8k_sft_grpo.py \
    --mode all \
    --sft_steps  50 \
    --grpo_steps 50 \
    --train_size 200 \
    --eval_size  50
```

---

## Key CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `all` | `train_sft / eval_sft / train_grpo / eval_grpo / all` |
| `--model_name` | `Qwen/Qwen2.5-1.5B-Instruct` | HuggingFace model ID |
| `--K` | `2` | GRPO samples per prompt |
| `--sft_steps` | `400` | SFT gradient update steps |
| `--grpo_steps` | `300` | GRPO update steps |
| `--train_size` | `1000` | Examples from GSM8K train |
| `--eval_size` | `200` | Examples from GSM8K test |
| `--beta_kl` | `0.05` | KL penalty weight (GRPO) |
| `--temperature` | `0.8` | Sampling temperature (GRPO) |
| `--seed` | `42` | Global random seed |

---

## Expected runtimes on L40S (48 GB)

| Phase | Steps | Rough time |
|-------|-------|------------|
| SFT training | 400 steps, bs=4, accum=8 | 25–40 min |
| SFT eval | 200 examples, greedy | 3–5 min |
| GRPO training | 300 steps, K=2, 2 prompts/step | 30–55 min |
| GRPO eval | 200 examples, greedy | 3–5 min |
| **Total** | | **~1.0–1.7 h** |

With `K=4`: GRPO training rises to ≈ 55–90 min.

---

## Method overview

### SFT

Standard causal language-modelling on `(prompt, answer)` pairs.
Prompt labels are masked; only the answer tokens contribute to the loss.

### GRPO (Group-Relative Policy Optimisation)

For each prompt **x**, sample **K** completions { y₁ … yK } from the current
policy, then:

1. Compute reward  rᵢ = R(x, yᵢ)  (verifiable: exact numeric match)
2. Group baseline  b  = mean(rᵢ)
3. Advantage       Aᵢ = rᵢ − b
4. PPO-clip loss (sequence-level ratio):

```
ratio   = exp( Σ log p_new(y) − Σ log p_old(y) )
L_clip  = −E[ min( ratio · A, clip(ratio, 1±ε) · A ) ]
```

5. KL penalty vs. frozen reference (SFT model):

```
KL      ≈ mean_token[ log p_new − log p_ref ]
L_total = L_clip + β · KL
```

No value network, no GAE.

### Reward function

| Condition | Reward |
|-----------|--------|
| Exact numeric match after `####` | +1.0 |
| `####` present but wrong number | 0.0 |
| No `####` in output | −0.2 |

---

## Troubleshooting

### Out of Memory

- Reduce `grpo_batch_prompts` to 1 and/or `K` to 2 (already the defaults).
- Reduce `sft_batch_size` to 2 and `sft_grad_accum` to 16.
- Use `Qwen/Qwen2.5-0.5B-Instruct` instead of 1.5B:
  ```bash
  python pilot_gsm8k_sft_grpo.py --model_name Qwen/Qwen2.5-0.5B-Instruct --mode all
  ```

### Slow generation

- During GRPO, generation dominates. Reduce `grpo_max_new_tokens` to 64.
- Reduce `grpo_K` to 2 (already default).

### bf16 / fp16 fallback

The script auto-detects `torch.cuda.is_bf16_supported()` and falls back to
fp16 if needed. On L40S, bf16 is supported and preferred.

### bitsandbytes CUDA errors

```bash
pip install bitsandbytes --upgrade
# or pin to a known-good version:
pip install bitsandbytes==0.43.3
```

### Model download issues

```bash
huggingface-cli login
# or set the token env var:
export HUGGING_FACE_HUB_TOKEN=hf_...
```
