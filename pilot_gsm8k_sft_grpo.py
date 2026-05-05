#!/usr/bin/env python3
"""
SFT vs GRPO on GSM8K / MATH.
Uses QLoRA (4-bit + LoRA) to fit on a single L40S.

Modes: eval_base, train_sft, eval_sft, train_grpo, eval_grpo, all
Datasets: gsm8k (####), math (\\boxed{}), both
"""

from __future__ import annotations

import gc
import json
import os
import random
import re
import time
import argparse
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import multiprocessing
import platform
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from datasets import Dataset as HFDataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from trl import GRPOConfig, GRPOTrainer

# --- Config ---

@dataclass
class PilotConfig:
    # model
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # data
    dataset: str = "gsm8k"       # "gsm8k" | "math" | "both"
    train_size: int = -1         # -1 = full dataset
    eval_size: int = -1          # -1 = full test split
    seed: int = 42

    # SFT
    sft_max_steps: int = 400
    sft_batch_size: int = 4
    sft_grad_accum: int = 8
    sft_lr: float = 2e-4
    sft_warmup_ratio: float = 0.05
    sft_max_seq_len: int = 512

    # GRPO
    grpo_train_size: int = -1    # -1 = full training set
    grpo_max_steps: int = 300
    grpo_batch_prompts: int = 2
    grpo_K: int = 2
    grpo_max_new_tokens: int = 128
    grpo_temperature: float = 0.8
    grpo_lr: float = 1e-5
    grpo_beta_kl: float = 0.05

    # reward shaping -- binary (default), format_bonus, or length_penalty
    reward_type: str = "binary"
    format_bonus_value: float = 0.1
    length_penalty_lambda: float = 0.001

    # eval
    eval_max_new_tokens: int = 128
    eval_batch_size: int = 8
    pass_at_k: int = 4            # k for pass@k metric (set 1 to skip)
    eval_temperature: float = 0.8  # temperature for pass@k stochastic sampling

    # paths
    sft_ckpt: str = "checkpoints/sft_lora"
    grpo_ckpt: str = "checkpoints/grpo_lora"
    results_dir: str = "results"

    mode: str = "all"


def build_config() -> PilotConfig:
    p = argparse.ArgumentParser(description="GSM8K/MATH SFT vs GRPO Pilot")
    p.add_argument("--mode", default="all",
                   choices=["eval_base", "train_sft", "eval_sft",
                            "train_grpo", "eval_grpo", "all"])
    p.add_argument("--model_name",  default=None)
    p.add_argument("--dataset",     default=None,
                   choices=["gsm8k", "math", "both"])
    p.add_argument("--K",           type=int,   default=None)
    p.add_argument("--sft_steps",   type=int,   default=None)
    p.add_argument("--grpo_steps",  type=int,   default=None)
    p.add_argument("--seed",        type=int,   default=None)
    p.add_argument("--beta_kl",       type=float, default=None)
    p.add_argument("--batch_prompts", type=int,   default=None)
    p.add_argument("--max_new_tokens", type=int,  default=None)
    p.add_argument("--pass_at_k",      type=int,   default=None,
                   help="k for pass@k eval metric (default 4, set 1 to skip)")
    p.add_argument("--reward_type", default=None,
                   choices=["binary", "format_bonus", "length_penalty"])
    # output paths (useful when running multiple experiments in parallel)
    p.add_argument("--sft_ckpt",    default=None, help="override SFT checkpoint dir")
    p.add_argument("--grpo_ckpt",   default=None, help="override GRPO checkpoint dir")
    p.add_argument("--results_dir", default=None, help="override results output dir")
    args = p.parse_args()

    cfg = PilotConfig()
    cfg.mode = args.mode
    if args.model_name:         cfg.model_name          = args.model_name
    if args.dataset:            cfg.dataset             = args.dataset
    if args.K:                  cfg.grpo_K              = args.K
    if args.sft_steps:          cfg.sft_max_steps       = args.sft_steps
    if args.grpo_steps:         cfg.grpo_max_steps      = args.grpo_steps
    if args.seed:               cfg.seed                = args.seed
    if args.beta_kl:            cfg.grpo_beta_kl        = args.beta_kl
    if args.batch_prompts:      cfg.grpo_batch_prompts  = args.batch_prompts
    if args.max_new_tokens:
        cfg.grpo_max_new_tokens = args.max_new_tokens
        cfg.eval_max_new_tokens = args.max_new_tokens
    if args.pass_at_k   is not None:  cfg.pass_at_k    = args.pass_at_k
    if args.reward_type is not None:  cfg.reward_type  = args.reward_type
    if args.sft_ckpt    is not None:  cfg.sft_ckpt     = args.sft_ckpt
    if args.grpo_ckpt   is not None:  cfg.grpo_ckpt    = args.grpo_ckpt
    if args.results_dir is not None:  cfg.results_dir  = args.results_dir

    # Auto-nest results under model short name (e.g. results/Qwen2.5-3B-Instruct/...)
    model_short = cfg.model_name.split("/")[-1]
    if not cfg.results_dir.startswith(f"results/{model_short}"):
        sub = cfg.results_dir[len("results/"):] if cfg.results_dir.startswith("results/") else cfg.results_dir.lstrip("results")
        cfg.results_dir = os.path.join("results", model_short, sub)
    return cfg


# --- Utilities ---

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def log_jsonl(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def free_gpu(*models) -> None:
    for m in models:
        if m is not None:
            del m
    gc.collect()
    torch.cuda.empty_cache()


def device_of(model) -> torch.device:
    return next(model.parameters()).device


def _compute_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


# --- Data & answer parsing ---

# prompt templates

GSM8K_PROMPT_TMPL = (
    "Solve the following math problem step by step.\n"
    "Show your reasoning, then end with exactly this format:\n"
    "#### <final numeric answer>\n\n"
    "Problem: {question}"
)

MATH_PROMPT_TMPL = (
    "Solve the following math problem step by step.\n"
    "Show your reasoning, then put your final answer inside \\boxed{{}}.\n\n"
    "Problem: {question}"
)


def make_prompt(ex: dict) -> str:
    q = ex["question"].strip()
    if ex.get("dataset") == "math":
        return MATH_PROMPT_TMPL.format(question=q)
    return GSM8K_PROMPT_TMPL.format(question=q)


# answer extraction

def _parse_answer(text: str) -> Optional[str]:
    """Pull out the number after #### in GSM8K-style answers."""
    m = re.search(r"####\s*([\-\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    return None


def _parse_boxed(text: str) -> Optional[str]:
    """Get whatever's inside the last \\boxed{...}, handles nested braces."""
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        idx = text.rfind(r"\boxed {")
    if idx == -1:
        return None
    brace_start = text.find("{", idx)
    if brace_start == -1:
        return None
    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1 : i].strip()
    return None


def _to_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _normalize_math_str(s: str) -> str:
    """Strip whitespace/commas so we can compare math answers."""
    s = s.strip().lower()
    s = re.sub(r"\s+", "", s)
    s = s.replace(",", "")
    return s


# math-verify is the canonical SymPy-based equivalence checker for MATH-style
# answers (handles \frac vs \dfrac, ordered tuples, units, etc.). If it isn't
# installed we fall through to a hand-rolled latex normalizer.
try:
    from math_verify import parse as _mv_parse, verify as _mv_verify  # type: ignore
    HAS_MATH_VERIFY = True
except Exception:
    HAS_MATH_VERIFY = False


_LATEX_DROP_TOKENS = (
    r"\!", r"\,", r"\;", r"\:", r"\ ",
    r"\left", r"\right",
    r"\displaystyle", r"\,\!",
)


def _frac_to_div(s: str) -> str:
    """Rewrite \\frac{a}{b} as a/b until none remain."""
    if s is None:
        return ""
    while True:
        m = re.search(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", s)
        if not m:
            break
        s = s[: m.start()] + f"{m.group(1)}/{m.group(2)}" + s[m.end():]
    return s


def _strip_latex(s: str) -> str:
    """Aggressive latex normalizer for MATH-style answers."""
    if s is None:
        return ""
    out = s.strip().replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    out = _frac_to_div(out)
    for tok in _LATEX_DROP_TOKENS:
        out = out.replace(tok, "")
    out = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", out)
    out = re.sub(r"\\mathrm\s*\{([^}]*)\}", r"\1", out)
    out = re.sub(r"\\mbox\s*\{([^}]*)\}", r"\1", out)
    out = out.replace(r"^\circ", "").replace(r"^{\circ}", "").replace("°", "")
    out = out.replace(r"\%", "").replace("%", "")
    out = out.replace(r"\$", "").replace("$", "")
    out = out.replace(r"\cdot", "*").replace(r"\times", "*")
    out = out.replace("{", "").replace("}", "")
    out = out.replace("(", "").replace(")", "")
    out = re.sub(r"\s+", "", out)
    return out.lower()


def _math_verify_equiv(pred: str, gt: str) -> Optional[bool]:
    """True/False if math-verify parses both, else None."""
    if not HAS_MATH_VERIFY:
        return None
    try:
        p = _mv_parse(f"${pred}$")
        g = _mv_parse(f"${gt}$")
        return bool(_mv_verify(g, p))
    except Exception:
        return None


def _format_ok(output: str, gt_answer: str) -> bool:
    """Check if the output has an extractable final answer."""
    if "####" in gt_answer:
        return _parse_answer(output) is not None
    return _parse_boxed(output) is not None


def _is_correct(output: str, gt_answer: str) -> bool:
    """Check if the model got the right answer (works for both #### and \\boxed{})."""
    if "####" in gt_answer:
        # gsm8k
        pred = _parse_answer(output)
        if pred is None:
            return False
        gt = _parse_answer(gt_answer) or gt_answer.replace(",", "").strip()
    elif r"\boxed" in gt_answer:
        # math (full solution stored, try extracting from both)
        pred = _parse_boxed(output)
        if pred is None:
            pred = _parse_answer(output)   # model may have used #### fallback
        if pred is None:
            return False
        gt = _parse_boxed(gt_answer)
        if gt is None:
            gt = gt_answer.strip()
    else:
        # gt is already the extracted answer string
        pred = _parse_boxed(output) or _parse_answer(output) or output.strip()
        gt = gt_answer.strip()

    pf, gf = _to_float(pred), _to_float(gt)
    if pf is not None and gf is not None:
        return abs(pf - gf) < 1e-3

    # MATH-style answers can have wildly different latex spellings of the same
    # value (\dfrac vs \frac, parens, units, ordered tuples). Try math-verify
    # first, then a latex normalizer, before falling back to whitespace-strip.
    if "####" not in gt_answer:
        mv = _math_verify_equiv(pred, gt)
        if mv is True:
            return True
        a, b = _strip_latex(pred), _strip_latex(gt)
        if a and a == b:
            return True

    return _normalize_math_str(pred) == _normalize_math_str(gt)


# dataset loading

def _slice(lst: list, n: int) -> list:
    """lst[:n], or the whole thing if n == -1."""
    return lst if n == -1 else lst[:n]


def _normalize_gsm8k_entry(ex: dict) -> dict:
    return {"question": ex["question"], "answer": ex["answer"], "dataset": "gsm8k"}


def _normalize_math_entry(ex: dict) -> dict:
    """Turn a MATH dataset entry into our standard format."""
    question = ex.get("problem") or ex.get("question") or ""
    answer   = ex.get("solution") or ex.get("answer") or ""
    return {
        "question": question,
        "answer":   answer,
        "dataset":  "math",
        "level":    ex.get("level", ""),
        "type":     ex.get("type", ""),
    }


def load_gsm8k(cfg: PilotConfig) -> Tuple[list, list]:
    print("Loading GSM8K …")
    ds    = load_dataset("gsm8k", "main")
    train = [_normalize_gsm8k_entry(e) for e in ds["train"]]
    test  = [_normalize_gsm8k_entry(e) for e in ds["test"]]
    train = _slice(train, cfg.train_size)
    test  = _slice(test,  cfg.eval_size)
    print(f"  GSM8K  train={len(train):,}  eval={len(test):,}")
    return train, test


def load_math(cfg: PilotConfig) -> Tuple[list, list]:
    print("Loading MATH …")
    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default")
    train = [_normalize_math_entry(e) for e in ds["train"]]
    test  = [_normalize_math_entry(e) for e in ds["test"]]
    train = _slice(train, cfg.train_size)
    test  = _slice(test,  cfg.eval_size)
    print(f"  MATH   train={len(train):,}  eval={len(test):,}")
    return train, test


def load_data(cfg: PilotConfig) -> Tuple[list, list]:
    """Load train/eval splits based on cfg.dataset."""
    if cfg.dataset == "gsm8k":
        return load_gsm8k(cfg)
    if cfg.dataset == "math":
        return load_math(cfg)
    if cfg.dataset == "both":
        g_train, g_test = load_gsm8k(cfg)
        m_train, m_test = load_math(cfg)
        rng = random.Random(cfg.seed)
        combined_train = g_train + m_train
        combined_test  = g_test  + m_test
        rng.shuffle(combined_train)
        rng.shuffle(combined_test)
        print(f"  Combined  train={len(combined_train):,}  eval={len(combined_test):,}")
        return combined_train, combined_test
    raise ValueError(f"Unknown dataset '{cfg.dataset}'. Choose: gsm8k | math | both")


# --- Model loading ---

def _bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=_compute_dtype(),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def _lora_targets(model) -> List[str]:
    candidates = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
    existing = {n.split(".")[-1] for n, _ in model.named_modules()}
    found = [c for c in candidates if c in existing]
    return found if found else ["q_proj", "v_proj"]


def load_tokenizer(cfg: PilotConfig):
    tok = AutoTokenizer.from_pretrained(
        cfg.model_name, padding_side="left", trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(cfg: PilotConfig):
    print(f"Loading base model: {cfg.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=_bnb_config() if cfg.load_in_4bit else None,
        device_map="auto",
        torch_dtype=_compute_dtype(),
        trust_remote_code=True,
    )
    model.config.use_cache = False
    return model


def apply_lora(model, cfg: PilotConfig):
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=_lora_targets(model),
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    return model


def load_for_inference(cfg: PilotConfig, ckpt: Optional[str] = None):
    """Load base model, optionally with a LoRA adapter on top."""
    model = load_base_model(cfg)
    if ckpt is not None:
        model = PeftModel.from_pretrained(model, ckpt, is_trainable=False)
    model.eval()
    return model


# --- SFT ---

class SFTDataset(Dataset):
    def __init__(self, data: list, tok, max_len: int):
        self.data, self.tok, self.max_len = data, tok, max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        ex     = self.data[idx]
        prompt = make_prompt(ex)
        answer = ex["answer"].strip()
        full   = prompt + "\n" + answer

        enc   = self.tok(full,   truncation=True, max_length=self.max_len,
                         padding=False, return_tensors=None)
        p_enc = self.tok(prompt, truncation=True, max_length=self.max_len,
                         padding=False, return_tensors=None)
        p_len  = len(p_enc["input_ids"])
        labels = enc["input_ids"].copy()
        labels[:p_len] = [-100] * p_len   # mask the prompt

        return {
            "input_ids":      torch.tensor(enc["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels":         torch.tensor(labels,                dtype=torch.long),
        }


def _sft_collate(batch: list, pad_id: int) -> dict:
    max_len = max(x["input_ids"].size(0) for x in batch)

    def lpad(t: torch.Tensor, val: int) -> torch.Tensor:
        return F.pad(t, (max_len - t.size(0), 0), value=val)

    return {
        "input_ids":      torch.stack([lpad(x["input_ids"],      pad_id) for x in batch]),
        "attention_mask": torch.stack([lpad(x["attention_mask"], 0)      for x in batch]),
        "labels":         torch.stack([lpad(x["labels"],         -100)   for x in batch]),
    }


def train_sft(cfg: PilotConfig) -> float:
    """Train SFT. Returns wall-clock seconds elapsed."""
    print("\n" + "═" * 60)
    print("PHASE – SFT Training")
    print("═" * 60)
    phase_t0 = time.time()
    set_seed(cfg.seed)

    train_data, _ = load_data(cfg)
    tok   = load_tokenizer(cfg)
    model = load_base_model(cfg)
    model = apply_lora(model, cfg)

    ds     = SFTDataset(train_data, tok, cfg.sft_max_seq_len)
    loader = DataLoader(ds, batch_size=cfg.sft_batch_size, shuffle=True,
                        collate_fn=lambda b: _sft_collate(b, tok.pad_token_id))

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.sft_lr, weight_decay=0.01,
    )
    sched = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=int(cfg.sft_max_steps * cfg.sft_warmup_ratio),
        num_training_steps=cfg.sft_max_steps,
    )

    log_path = f"{cfg.results_dir}/sft_train.jsonl"
    model.train()
    opt.zero_grad()
    step = accum = 0
    running_loss = 0.0
    it = iter(loader)
    t0 = time.time()

    while step < cfg.sft_max_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        batch = {k: v.to(device_of(model)) for k, v in batch.items()}
        loss  = model(**batch).loss / cfg.sft_grad_accum
        loss.backward()
        running_loss += loss.item()
        accum += 1

        if accum == cfg.sft_grad_accum:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            accum = 0

            if step % 20 == 0:
                avg = running_loss / 20
                running_loss = 0.0
                print(f"  [SFT] step {step:>4}/{cfg.sft_max_steps}  "
                      f"loss={avg:.4f}  lr={sched.get_last_lr()[0]:.2e}  "
                      f"{time.time()-t0:.0f}s")
                log_jsonl(log_path, {"step": step, "loss": avg})

    os.makedirs(cfg.sft_ckpt, exist_ok=True)
    model.save_pretrained(cfg.sft_ckpt)
    tok.save_pretrained(cfg.sft_ckpt)
    elapsed = time.time() - phase_t0
    print(f"SFT adapter saved → {cfg.sft_ckpt}  ({elapsed:.0f}s)")
    free_gpu(model)
    return elapsed


# --- GRPO dataset + reward (for TRL GRPOTrainer) ---

def make_grpo_dataset(data: list) -> HFDataset:
    """Dataset with 'prompt' column for generation and 'answer' for reward."""
    rows = [{"prompt": make_prompt(ex), "answer": ex["answer"].strip()}
            for ex in data]
    return HFDataset.from_list(rows)


# --- Reward shaping (ablation variants) ---

def _reward_components(output: str, gt_answer: str) -> dict:
    """Get the raw components (correct, format, length) for a single output."""
    return {
        "correct": 1.0 if _is_correct(output, gt_answer) else 0.0,
        "format":  1.0 if _format_ok(output, gt_answer)  else 0.0,
        "length":  float(len(output.split())),
    }


def compute_shaped_reward(
    output: str,
    gt_answer: str,
    cfg: "PilotConfig",
) -> Tuple[float, dict]:
    """
    Compute reward based on cfg.reward_type:
      binary         -> 1/0
      format_bonus   -> correct + beta * format_ok
      length_penalty -> correct - lambda * word_count
    Returns (reward, breakdown_dict).
    """
    bd = _reward_components(output, gt_answer)
    correct = bd["correct"]
    fmt     = bd["format"]
    length  = bd["length"]

    if cfg.reward_type == "binary":
        reward = correct

    elif cfg.reward_type == "format_bonus":
        reward = correct + cfg.format_bonus_value * fmt

    elif cfg.reward_type == "length_penalty":
        reward = correct - cfg.length_penalty_lambda * length

    else:
        raise ValueError(
            f"Unknown reward_type '{cfg.reward_type}'. "
            "Valid choices: binary | format_bonus | length_penalty"
        )

    bd["reward"] = reward
    return reward, bd


def make_grpo_reward_fn(cfg: "PilotConfig"):
    """Build a reward function for TRL's GRPOTrainer. Also logs stats to grpo_reward_log.jsonl."""
    log_path  = os.path.join(cfg.results_dir, "grpo_reward_log.jsonl")
    call_idx  = [0]   # mutable counter captured by closure

    def _reward_fn(
        completions: List[str],
        answer: List[str],
        **kwargs,
    ) -> List[float]:
        rewards, corrects, fmts, lengths = [], [], [], []
        for c, a in zip(completions, answer):
            r, bd = compute_shaped_reward(c, a, cfg)
            rewards.append(r)
            corrects.append(bd["correct"])
            fmts.append(bd["format"])
            lengths.append(bd["length"])

        call_idx[0] += 1
        n = len(rewards)
        log_jsonl(log_path, {
            "call":                   call_idx[0],
            "reward_type":            cfg.reward_type,
            "format_bonus_value":     cfg.format_bonus_value     if cfg.reward_type == "format_bonus"   else None,
            "length_penalty_lambda":  cfg.length_penalty_lambda  if cfg.reward_type == "length_penalty" else None,
            "avg_reward":             sum(rewards)  / n,
            "avg_correct":            sum(corrects) / n,
            "avg_format":             sum(fmts)     / n,
            "avg_length":             sum(lengths)  / n,
            "n":                      n,
        })
        return rewards

    return _reward_fn


# --- GRPO training ---

def train_grpo(cfg: PilotConfig) -> float:
    """Train GRPO. Returns wall-clock seconds elapsed."""
    print("\n" + "═" * 60)
    print("PHASE – GRPO Training  (TRL GRPOTrainer)")
    print("═" * 60)
    phase_t0 = time.time()
    set_seed(cfg.seed)

    train_data, _ = load_data(cfg)
    grpo_data = _slice(train_data, cfg.grpo_train_size)
    tok     = load_tokenizer(cfg)
    model   = load_base_model(cfg)
    dataset = make_grpo_dataset(grpo_data)

    # Start from SFT adapter weights
    model = PeftModel.from_pretrained(model, cfg.sft_ckpt, is_trainable=True)
    model.config.use_cache = False
    model.enable_input_require_grads()

    # Log reward configuration so every run is self-documenting.
    reward_cfg_record = {
        "reward_type":            cfg.reward_type,
        "format_bonus_value":     cfg.format_bonus_value,
        "length_penalty_lambda":  cfg.length_penalty_lambda,
    }
    save_json(os.path.join(cfg.results_dir, "grpo_reward_config.json"), reward_cfg_record)
    print(f"  Reward config: {reward_cfg_record}")
    print(f"  GRPO dataset size: {len(grpo_data):,}")

    use_bf16 = torch.cuda.is_bf16_supported()

    grpo_args = GRPOConfig(
        output_dir=cfg.grpo_ckpt,
        max_steps=cfg.grpo_max_steps,
        per_device_train_batch_size=cfg.grpo_batch_prompts,
        learning_rate=cfg.grpo_lr,
        # GRPO-specific
        num_generations=cfg.grpo_K,
        max_prompt_length=512,
        max_completion_length=cfg.grpo_max_new_tokens,
        temperature=cfg.grpo_temperature,
        beta=cfg.grpo_beta_kl,
        # keep dataset columns so reward_fn receives 'answer'
        remove_unused_columns=False,
        # compute
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        logging_steps=20,
        save_strategy="no",
        seed=cfg.seed,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=dataset,
        reward_funcs=[make_grpo_reward_fn(cfg)],   # shaped reward (default: binary)
        processing_class=tok,
    )

    trainer.train()

    os.makedirs(cfg.grpo_ckpt, exist_ok=True)
    trainer.save_model(cfg.grpo_ckpt)
    tok.save_pretrained(cfg.grpo_ckpt)
    elapsed = time.time() - phase_t0
    print(f"GRPO adapter saved → {cfg.grpo_ckpt}  ({elapsed:.0f}s)")
    free_gpu(model)
    return elapsed


# --- Evaluation ---

@torch.no_grad()
def run_eval(cfg: PilotConfig, phase: str, ckpt: Optional[str] = None) -> dict:
    """Run eval: greedy accuracy + pass@k + format success + avg length."""
    print("\n" + "═" * 60)
    print(f"PHASE – Eval {phase.upper()}")
    print("═" * 60)
    phase_t0 = time.time()

    _, eval_data = load_data(cfg)
    set_seed(cfg.seed)
    tok   = load_tokenizer(cfg)
    model = load_for_inference(cfg, ckpt)
    dev   = device_of(model)

    # greedy decode pass
    correct = fmt_ok = total = 0
    total_reward = total_length = 0.0
    qual: list = []

    for i in range(0, len(eval_data), cfg.eval_batch_size):
        batch   = eval_data[i : i + cfg.eval_batch_size]
        prompts = [make_prompt(ex) for ex in batch]
        enc     = tok(prompts, padding=True, truncation=True,
                      max_length=512, return_tensors="pt").to(dev)
        out     = model.generate(
            **enc,
            max_new_tokens=cfg.eval_max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        decoded = tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True)

        for ex, dec in zip(batch, decoded):
            gt = ex["answer"]
            if _is_correct(dec, gt):  correct += 1
            if _format_ok(dec, gt):   fmt_ok  += 1
            total += 1
            r_shaped, bd = compute_shaped_reward(dec, gt, cfg)
            total_reward += r_shaped
            total_length += bd["length"]
            if len(qual) < 10:
                qual.append({
                    "phase":    phase,
                    "dataset":  ex.get("dataset", "gsm8k"),
                    "question": ex["question"][:200],
                    "gt":       gt[:200],
                    "output":   dec[:400],
                    "pred":     _parse_boxed(dec) or _parse_answer(dec),
                    "correct":  _is_correct(dec, gt),
                })

        if (i // cfg.eval_batch_size + 1) % 5 == 0:
            print(f"  … greedy {total}/{len(eval_data)}")

    accuracy       = correct / max(total, 1)
    format_success = fmt_ok  / max(total, 1)
    avg_length     = total_length / max(total, 1)
    avg_reward     = total_reward / max(total, 1)

    print(f"  [Greedy]  Accuracy={accuracy:.3f}  "
          f"FormatSuccess={format_success:.3f}  "
          f"AvgLen={avg_length:.1f}w")

    # pass@k via stochastic sampling
    pass_at_k_rate = None
    k = cfg.pass_at_k
    if k > 1:
        pass_correct = 0
        for i in range(0, len(eval_data), cfg.eval_batch_size):
            batch   = eval_data[i : i + cfg.eval_batch_size]
            B       = len(batch)
            prompts = [make_prompt(ex) for ex in batch]
            enc     = tok(prompts, padding=True, truncation=True,
                          max_length=512, return_tensors="pt").to(dev)
            # Generate k samples per problem in a single forward pass
            out = model.generate(
                **enc,
                max_new_tokens=cfg.eval_max_new_tokens,
                do_sample=True,
                temperature=cfg.eval_temperature,
                top_p=0.95,
                num_return_sequences=k,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
            # out shape: (B*k, seq_len) — grouped as [ex0_s0, ex0_s1, ..., ex1_s0, ...]
            p_len   = enc["input_ids"].shape[1]
            decoded = tok.batch_decode(out[:, p_len:], skip_special_tokens=True)

            for j, ex in enumerate(batch):
                gt      = ex["answer"]
                samples = decoded[j * k : (j + 1) * k]
                if any(_is_correct(s, gt) for s in samples):
                    pass_correct += 1

            if (i // cfg.eval_batch_size + 1) % 5 == 0:
                processed = min(i + cfg.eval_batch_size, len(eval_data))
                print(f"  … pass@{k} {processed}/{len(eval_data)}")

        pass_at_k_rate = pass_correct / max(total, 1)
        print(f"  [pass@{k}] {pass_at_k_rate:.3f}")

    eval_elapsed = time.time() - phase_t0

    # put metrics together
    metrics = {
        "phase":              phase,
        "dataset":            cfg.dataset,
        # primary
        "accuracy":           accuracy,
        # secondary
        "pass_at_k":          pass_at_k_rate,
        "k":                  k,
        "format_success":     format_success,
        "avg_output_length":  avg_length,
        # bookkeeping
        "correct":            correct,
        "total":              total,
        "reward_type":        cfg.reward_type,
        "avg_reward":         avg_reward,
        "runtime_seconds":    round(eval_elapsed, 1),
    }

    save_json(f"{cfg.results_dir}/{phase}_metrics.json", metrics)
    qpath = f"{cfg.results_dir}/qualitative.jsonl"
    with open(qpath, "a") as f:
        for q in qual:
            f.write(json.dumps(q) + "\n")

    free_gpu(model)
    return metrics


# --- System info ---

def _run(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "n/a"


def collect_system_info(cfg: PilotConfig) -> dict:
    """Grab GPU/CPU/RAM/software info and save it for the writeup."""
    info: dict = {}

    # cpu
    info["cpu_physical_cores"]  = multiprocessing.cpu_count()
    info["cpu_model"]           = _run(
        "lscpu | grep 'Model name' | awk -F: '{print $2}' | xargs"
    )

    # ram
    mem_kb = _run("grep MemTotal /proc/meminfo | awk '{print $2}'")
    try:
        info["ram_gb"] = round(int(mem_kb) / 1024 / 1024, 1)
    except ValueError:
        info["ram_gb"] = "n/a"

    # gpu
    if torch.cuda.is_available():
        gpus = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpus.append({
                "index":        i,
                "name":         p.name,
                "vram_gb":      round(p.total_memory / 1024**3, 1),
                "sm_count":     p.multi_processor_count,
                "cuda_cap":     f"{p.major}.{p.minor}",
            })
        info["gpus"]       = gpus
        info["cuda_version"] = torch.version.cuda
        info["num_gpus"]   = len(gpus)
    else:
        info["gpus"]       = []
        info["cuda_version"] = "n/a"
        info["num_gpus"]   = 0

    # software versions
    info["python_version"]  = platform.python_version()
    info["torch_version"]   = torch.__version__
    info["platform"]        = platform.platform()
    info["hostname"]        = platform.node()

    # experiment config snapshot
    info["experiment"] = {
        "model_name":    cfg.model_name,
        "dataset":       cfg.dataset,
        "mode":          cfg.mode,
        "train_size":    cfg.train_size,
        "eval_size":     cfg.eval_size,
        "grpo_train_size": cfg.grpo_train_size,
        "reward_type":   cfg.reward_type,
        "pass_at_k":     cfg.pass_at_k,
        "sft_max_steps": cfg.sft_max_steps,
        "grpo_max_steps": cfg.grpo_max_steps,
        "grpo_K":        cfg.grpo_K,
        "lora_r":        cfg.lora_r,
        "lora_alpha":    cfg.lora_alpha,
    }

    return info


def _print_system_info(info: dict) -> None:
    print("\n" + "═" * 60)
    print("EXPERIMENTAL SETUP")
    print("═" * 60)
    print(f"  Host:        {info['hostname']}")
    print(f"  OS:          {info['platform']}")
    print(f"  Python:      {info['python_version']}")
    print(f"  PyTorch:     {info['torch_version']}  (CUDA {info['cuda_version']})")
    print(f"  CPU cores:   {info['cpu_physical_cores']}  –  {info['cpu_model']}")
    print(f"  RAM:         {info['ram_gb']} GB")
    if info["gpus"]:
        for g in info["gpus"]:
            print(f"  GPU [{g['index']}]:    {g['name']}  {g['vram_gb']} GB VRAM  "
                  f"{g['sm_count']} SMs  CUDA {g['cuda_cap']}")
    else:
        print("  GPU:         none detected")
    exp = info["experiment"]
    print(f"  Model:       {exp['model_name']}")
    print(f"  Dataset:     {exp['dataset']}  "
          f"(train={exp['train_size']}, eval={exp['eval_size']})")
    print(f"  Mode:        {exp['mode']}")
    print(f"  Reward:      {exp['reward_type']}")
    print(f"  pass@k:      k={exp['pass_at_k']}")
    print("═" * 60)


# --- Main ---

def main() -> None:
    cfg = build_config()
    set_seed(cfg.seed)
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not found — this will be extremely slow.")

    # log system info before we do anything
    sys_info = collect_system_info(cfg)
    save_json(f"{cfg.results_dir}/system_info.json", sys_info)
    _print_system_info(sys_info)

    print("\n" + "═" * 60)
    print(f"GSM8K/MATH  SFT vs GRPO  Pilot  (dataset={cfg.dataset})")
    print("═" * 60)
    print(json.dumps(asdict(cfg), indent=2))

    results: dict = {}
    runtimes: dict = {}
    total_t0 = time.time()

    if cfg.mode in ("eval_base", "all"):
        results["base"] = run_eval(cfg, phase="base", ckpt=None)
        runtimes["eval_base"] = results["base"]["runtime_seconds"]

    if cfg.mode in ("train_sft", "all"):
        runtimes["train_sft"] = round(train_sft(cfg), 1)

    if cfg.mode in ("eval_sft", "all"):
        results["sft"] = run_eval(cfg, phase="sft", ckpt=cfg.sft_ckpt)
        runtimes["eval_sft"] = results["sft"]["runtime_seconds"]

    if cfg.mode in ("train_grpo", "all"):
        runtimes["train_grpo"] = round(train_grpo(cfg), 1)

    if cfg.mode in ("eval_grpo", "all"):
        results["grpo"] = run_eval(cfg, phase="grpo", ckpt=cfg.grpo_ckpt)
        runtimes["eval_grpo"] = results["grpo"]["runtime_seconds"]

    runtimes["total"] = round(time.time() - total_t0, 1)
    save_json(f"{cfg.results_dir}/runtimes.json", runtimes)

    if results:
        save_json(f"{cfg.results_dir}/summary.json", results)
        print("\n" + "═" * 60)
        print("FINAL SUMMARY")
        print("═" * 60)
        k = cfg.pass_at_k
        for phase, m in results.items():
            pak = f"  pass@{k}={m['pass_at_k']:.3f}" if m.get("pass_at_k") is not None else ""
            rt  = f"  {m['runtime_seconds']}s" if m.get("runtime_seconds") else ""
            print(f"  {phase.upper():8s}  "
                  f"acc={m['accuracy']:.3f}  "
                  f"fmt={m['format_success']:.3f}  "
                  f"len={m['avg_output_length']:.1f}w"
                  f"{pak}  "
                  f"({m['correct']}/{m['total']})"
                  f"{rt}")
        print(f"\n  Runtimes: {json.dumps(runtimes)}")
        print("═" * 60)


if __name__ == "__main__":
    main()
