#!/bin/bash
# download_models.sh - run on login node to pre-cache models/datasets
# usage: bash jobs/download_models.sh

source /share/pkg.8/miniconda/24.5.0/install/etc/profile.d/conda.sh
conda activate /projectnb/vkolagrp/kelzokm/conda_envs/grpo
export PYTHONNOUSERSITE=1

python -c "
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset

# Models to cache
models = [
    'Qwen/Qwen2.5-3B-Instruct',
    'Qwen/Qwen2.5-1.5B-Instruct',
    'meta-llama/Llama-3.2-1B-Instruct',
    'meta-llama/Llama-3.2-3B-Instruct',
]

for m in models:
    print(f'Caching {m} ...')
    try:
        AutoConfig.from_pretrained(m)
        AutoTokenizer.from_pretrained(m)
        print(f'  OK: {m}')
    except Exception as e:
        print(f'  FAILED: {m}: {e}')

# Datasets
print('Caching GSM8K ...')
load_dataset('gsm8k', 'main')
print('  OK')

print('Caching MATH ...')
load_dataset('DigitalLearningGmbH/MATH-lighteval', 'default')
print('  OK')

print('Done.')
"
