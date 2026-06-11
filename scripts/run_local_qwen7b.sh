#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-qwen7b_local_example}"
OUT_ROOT="${OUT_ROOT:-outputs}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME

python src/local_repetition_attack.py \
  --model "${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-1024}" \
  --max-evals "${MAX_EVALS:-16}" \
  --low-dim "${LOW_DIM:-8}" \
  --prompt-len "${PROMPT_LEN:-20}" \
  --output-dir "$OUT_ROOT/$RUN_NAME"
