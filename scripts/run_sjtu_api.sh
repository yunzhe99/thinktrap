#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-sjtu_api_example}"
OUT_ROOT="${OUT_ROOT:-outputs}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME

if [[ -z "${SJTU_API_KEY:-}" ]]; then
  echo "Missing SJTU_API_KEY. Export it in the shell or load it from a private .env file." >&2
  exit 2
fi

"$PYTHON_BIN" src/api_attack.py \
  --base-url "${SJTU_API_BASE_URL:-https://models.sjtu.edu.cn/api/v1}" \
  --api-model "${SJTU_API_MODEL:-deepseek-reasoner}" \
  --api-key-env SJTU_API_KEY \
  --embedding-model "${EMBEDDING_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-4096}" \
  --max-calls "${MAX_CALLS:-40}" \
  --min-request-interval "${MIN_REQUEST_INTERVAL:-6.5}" \
  --request-timeout "${REQUEST_TIMEOUT:-180}" \
  --retry-attempts "${RETRY_ATTEMPTS:-1}" \
  --low-dim "${LOW_DIM:-8}" \
  --prompt-len "${PROMPT_LEN:-20}" \
  --output-dir "$OUT_ROOT/$RUN_NAME"
