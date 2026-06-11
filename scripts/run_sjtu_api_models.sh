#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-outputs/sjtu_api_models}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
API_MODELS="${API_MODELS:-deepseek-reasoner deepseek-chat glm-5 minimax-m2.5 qwen3coder qwen3vl}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME

if [[ -z "${SJTU_API_KEY:-}" ]]; then
  echo "Missing SJTU_API_KEY. Export it in the shell or load it from a private .env file." >&2
  exit 2
fi

for model in $API_MODELS; do
  safe_model="${model//[^A-Za-z0-9_.-]/_}"
  echo "==> running SJTU API search for ${model}" >&2
  "$PYTHON_BIN" src/api_attack.py \
    --base-url "${SJTU_API_BASE_URL:-https://models.sjtu.edu.cn/api/v1}" \
    --api-model "$model" \
    --api-key-env SJTU_API_KEY \
    --embedding-model "${EMBEDDING_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-4096}" \
    --max-calls "${MAX_CALLS:-6}" \
    --min-request-interval "${MIN_REQUEST_INTERVAL:-6.5}" \
    --request-timeout "${REQUEST_TIMEOUT:-240}" \
    --retry-attempts "${RETRY_ATTEMPTS:-1}" \
    --low-dim "${LOW_DIM:-8}" \
    --prompt-len "${PROMPT_LEN:-20}" \
    --seed "${SEED:-20260611}" \
    --output-dir "$OUT_ROOT/$safe_model"
done
