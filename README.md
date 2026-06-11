# Thinktrap

This toolkit searches for short prompts that induce unusually long or repetitive LLM outputs. It supports local Hugging Face causal language models and OpenAI-compatible chat APIs.

This repository contains only source code and runnable examples. It does not include model weights, API keys, full generated outputs, saved prompts, or experiment result dumps.

## Layout

```text
src/
  local_repetition_attack.py  # Local Hugging Face model search
  api_attack.py               # OpenAI-compatible API search
  thinktrap_common.py         # Shared generation and repetition metrics
scripts/
  run_local_qwen7b.sh         # Local model example
```

## Install

```bash
pip install -r requirements.txt
```

If Hugging Face downloads are slow, set a mirror and cache path before running:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/path/to/hf_cache
```

## Local Model Example

Run a local search on a Hugging Face causal LM:

```bash
CUDA_VISIBLE_DEVICES=0 python src/local_repetition_attack.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --max-new-tokens 1024 \
  --max-evals 16 \
  --low-dim 8 \
  --prompt-len 20 \
  --output-dir outputs/qwen7b_1024
```

For a smaller model with a longer generation cap:

```bash
CUDA_VISIBLE_DEVICES=0 python src/local_repetition_attack.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --max-new-tokens 4096 \
  --max-evals 16 \
  --low-dim 8 \
  --prompt-len 20 \
  --output-dir outputs/qwen15b_4096
```

The wrapper script exposes the same path with environment variables:

```bash
CUDA_VISIBLE_DEVICES=0 MAX_NEW_TOKENS=1024 MAX_EVALS=16 bash scripts/run_local_qwen7b.sh
```

If `cma` is unavailable, the local script falls back to random sampling in the same projected prompt space.

## API Example

`api_attack.py` uses a local tokenizer/embedding model to map continuous optimization vectors back to text prompts, then evaluates those prompts through an OpenAI-compatible API endpoint.

```bash
export API_KEY=...
python src/api_attack.py \
  --base-url https://example.com/api/v1 \
  --api-model model-name \
  --api-key-env API_KEY \
  --embedding-model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --max-new-tokens 4096 \
  --max-calls 40 \
  --low-dim 8 \
  --prompt-len 20 \
  --output-dir outputs/api_model
```

## Outputs

Each run writes a JSONL file with prompts, generated text, and metrics; a CSV file with compact previews; and a metadata JSON file with aggregate run settings and summary statistics. Full outputs can be large and should normally stay out of the repository.

## Citation

```bibtex
@article{li2026availability,
  title={On the Availability Risks of Production LLM Services under Unbounded Inference},
  author={Li, Yunzhe and Wang, Jianan and Zhu, Hongzi and Lin, James and Chang, Shan and Guo, Minyi},
  journal={IEEE Transactions on Dependable and Secure Computing},
  year={2026}
}

@inproceedings{li2026thinktrap,
  title={ThinkTrap: Denial-of-Service Attacks against Black-box LLM Services via Infinite Thinking},
  author={Li, Yunzhe and Wang, Jianan and Zhu, Hongzi and Lin, James and Chang, Shan and Guo, Minyi},
  booktitle={Proceedings of the Network and Distributed System Security Symposium (NDSS)},
  year={2026}
}
```

## Safety

Do not commit `.env`, API keys, model weights, or generated full-output logs. `api_attack.py` reads keys only from environment variables.
