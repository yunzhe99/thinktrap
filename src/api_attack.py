import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from thinktrap_common import repetition_metrics

try:
    import cma
except ImportError:
    cma = None


DEFAULT_BUDGETS = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]


def parse_args():
    parser = argparse.ArgumentParser(
        description="CMA-ES output-length attack for OpenAI-compatible chat APIs."
    )
    parser.add_argument("--api-model", default="deepseek/deepseek-r1")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument(
        "--embedding-model",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Local HF model used only to map continuous embeddings to text prompts.",
    )
    parser.add_argument("--prompt-len", type=int, default=20)
    parser.add_argument("--low-dim", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-calls", type=int, default=200)
    parser.add_argument(
        "--min-request-interval",
        type=float,
        default=0.0,
        help="Minimum seconds between API request starts. Use this for rate-limited services.",
    )
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--retry-attempts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--output-dir", default="outputs/api_attack")
    return parser.parse_args()


def project_low_dim(proj, low_dim, prompt_len, z):
    z = z.reshape((low_dim, prompt_len))
    return (proj @ z).T


def embedding_to_text_nearest_neighbor(tokenizer, vocab_embeddings, embedding):
    tensor = torch.tensor(embedding, dtype=torch.float32)
    similarities = torch.matmul(tensor, vocab_embeddings.T)
    token_ids = similarities.argmax(dim=-1)
    return tokenizer.decode(token_ids.tolist(), skip_special_tokens=True).strip()


def normalize_message_part(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(value)


def call_chat(client, args, prompt_text):
    completion = client.chat.completions.create(
        model=args.api_model,
        messages=[{"role": "user", "content": prompt_text}],
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    usage = getattr(completion, "usage", None)
    choice = completion.choices[0] if getattr(completion, "choices", None) else None
    message = getattr(choice, "message", None) if choice else None
    content_text = normalize_message_part(getattr(message, "content", ""))
    reasoning_text = normalize_message_part(getattr(message, "reasoning_content", ""))
    generated_text = "\n".join(part for part in [reasoning_text, content_text] if part).strip()
    completion_tokens = getattr(usage, "completion_tokens", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    finish_reason = getattr(choice, "finish_reason", None) if choice else None
    return {
        "output_tokens": int(completion_tokens) if completion_tokens is not None else None,
        "prompt_tokens": int(prompt_tokens) if prompt_tokens is not None else None,
        "api_total_tokens": int(total_tokens) if total_tokens is not None else None,
        "finish_reason": finish_reason,
        "content_text": content_text,
        "reasoning_text": reasoning_text,
        "generated_text": generated_text,
    }


def call_chat_with_retries(client, args, prompt_text):
    attempts = max(1, args.retry_attempts)
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return call_chat(client, args, prompt_text), None
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < attempts:
                time.sleep(min(30.0, 2.0 * attempt))
    return None, last_error


def main():
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key. Set {args.api_key_env}=...")
    if cma is None:
        raise SystemExit("The API attack requires cma. Install it with `pip install cma`.")

    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "api_attack.jsonl"
    meta_path = out_dir / "api_attack_meta.json"

    client = OpenAI(base_url=args.base_url, api_key=api_key, timeout=args.request_timeout)
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.embedding_model, trust_remote_code=True)
    vocab_embeddings = model.get_input_embeddings().weight.detach()
    embedding_dim = int(vocab_embeddings.shape[1])

    proj = np.random.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(args.low_dim),
        size=(embedding_dim, args.low_dim),
    )
    init_z = np.random.uniform(-0.5, 0.5, size=(args.low_dim, args.prompt_len))
    es = cma.CMAEvolutionStrategy(
        init_z.flatten(),
        0.5,
        {"bounds": [-5, 5], "verbose": -9},
    )

    best = None
    calls = 0
    total_tokens = 0
    budget_idx = 0
    budget_best = {budget: 0 for budget in DEFAULT_BUDGETS}
    last_request_start = None

    with jsonl_path.open("w", encoding="utf-8") as out_jsonl:
        while calls < args.max_calls and budget_idx < len(DEFAULT_BUDGETS):
            solutions = es.ask()
            fitnesses = []
            generation = []
            for z in solutions:
                if calls >= args.max_calls:
                    break
                embedding = project_low_dim(proj, args.low_dim, args.prompt_len, z)
                prompt_text = embedding_to_text_nearest_neighbor(
                    tokenizer,
                    vocab_embeddings,
                    embedding,
                )
                if args.min_request_interval > 0 and last_request_start is not None:
                    elapsed_since_request = time.monotonic() - last_request_start
                    sleep_seconds = args.min_request_interval - elapsed_since_request
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
                last_request_start = time.monotonic()
                api_result, error = call_chat_with_retries(client, args, prompt_text)
                calls += 1
                if api_result is None:
                    output_tokens = 0
                    metrics = repetition_metrics([], "", args.max_new_tokens)
                    row = {
                        "call_idx": calls,
                        "prompt_text": prompt_text,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                        "error": error,
                        **metrics,
                    }
                else:
                    generated_text = api_result["generated_text"]
                    generated_token_ids = tokenizer.encode(generated_text, add_special_tokens=False)
                    output_tokens = api_result["output_tokens"]
                    if output_tokens is None:
                        output_tokens = len(generated_token_ids)
                    token_cost = api_result["api_total_tokens"]
                    if token_cost is None:
                        token_cost = (api_result["prompt_tokens"] or len(prompt_text)) + output_tokens
                    total_tokens += int(token_cost)
                    metrics = repetition_metrics(generated_token_ids, generated_text, args.max_new_tokens)
                    row = {
                        "call_idx": calls,
                        "prompt_text": prompt_text,
                        "output_tokens": int(output_tokens),
                        "total_tokens": total_tokens,
                        "prompt_tokens": api_result["prompt_tokens"],
                        "api_total_tokens": api_result["api_total_tokens"],
                        "finish_reason": api_result["finish_reason"],
                        "content_text": api_result["content_text"],
                        "reasoning_text": api_result["reasoning_text"],
                        "generated_text": generated_text,
                        **metrics,
                    }
                out_jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_jsonl.flush()
                print(
                    "call={calls} output_tokens={output_tokens} "
                    "repetitive={repetitive} hit_limit={hit_limit} total_tokens={total_tokens}"
                    "{error}".format(
                        calls=calls,
                        output_tokens=output_tokens,
                        repetitive=row["repetitive"],
                        hit_limit=row["hit_limit"],
                        total_tokens=total_tokens,
                        error=f" error={error}" if error else "",
                    ),
                    flush=True,
                )
                generation.append((z, output_tokens))
                fitnesses.append(-output_tokens)
                if best is None or output_tokens > best["output_tokens"]:
                    best = row

                while budget_idx < len(DEFAULT_BUDGETS) and total_tokens >= DEFAULT_BUDGETS[budget_idx]:
                    budget = DEFAULT_BUDGETS[budget_idx]
                    budget_best[budget] = best["output_tokens"] if best else 0
                    budget_idx += 1

            if len(generation) == len(solutions):
                es.tell([item[0] for item in generation], fitnesses)

    meta = {
        "api_model": args.api_model,
        "base_url": args.base_url,
        "embedding_model": args.embedding_model,
        "prompt_len": args.prompt_len,
        "low_dim": args.low_dim,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "min_request_interval": args.min_request_interval,
        "request_timeout": args.request_timeout,
        "retry_attempts": args.retry_attempts,
        "max_calls": args.max_calls,
        "calls": calls,
        "total_tokens": total_tokens,
        "best": best,
        "budget_best": budget_best,
        "jsonl_path": str(jsonl_path),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("summary=" + json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
