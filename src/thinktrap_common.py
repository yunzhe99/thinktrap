import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_causal_lm(model_name, local_files_only=False):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    model.eval()
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.pad_token_id
    return tokenizer, model


def to_chat_prompt(tokenizer, text):
    messages = [{"role": "user", "content": text}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return text


def generate_once(args, tokenizer, model, prompt_text):
    model_prompt = to_chat_prompt(tokenizer, prompt_text)
    inputs = tokenizer(model_prompt, return_tensors="pt").to(model.device)
    started = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - started

    input_tokens = int(inputs["input_ids"].shape[1])
    output_tokens = int(outputs.shape[1] - input_tokens)
    generated_ids = outputs[0, input_tokens:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    metrics = repetition_metrics(
        generated_ids.tolist(),
        generated_text,
        args.max_new_tokens,
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        **metrics,
        "elapsed_sec": round(elapsed, 4),
        "prompt_text": prompt_text,
        "generated_text": generated_text,
    }


def max_ngram_count(token_ids, n):
    if len(token_ids) < n:
        return 0
    counts = {}
    for i in range(len(token_ids) - n + 1):
        key = tuple(token_ids[i : i + n])
        counts[key] = counts.get(key, 0) + 1
    return max(counts.values()) if counts else 0


def longest_repeated_text_count(text, width=24):
    compact = " ".join(text.split())
    if len(compact) < width:
        return 0
    counts = {}
    step = max(1, width // 3)
    for i in range(0, len(compact) - width + 1, step):
        key = compact[i : i + width]
        counts[key] = counts.get(key, 0) + 1
    return max(counts.values()) if counts else 0


def repetition_metrics(generated_token_ids, generated_text, max_new_tokens):
    output_tokens = len(generated_token_ids)
    tail = generated_token_ids[-min(96, output_tokens) :] if output_tokens else []
    tail_unique_ratio = (len(set(tail)) / len(tail)) if tail else 1.0
    max_4gram = max_ngram_count(generated_token_ids, 4)
    max_8gram = max_ngram_count(generated_token_ids, 8)
    text_repeat_count = longest_repeated_text_count(generated_text)
    hit_limit = output_tokens >= max_new_tokens
    repetitive = bool(
        output_tokens >= min(64, max_new_tokens)
        and (
            tail_unique_ratio <= 0.35
            or max_4gram >= 5
            or max_8gram >= 3
            or text_repeat_count >= 4
        )
    )
    return {
        "hit_limit": hit_limit,
        "repetitive": repetitive,
        "tail_unique_ratio": round(tail_unique_ratio, 4),
        "max_4gram_count": int(max_4gram),
        "max_8gram_count": int(max_8gram),
        "text_repeat_count": int(text_repeat_count),
    }


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
