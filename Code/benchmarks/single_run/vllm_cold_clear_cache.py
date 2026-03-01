"""
Cold-start vLLM benchmark with OS page-cache clearing between iterations.

After each iteration the model weights are evicted from the OS page cache via
posix_fadvise(POSIX_FADV_DONTNEED) — no root/sudo required. This ensures
every iteration reads weights from disk rather than from RAM, giving a true
cold-start measurement.

Output CSV columns: iteration, load_time, infer_time, total_time, throughput, ttft

Usage (run from benchmarks/):
  python single_run/vllm_cold_clear_cache.py --model EleutherAI/gpt-j-6b
  python single_run/vllm_cold_clear_cache.py --model EleutherAI/gpt-j-6b \
      --iterations 5 --output_csv results/EleutherAI/gpt-j-6b/true_cold.csv
"""

import argparse
import ctypes
import ctypes.util
import gc
import os
import sys
import time
from pathlib import Path
from timeit import default_timer as timer

import pandas as pd
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

DEFAULT_NR_INPUT_TOKENS = 128
DEFAULT_NR_OUTPUT_TOKENS = 128
DEFAULT_MEM_UTIL = 0.85
DEFAULT_ITERATIONS = 5

# ---------------------------------------------------------------------------
# Page-cache eviction — no sudo required
# ---------------------------------------------------------------------------
_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
POSIX_FADV_DONTNEED = 4  # Linux


def _drop_file(path: Path):
    size = path.stat().st_size
    with open(path, "rb") as f:
        ret = _libc.posix_fadvise(f.fileno(), 0, size, POSIX_FADV_DONTNEED)
        if ret != 0:
            print(f"  [warn] posix_fadvise failed for {path.name}: errno={ctypes.get_errno()}")


def drop_model_from_page_cache(model_name: str, cache_dir: str | None = None):
    if cache_dir is None:
        cache_dir = os.environ.get(
            "HF_HOME",
            os.environ.get("TRANSFORMERS_CACHE",
                           os.path.expanduser("~/.cache/huggingface/hub")),
        )
    model_dir = Path(cache_dir) / ("models--" + model_name.replace("/", "--"))
    if not model_dir.exists():
        print(f"  [warn] model cache not found at {model_dir}")
        return
    dropped = 0
    for ext in ("*.safetensors", "*.bin", "*.pt"):
        for f in model_dir.rglob(ext):
            _drop_file(f)
            dropped += 1
    print(f"  dropped {dropped} weight file(s) from page cache")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_prompt(nr_tokens, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    text = "just some text as input for benchmarking"
    repeat = nr_tokens // len(tokenizer.encode(text)) + 1
    tokens = tokenizer.encode(text * repeat)[: nr_tokens - 1]
    if len(tokens) < nr_tokens:
        tokens += [tokenizer.pad_token_id] * (nr_tokens - len(tokens) - 1)
    return [tokenizer.decode(tokens, skip_special_tokens=True)]


def run_iteration(model_name, prompt, nr_output, memory_rate):
    torch.cuda.empty_cache()
    gc.collect()

    t0 = timer()
    llm = LLM(model_name, gpu_memory_utilization=memory_rate)
    load_time = timer() - t0

    sampling_params = SamplingParams(max_tokens=nr_output, temperature=0)
    t1 = timer()
    outputs = llm.generate(prompt, sampling_params)
    infer_time = timer() - t1

    tokens_out = sum(len(o.outputs[0].token_ids) for o in outputs if o.outputs)
    throughput = tokens_out / infer_time if infer_time > 0 else 0
    ttft = infer_time / len(outputs)

    del llm
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "load_time": load_time,
        "infer_time": infer_time,
        "total_time": load_time + infer_time,
        "throughput": throughput,
        "ttft": ttft,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(model_name, iterations, nr_input_tokens, nr_output_tokens,
        memory_rate, output_csv, cache_dir):
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    prompt = make_prompt(nr_input_tokens, model_name)

    rows = []
    for i in range(iterations):
        print(f"\n--- Iteration {i + 1}/{iterations} ---")
        res = run_iteration(model_name, prompt, nr_output_tokens, memory_rate)
        print(f"  load: {res['load_time']:.3f}s | infer: {res['infer_time']:.3f}s "
              f"| throughput: {res['throughput']:.1f} tok/s")
        rows.append({"iteration": i + 1, **res})

        if i < iterations - 1:
            print("  clearing page cache...")
            drop_model_from_page_cache(model_name, cache_dir)
            time.sleep(0.5)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cold-start vLLM with page-cache clearing between iterations"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--output_csv", type=str, default="true_cold.csv")
    parser.add_argument("--nr_input_tokens", type=int, default=DEFAULT_NR_INPUT_TOKENS)
    parser.add_argument("--nr_output_tokens", type=int, default=DEFAULT_NR_OUTPUT_TOKENS)
    parser.add_argument("--memory_rate", type=float, default=DEFAULT_MEM_UTIL,
                        help="vLLM gpu_memory_utilization (default: 0.85)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HuggingFace cache dir (default: auto-detect from "
                             "HF_HOME / TRANSFORMERS_CACHE / ~/.cache/huggingface/hub)")

    args = parser.parse_args()
    run(args.model, args.iterations, args.nr_input_tokens, args.nr_output_tokens,
        args.memory_rate, args.output_csv, args.cache_dir)
