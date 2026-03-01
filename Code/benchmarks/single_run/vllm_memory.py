"""
Single-run vLLM memory profiler.

Loads the model once, runs one inference, and captures GPU memory usage
over time via the GpuWatcher (nvidia-smi at 10 ms intervals).

Usage:
  python vllm_memory.py --model EleutherAI/gpt-j-6b
  python vllm_memory.py --model EleutherAI/gpt-j-6b --output_csv out.csv \
      --nr_input_tokens 256 --nr_output_tokens 128 --memory_rate 0.85
"""

import argparse
import gc
import os
import sys
import time
from timeit import default_timer as timer

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch

# resolve watcher from parent benchmarks/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from watcher import GpuWatcher


DEFAULT_NR_INPUT_TOKENS = 128
DEFAULT_NR_OUTPUT_TOKENS = 128
DEFAULT_MEM_UTIL = 0.85


def make_prompt(nr_tokens, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    text = "just some text as input for benchmarking"
    text_token_length = len(tokenizer.encode(text))
    token_text = text * (nr_tokens // text_token_length + 1)
    tokens = tokenizer.encode(token_text)[: nr_tokens - 1]

    if len(tokens) < nr_tokens:
        tokens += [tokenizer.pad_token_id] * (nr_tokens - len(tokens) - 1)

    return tokenizer.decode(tokens, skip_special_tokens=True)


def run(model_name, output_csv, nr_input_tokens, nr_output_tokens, memory_rate):
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    watcher = GpuWatcher(gpu_id=0, save_loc=output_csv)
    watcher.start()

    prompt = make_prompt(nr_input_tokens, model_name)

    torch.cuda.empty_cache()
    gc.collect()

    print(f"Loading model: {model_name}")
    t0 = timer()
    llm = LLM(model_name, gpu_memory_utilization=memory_rate)
    load_time = timer() - t0
    print(f"  load time : {load_time:.3f} s")

    print("Running inference...")
    sampling_params = SamplingParams(max_tokens=nr_output_tokens, temperature=0)
    t1 = timer()
    outputs = llm.generate([prompt], sampling_params)
    infer_time = timer() - t1
    tokens_out = sum(len(o.outputs[0].token_ids) for o in outputs if o.outputs)
    print(f"  infer time: {infer_time:.3f} s  |  tokens generated: {tokens_out}")

    del llm
    torch.cuda.empty_cache()
    gc.collect()

    time.sleep(0.5)
    watcher.stop()

    print(f"\nMemory trace saved to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single vLLM inference with GPU memory profiling"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name")
    parser.add_argument("--output_csv", type=str, default="memory_trace.csv",
                        help="Path for GPU memory time-series CSV")
    parser.add_argument("--nr_input_tokens", type=int, default=DEFAULT_NR_INPUT_TOKENS)
    parser.add_argument("--nr_output_tokens", type=int, default=DEFAULT_NR_OUTPUT_TOKENS)
    parser.add_argument("--memory_rate", type=float, default=DEFAULT_MEM_UTIL,
                        help="vLLM gpu_memory_utilization (0.0-1.0)")

    args = parser.parse_args()
    run(args.model, args.output_csv,
        args.nr_input_tokens, args.nr_output_tokens, args.memory_rate)
