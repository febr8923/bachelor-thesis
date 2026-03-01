"""
Single-run vLLM CPU memory profiler.

Forces CPU-only execution (CUDA_VISIBLE_DEVICES=""), loads the model once,
runs one inference, and captures RAM usage over time via CpuWatcher (psutil,
0.5 s intervals).

Usage:
  python vllm_memory_cpu.py --model EleutherAI/gpt-j-6b
  python vllm_memory_cpu.py --model EleutherAI/gpt-j-6b \
      --output_csv results/EleutherAI/gpt-j-6b/single_cpu_memory.csv \
      --nr_input_tokens 128 --nr_output_tokens 32
"""

import argparse
import gc
import os
import sys
import time
from timeit import default_timer as timer

# Must be set before importing vLLM / torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from watcher import CpuWatcher


DEFAULT_NR_INPUT_TOKENS = 128
DEFAULT_NR_OUTPUT_TOKENS = 32  # lower default — CPU inference is slow


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


def run(model_name, output_csv, nr_input_tokens, nr_output_tokens):
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    watcher = CpuWatcher(id=0, save_loc=output_csv, interval=0.5)
    watcher.start()

    prompt = make_prompt(nr_input_tokens, model_name)

    gc.collect()

    print(f"Loading model: {model_name} (CPU)")
    t0 = timer()
    llm = LLM(model_name, max_num_batched_tokens=2048, max_model_len=2048)
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
    gc.collect()

    time.sleep(1.0)
    watcher.stop()

    print(f"\nMemory trace saved to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single vLLM CPU inference with RAM profiling"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name")
    parser.add_argument("--output_csv", type=str, default="cpu_memory_trace.csv",
                        help="Output CSV path for RAM time-series")
    parser.add_argument("--nr_input_tokens", type=int, default=DEFAULT_NR_INPUT_TOKENS)
    parser.add_argument("--nr_output_tokens", type=int, default=DEFAULT_NR_OUTPUT_TOKENS)

    args = parser.parse_args()
    run(args.model, args.output_csv, args.nr_input_tokens, args.nr_output_tokens)
