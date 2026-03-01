"""
Cold-start vLLM benchmark.
Every measurement iteration creates a fresh LLM from scratch (no warmup),
so load_time, TTFT, throughput, and total are all captured per cold start.
Returns the same CSV schema as run-vllm.py (warm) plus load_time columns.

Usage:
  python run-vllm-cold.py --model <model_name> --execution_location gpu --mode 5
  python run-vllm-cold.py --model <model_name> --execution_location gpu --mode 1
"""

from vllm import LLM, SamplingParams
import argparse
from timeit import default_timer as timer
import pandas as pd
import os
from transformers import AutoTokenizer
import time
from watcher import GpuWatcher, CpuWatcher
import torch
import gc

NR_ITERATIONS = 5

DEFAULT_MEM_UTIL = 0.85
DEFAULT_THREAD_PERCENTAGE = 100
DEFAULT_NR_BATCHES = 1
DEFAULT_NR_INPUT_TOKENS = 128


# ---------------------------------------------------------------------------
# BenchmarkResult — same columns as run-vllm.py + load_time stats
# ---------------------------------------------------------------------------
class BenchmarkResult:

    COLUMNS = [
        "nr_input_tokens", "nr_batches", "thread_percentage", "memory_rate",
        "cold_start", "model_loc", "exec_loc",
        "avg_ttft", "max_ttft", "min_ttft",
        "avg_throughput", "max_throughput", "min_throughput",
        "avg_total", "max_total", "min_total",
        "avg_load_time", "max_load_time", "min_load_time",
    ]

    def __init__(self):
        self.data = pd.DataFrame({c: [] for c in self.COLUMNS})

    def add_raw_result(self, raw, *, nr_input_tokens, nr_batches,
                       thread_percentage, memory_rate, model_loc, exec_loc):
        n = len(raw["ttfts"])
        lt = raw.get("load_times", [0.0] * n)
        nl = len(lt)
        row = pd.DataFrame({
            "nr_input_tokens": [nr_input_tokens],
            "nr_batches": [nr_batches],
            "thread_percentage": [thread_percentage],
            "memory_rate": [memory_rate],
            "cold_start": [True],
            "model_loc": [model_loc],
            "exec_loc": [exec_loc],
            "avg_ttft": [sum(raw["ttfts"]) / n],
            "max_ttft": [max(raw["ttfts"])],
            "min_ttft": [min(raw["ttfts"])],
            "avg_throughput": [sum(raw["throughputs"]) / n],
            "max_throughput": [max(raw["throughputs"])],
            "min_throughput": [min(raw["throughputs"])],
            "avg_total": [sum(raw["totals"]) / n],
            "max_total": [max(raw["totals"])],
            "min_total": [min(raw["totals"])],
            "avg_load_time": [sum(lt) / nl],
            "max_load_time": [max(lt)],
            "min_load_time": [min(lt)],
        })
        self.data = pd.concat([self.data, row], ignore_index=True)

    def save_to_csv(self, dir_path, name):
        os.makedirs(dir_path, exist_ok=True)
        path = f"{dir_path}/{name}.csv"
        write_header = not os.path.exists(path)
        self.data.to_csv(path, mode="a",
                         header=write_header, index=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_prompt(nr_tokens, model_name, nr_batches=1):
    text = "just some text as input for benchmarking"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    text_token_length = len(tokenizer.encode(text))
    token_text = text * (nr_tokens // text_token_length + 1)
    tokens = tokenizer.encode(token_text)[: (nr_tokens - 1)]

    if len(tokens) < nr_tokens:
        tokens += [tokenizer.pad_token_id] * (nr_tokens - len(tokens) - 1)

    one_batch = tokenizer.decode(tokens, skip_special_tokens=True)
    return [one_batch] * nr_batches


def _run_single_inference(llm, prompt, nr_output):
    """Run one generate() call and return per-request ttft, throughput, total."""
    sampling_params = SamplingParams(max_tokens=nr_output, temperature=0)

    start = timer()
    outputs = llm.generate(prompt, sampling_params)
    end = timer()
    total_time = end - start

    tokens_generated = sum(
        len(o.outputs[0].token_ids) for o in outputs if o.outputs
    )
    throughput = tokens_generated / total_time if total_time > 0 else 0
    avg_time = total_time / len(outputs)

    return {"ttft": avg_time, "throughput": throughput, "total": total_time}


def run_cold(model_name, prompt, nr_output, *,
             gpu_memory_utilization=None):
    """Cold-start: create a fresh LLM (no warmup), run a single inference.

    Returns the same dict shape as run_vllm_warm in run-vllm.py:
        {"ttfts": [], "throughputs": [], "totals": [], "load_times": []}

    Only 1 iteration per call — the caller (shell script) invokes this in a
    fresh process N times so each gets a truly cold CUDA context.
    """
    torch.cuda.empty_cache()
    gc.collect()

    # --- model load (timed) ---
    start = timer()
    if gpu_memory_utilization is not None:
        llm = LLM(model_name,
                   gpu_memory_utilization=gpu_memory_utilization)
    else:
        llm = LLM(model_name,
                   max_num_batched_tokens=2048, max_model_len=2048)
    load_time = timer() - start

    # --- single inference (no warmup — that's the point) ---
    res = _run_single_inference(llm, prompt, nr_output)

    del llm
    torch.cuda.empty_cache()
    gc.collect()

    return {"ttfts": [res["ttft"]], "throughputs": [res["throughput"]],
            "totals": [res["total"]], "load_times": [load_time]}


# ---------------------------------------------------------------------------
# Benchmark variants (mirror run-vllm.py but cold-start everywhere)
# ---------------------------------------------------------------------------
def gpu_benchmark_changing_sm_percentage(model_name, nr_outputs=128,
                                         watcher=None):
    print("\n------ [COLD] benchmark_changing_sm_percentage ------\n")
    prompt = make_prompt(nr_tokens=DEFAULT_NR_INPUT_TOKENS,
                         model_name=model_name,
                         nr_batches=DEFAULT_NR_BATCHES)
    result = BenchmarkResult()

    for pct in range(10, 101, 10):
        print(f"SM percentage: {pct}")
        try:
            os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(pct)

            result_i = run_cold(model_name, prompt, nr_outputs,
                                gpu_memory_utilization=DEFAULT_MEM_UTIL)
            result_ttft = run_cold(model_name, prompt, 1,
                                   gpu_memory_utilization=DEFAULT_MEM_UTIL)
            result_i["ttfts"] = result_ttft["ttfts"]

            result.add_raw_result(result_i,
                nr_input_tokens=DEFAULT_NR_INPUT_TOKENS,
                nr_batches=DEFAULT_NR_BATCHES,
                thread_percentage=pct,
                memory_rate=DEFAULT_MEM_UTIL,
                model_loc="gpu", exec_loc="gpu")

            torch.cuda.empty_cache(); gc.collect(); time.sleep(0.5)
        except Exception as e:
            print(f"Failed: {e}")
            torch.cuda.empty_cache(); gc.collect()
            continue

    return result


def benchmark_changing_batch_size(model_name, exec_loc="gpu",
                                  nr_outputs=128, watcher=None):
    print("\n------ [COLD] benchmark_changing_batch_size ------\n")
    if exec_loc == "cpu":
        nr_outputs = 32
    gpu_mem = DEFAULT_MEM_UTIL if exec_loc == "gpu" else None
    result = BenchmarkResult()

    for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        print(f"Batch size: {bs}")
        try:
            prompt = make_prompt(nr_tokens=DEFAULT_NR_INPUT_TOKENS,
                                 model_name=model_name, nr_batches=bs)

            result_i = run_cold(model_name, prompt, nr_outputs,
                                gpu_memory_utilization=gpu_mem)
            result_ttft = run_cold(model_name, prompt, 1,
                                   gpu_memory_utilization=gpu_mem)
            result_i["ttfts"] = result_ttft["ttfts"]

            result.add_raw_result(result_i,
                nr_input_tokens=DEFAULT_NR_INPUT_TOKENS,
                nr_batches=bs,
                thread_percentage=DEFAULT_THREAD_PERCENTAGE,
                memory_rate=DEFAULT_MEM_UTIL,
                model_loc=exec_loc, exec_loc=exec_loc)

            torch.cuda.empty_cache(); gc.collect(); time.sleep(0.5)
        except Exception as e:
            print(f"Failed: {e}")
            torch.cuda.empty_cache(); gc.collect()
            continue

    return result


def benchmark_changing_input_length(model_name, exec_loc="gpu",
                                    nr_outputs=128, watcher=None):
    print("\n------ [COLD] benchmark_changing_input_length ------\n")
    if exec_loc == "cpu":
        nr_outputs = 32
    gpu_mem = DEFAULT_MEM_UTIL if exec_loc == "gpu" else None
    result = BenchmarkResult()

    for length in [64, 128, 256, 512, 1024, 2048]:
        print(f"Input length: {length}")
        try:
            prompt = make_prompt(nr_tokens=length, model_name=model_name,
                                 nr_batches=DEFAULT_NR_BATCHES)

            result_i = run_cold(model_name, prompt, nr_outputs,
                                gpu_memory_utilization=gpu_mem)
            result_ttft = run_cold(model_name, prompt, 1,
                                   gpu_memory_utilization=gpu_mem)
            result_i["ttfts"] = result_ttft["ttfts"]

            result.add_raw_result(result_i,
                nr_input_tokens=length,
                nr_batches=DEFAULT_NR_BATCHES,
                thread_percentage=DEFAULT_THREAD_PERCENTAGE,
                memory_rate=DEFAULT_MEM_UTIL,
                model_loc=exec_loc, exec_loc=exec_loc)

            torch.cuda.empty_cache(); gc.collect(); time.sleep(0.5)
        except Exception as e:
            print(f"Failed: {e}")
            torch.cuda.empty_cache(); gc.collect()
            continue

    return result


def gpu_benchmark_changing_memory_utilization(model_name, nr_outputs=128,
                                               watcher=None):
    print("\n------ [COLD] benchmark_changing_gpu_memory_utilization ------\n")
    prompt = make_prompt(nr_tokens=DEFAULT_NR_INPUT_TOKENS,
                         model_name=model_name, nr_batches=DEFAULT_NR_BATCHES)
    result = BenchmarkResult()

    for pct in range(20, 91, 10):
        util = pct / 100
        print(f"Memory util: {util}")
        try:
            result_i = run_cold(model_name, prompt, nr_outputs,
                                gpu_memory_utilization=util)
            result_ttft = run_cold(model_name, prompt, 1,
                                   gpu_memory_utilization=util)
            result_i["ttfts"] = result_ttft["ttfts"]

            result.add_raw_result(result_i,
                nr_input_tokens=DEFAULT_NR_INPUT_TOKENS,
                nr_batches=DEFAULT_NR_BATCHES,
                thread_percentage=DEFAULT_THREAD_PERCENTAGE,
                memory_rate=util,
                model_loc="gpu", exec_loc="gpu")

            torch.cuda.empty_cache(); gc.collect(); time.sleep(0.5)
        except Exception as e:
            print(f"Failed at memory utilization {util}: {e}")
            torch.cuda.empty_cache(); gc.collect()
            continue

    return result


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def run(model_name, execution_loc="gpu", measure_memory=False, mode=1):
    nr_outputs = 128
    watcher = None
    dir_path = f"results/{model_name}"

    # Hide GPUs so vLLM is forced to use the CPU backend
    if execution_loc == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if measure_memory:
        if execution_loc == "gpu":
            watcher = GpuWatcher(gpu_id=0,
                                 save_loc=f"{dir_path}/gpu-{mode}-memory.csv")
            watcher.start()
        elif execution_loc == "cpu":
            watcher = CpuWatcher(id=0,
                                 save_loc=f"{dir_path}/cpu-{mode}-memory.csv")
            watcher.start()

    suffix = "-cold"

    if mode == 1:
        if execution_loc == "gpu":
            res = gpu_benchmark_changing_sm_percentage(
                model_name=model_name, nr_outputs=nr_outputs, watcher=watcher)
        else:
            # CPU cold-start: same loop but cpu flavour
            res = benchmark_changing_batch_size(
                model_name=model_name, exec_loc="cpu",
                nr_outputs=32, watcher=watcher)
        res.save_to_csv(dir_path, f"{execution_loc}-{mode}{suffix}")

    elif mode == 2:
        res = benchmark_changing_batch_size(
            model_name=model_name, exec_loc=execution_loc,
            nr_outputs=nr_outputs, watcher=watcher)
        res.save_to_csv(dir_path, f"{execution_loc}-{mode}{suffix}")

    elif mode == 3:
        res = benchmark_changing_input_length(
            model_name=model_name, exec_loc=execution_loc,
            nr_outputs=nr_outputs, watcher=watcher)
        res.save_to_csv(dir_path, f"{execution_loc}-{mode}{suffix}")

    elif mode == 4:
        if execution_loc == "gpu":
            res = gpu_benchmark_changing_memory_utilization(
                model_name=model_name, nr_outputs=nr_outputs, watcher=watcher)
        else:
            raise ValueError("Memory-utilization benchmark not available for cpu!")
        res.save_to_csv(dir_path, f"{execution_loc}-{mode}{suffix}")

    elif mode == 5:
        # Run all applicable benchmarks
        if execution_loc == "gpu":
            res = gpu_benchmark_changing_sm_percentage(
                model_name=model_name, nr_outputs=nr_outputs, watcher=watcher)
        else:
            res = benchmark_changing_batch_size(
                model_name=model_name, exec_loc="cpu",
                nr_outputs=32, watcher=watcher)
        res.save_to_csv(dir_path, f"{execution_loc}-1{suffix}")

        res = benchmark_changing_batch_size(
            model_name=model_name, exec_loc=execution_loc,
            nr_outputs=nr_outputs, watcher=watcher)
        res.save_to_csv(dir_path, f"{execution_loc}-2{suffix}")

        res = benchmark_changing_input_length(
            model_name=model_name, exec_loc=execution_loc,
            nr_outputs=nr_outputs, watcher=watcher)
        res.save_to_csv(dir_path, f"{execution_loc}-3{suffix}")

        if execution_loc == "gpu":
            res = gpu_benchmark_changing_memory_utilization(
                model_name=model_name, nr_outputs=nr_outputs, watcher=watcher)
            res.save_to_csv(dir_path, f"{execution_loc}-4{suffix}")

    if watcher:
        time.sleep(0.5)
        watcher.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cold-start vLLM benchmarks (fresh LLM every iteration)")
    parser.add_argument("--model", type=str, required=True,
                        help="Huggingface model name")
    parser.add_argument("--execution_location", type=str, default="gpu",
                        help="cpu or gpu")
    parser.add_argument("--mode", type=int, default=5,
                        help="Benchmark mode (1-5, same as run-vllm.py)")
    parser.add_argument("--measure_memory", action="store_true",
                        help="Enable memory watcher")

    args = parser.parse_args()

    run(args.model,
        execution_loc=args.execution_location,
        measure_memory=args.measure_memory,
        mode=args.mode)
