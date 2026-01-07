"""
vLLM Benchmark Runner
"""
import os
import time
import torch
import gc
from vllm import LLM, SamplingParams
from timeit import default_timer as timer
from transformers import AutoTokenizer
from typing import Dict, List, Optional

from core.config import *
from core.results import BenchmarkResult
from core.utils import cleanup_memory, start_memory_watcher, stop_memory_watcher


def make_prompt(nr_tokens: int, model_name: str, nr_batches: int = 1) -> List[str]:
    """Generate prompts with specified token count"""
    text = "just some text as input for benchmarking"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    text_token_length = len(tokenizer.encode(text))
    token_text = text * (nr_tokens // text_token_length + 1)
    tokens = tokenizer.encode(token_text)[:(nr_tokens-1)]

    if len(tokens) < nr_tokens:
        tokens += [tokenizer.pad_token_id] * (nr_tokens - len(tokens) - 1)

    one_batch = tokenizer.decode(tokens, skip_special_tokens=True)
    batches = [one_batch] * nr_batches
    return batches


def run_vllm_iteration(llm: LLM, prompt: List[str], nr_output: int) -> Dict:
    """Run single vLLM inference iteration"""
    sampling_params = SamplingParams(max_tokens=nr_output, temperature=0)

    start = timer()
    outputs = llm.generate(prompt, sampling_params)
    end = timer()
    total_time = end - start

    tokens_generated = sum(len(output.outputs[0].token_ids) for output in outputs if output.outputs)
    throughput = tokens_generated / total_time if total_time > 0 else 0

    avg_time = total_time / len(outputs) if outputs else 0

    return {"ttft": avg_time, "throughput": throughput, "total": total_time}


def run_vllm_warm(llm: LLM, prompt: List[str], nr_output: int) -> Dict:
    """Run warm benchmark with warmup iterations"""
    ttfts = []
    throughputs = []
    totals = []

    cleanup_memory()

    # Warmup
    for _ in range(NR_WARMUP_ITERATIONS):
        run_vllm_iteration(llm, prompt, nr_output)

    # Actual measurements
    for _ in range(NR_ITERATIONS):
        cleanup_memory()
        res = run_vllm_iteration(llm, prompt, nr_output)
        ttfts.append(res["ttft"])
        throughputs.append(res["throughput"])
        totals.append(res["total"])

    return {"ttfts": ttfts, "throughputs": throughputs, "totals": totals}


class VLLMBenchmark:
    """vLLM Benchmark Runner"""

    def __init__(self, model_name: str, exec_loc: str = "gpu", measure_memory: bool = False):
        self.model_name = model_name
        self.exec_loc = exec_loc
        self.measure_memory = measure_memory
        self.model_loc = exec_loc  # For vLLM, model and execution are on same device
        self.watcher = None

    def _start_watcher(self, mode: int):
        """Start memory watcher if enabled"""
        if self.measure_memory:
            self.watcher = start_memory_watcher(self.exec_loc, self.model_name, mode)

    def _stop_watcher(self):
        """Stop memory watcher"""
        stop_memory_watcher(self.watcher)
        self.watcher = None

    def benchmark_sm_percentage(self) -> BenchmarkResult:
        """Benchmark with varying GPU SM percentage"""
        print("\n------ Running benchmark_changing_sm_percentage ------\n")

        result = BenchmarkResult("vllm")
        prompt = make_prompt(DEFAULT_NR_INPUT_TOKENS, self.model_name, DEFAULT_BATCH_SIZE)

        for pct in GPU_SM_PERCENTAGES:
            print(f"SM Percentage: {pct}%")
            try:
                os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = f"{pct}"
                llm = LLM(self.model_name, gpu_memory_utilization=DEFAULT_VLLM_MEM_UTIL)
                result_i = run_vllm_warm(llm, prompt, DEFAULT_NR_OUTPUT_TOKENS)
                result.add_raw_result_vllm(
                    result_i, self.model_name, DEFAULT_NR_INPUT_TOKENS, DEFAULT_BATCH_SIZE,
                    pct, DEFAULT_VLLM_MEM_UTIL, False, self.model_loc, self.exec_loc
                )
                del llm
                cleanup_memory()
                time.sleep(0.5)
            except Exception as e:
                print(f"Failed: {str(e)}")
                if 'llm' in locals():
                    del llm
                cleanup_memory()
                continue

        return result

    def benchmark_batch_size(self) -> BenchmarkResult:
        """Benchmark with varying batch sizes"""
        print("\n------ Running benchmark_changing_batch_size ------\n")

        batch_sizes = BATCH_SIZES_CPU if self.exec_loc == "cpu" else BATCH_SIZES_GPU
        nr_outputs = 32 if self.exec_loc == "cpu" else DEFAULT_NR_OUTPUT_TOKENS

        if self.exec_loc == "cpu":
            llm = LLM(self.model_name, max_num_batched_tokens=2048, max_model_len=2048)
        else:
            llm = LLM(self.model_name, gpu_memory_utilization=DEFAULT_VLLM_MEM_UTIL)

        result = BenchmarkResult("vllm")

        for batch_size in batch_sizes:
            print(f"Batch size: {batch_size}")
            try:
                prompt = make_prompt(DEFAULT_NR_INPUT_TOKENS, self.model_name, batch_size)
                result_i = run_vllm_warm(llm, prompt, nr_outputs)
                result.add_raw_result_vllm(
                    result_i, self.model_name, DEFAULT_NR_INPUT_TOKENS, batch_size,
                    DEFAULT_THREAD_PERCENTAGE, DEFAULT_VLLM_MEM_UTIL,
                    False, self.model_loc, self.exec_loc
                )
                cleanup_memory()
                time.sleep(0.5)
            except Exception as e:
                print(f"Failed: {str(e)}")
                cleanup_memory()
                continue

        del llm
        return result

    def benchmark_input_length(self) -> BenchmarkResult:
        """Benchmark with varying input lengths"""
        print("\n------ Running benchmark_changing_input_length ------\n")

        nr_outputs = 32 if self.exec_loc == "cpu" else DEFAULT_NR_OUTPUT_TOKENS

        if self.exec_loc == "cpu":
            llm = LLM(self.model_name, max_num_batched_tokens=2048, max_model_len=2048)
        else:
            llm = LLM(self.model_name, gpu_memory_utilization=DEFAULT_VLLM_MEM_UTIL)

        result = BenchmarkResult("vllm")

        for length in INPUT_LENGTHS:
            print(f"Input length: {length}")
            try:
                prompt = make_prompt(length, self.model_name, DEFAULT_BATCH_SIZE)
                result_i = run_vllm_warm(llm, prompt, nr_outputs)
                result.add_raw_result_vllm(
                    result_i, self.model_name, length, DEFAULT_BATCH_SIZE,
                    DEFAULT_THREAD_PERCENTAGE, DEFAULT_VLLM_MEM_UTIL,
                    False, self.model_loc, self.exec_loc
                )
                cleanup_memory()
                time.sleep(0.5)
            except Exception as e:
                print(f"Failed: {str(e)}")
                cleanup_memory()
                continue

        del llm
        return result

    def benchmark_memory_utilization(self) -> BenchmarkResult:
        """Benchmark with varying GPU memory utilization"""
        print("\n------ Running benchmark_changing_gpu_memory_utilization ------\n")

        if self.exec_loc != "gpu":
            raise ValueError("Memory utilization benchmark only available for GPU")

        result = BenchmarkResult("vllm")
        prompt = make_prompt(DEFAULT_NR_INPUT_TOKENS, self.model_name, 4)

        for util in MEMORY_UTILS:
            print(f"Memory utilization: {util*100:.0f}%")
            try:
                llm = LLM(self.model_name, gpu_memory_utilization=util)
                result_i = run_vllm_warm(llm, prompt, DEFAULT_NR_OUTPUT_TOKENS)
                result.add_raw_result_vllm(
                    result_i, self.model_name, DEFAULT_NR_INPUT_TOKENS, DEFAULT_BATCH_SIZE,
                    DEFAULT_THREAD_PERCENTAGE, util, False, self.model_loc, self.exec_loc
                )
                del llm
                cleanup_memory()
                time.sleep(0.5)
            except Exception as e:
                print(f"Failed: {str(e)}")
                if 'llm' in locals():
                    del llm
                cleanup_memory()
                continue

        return result

    def benchmark_cpu_threads(self) -> BenchmarkResult:
        """Benchmark with varying CPU thread counts"""
        print("\n------ Running cpu_benchmark_changing_nr_threads ------\n")

        result = BenchmarkResult("vllm")
        prompt = make_prompt(DEFAULT_NR_INPUT_TOKENS, self.model_name, DEFAULT_BATCH_SIZE)

        for thread_count in CPU_THREAD_COUNTS:
            print(f"Thread count: {thread_count}")
            try:
                os.environ['OMP_NUM_THREADS'] = f"{thread_count}"
                os.environ['MKL_NUM_THREADS'] = f"{thread_count}"

                llm = LLM(self.model_name, max_num_batched_tokens=2048, max_model_len=2048)
                result_i = run_vllm_warm(llm, prompt, 32)
                result.add_raw_result_vllm(
                    result_i, self.model_name, DEFAULT_NR_INPUT_TOKENS, DEFAULT_BATCH_SIZE,
                    thread_count, 1.0, False, self.model_loc, self.exec_loc
                )
                del llm
                cleanup_memory()
                time.sleep(0.5)
            except Exception as e:
                print(f"Failed: {str(e)}")
                if 'llm' in locals():
                    del llm
                cleanup_memory()
                continue

        return result

    def run(self, mode: int = 5) -> List[BenchmarkResult]:
        """
        Run benchmarks based on mode

        Args:
            mode: Benchmark mode (1-5)

        Returns:
            List of BenchmarkResult objects
        """
        results = []
        dir_path = f"{RESULTS_DIR}/{self.model_name}"

        self._start_watcher(mode)

        try:
            if mode == 1:
                # Resource variation
                if self.exec_loc == "gpu":
                    res = self.benchmark_sm_percentage()
                else:
                    res = self.benchmark_cpu_threads()
                res.save_to_csv(dir_path, f"{self.exec_loc}-1")
                results.append(res)

            elif mode == 2:
                # Batch size
                res = self.benchmark_batch_size()
                res.save_to_csv(dir_path, f"{self.exec_loc}-2")
                results.append(res)

            elif mode == 3:
                # Input length
                res = self.benchmark_input_length()
                res.save_to_csv(dir_path, f"{self.exec_loc}-3")
                results.append(res)

            elif mode == 4:
                # Memory utilization
                res = self.benchmark_memory_utilization()
                res.save_to_csv(dir_path, f"{self.exec_loc}-4")
                results.append(res)

            elif mode == 5:
                # All benchmarks
                if self.exec_loc == "gpu":
                    res = self.benchmark_sm_percentage()
                    res.save_to_csv(dir_path, f"{self.exec_loc}-1")
                    results.append(res)
                else:
                    res = self.benchmark_cpu_threads()
                    res.save_to_csv(dir_path, f"{self.exec_loc}-1")
                    results.append(res)

                res = self.benchmark_batch_size()
                res.save_to_csv(dir_path, f"{self.exec_loc}-2")
                results.append(res)

                res = self.benchmark_input_length()
                res.save_to_csv(dir_path, f"{self.exec_loc}-3")
                results.append(res)

                if self.exec_loc == "gpu":
                    res = self.benchmark_memory_utilization()
                    res.save_to_csv(dir_path, f"{self.exec_loc}-4")
                    results.append(res)

        finally:
            self._stop_watcher()

        return results
