from vllm import LLM, SamplingParams
import argparse
from timeit import default_timer as timer
import pandas as pd
import os
from transformers import AutoTokenizer
import subprocess
import threading
import csv
import time
from watcher import GpuWatcher
from watcher import CpuWatcher
import torch
import gc

NR_ITERATIONS = 5
NR_WARMUP_ITERATIONS = 5
DEFAULT_MEM_UTIL = 0.85
DEFAULT_THREAD_PERCENTAGE = 100
DEFAULT_NR_BATCHES = 4
DEFAULT_NR_INPUT_TOKENS = 128

class BenchmarkResult:

    def __init__(self):
        self.data = pd.DataFrame({
            "nr_input_tokens": [],
            "nr_batches": [],
            "thread_percentage": [],
            "memory_rate": [],
            "cold_start": [],
            "model_loc": [],
            "exec_loc": [],
            "avg_ttft": [],
            "max_ttft": [],
            "min_ttft": [],
            "avg_throughput": [],
            "max_throughput": [],
            "min_throughput": [],
            "avg_total": [],
            "max_total": [],
            "min_total": [],
        })

    def add_datapoint(self, datapoint):
        if isinstance(datapoint, pd.DataFrame) and list(datapoint.columns) == list(self.data.columns):
            self.data = pd.concat([self.data, datapoint], ignore_index=True)
        else:
            raise ValueError("Invalid datapoint")

        
    def add_raw_result(self, raw_result, nr_input_tokens, nr_batches, thread_percentage, 
                       memory_rate, cold_start, model_loc, exec_loc):
        """Convert raw vllm results to datapoint and add to dataframe"""
        n = len(raw_result["ttfts"])
        datapoint = pd.DataFrame({
            "nr_input_tokens": [nr_input_tokens],
            "nr_batches": [nr_batches],
            "thread_percentage": [thread_percentage],
            "memory_rate": [memory_rate],
            "cold_start": [cold_start],
            "model_loc": [model_loc],
            "exec_loc": [exec_loc],
            "avg_ttft": [sum(raw_result["ttfts"]) / n],
            "max_ttft": [max(raw_result["ttfts"])],
            "min_ttft": [min(raw_result["ttfts"])],
            "avg_throughput": [sum(raw_result["throughputs"]) / n],
            "max_throughput": [max(raw_result["throughputs"])],
            "min_throughput": [min(raw_result["throughputs"])],
            "avg_total": [sum(raw_result["totals"]) / n],
            "max_total": [max(raw_result["totals"])],
            "min_total": [min(raw_result["totals"])],
        })
        self.add_datapoint(datapoint)

    def save_to_csv(self, dir_path, name):

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        file_path = f"{dir_path}/{name}.csv"
        self.data.to_csv(file_path, mode='w', header=True, index=False)
    

def make_prompt(nr_tokens, model_name, nr_batches=1):
    text = "just some text as input for benchmarking"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    text_token_length = len(tokenizer.encode(text))
    token_text = text * (nr_tokens // text_token_length + 1)
    tokens = tokenizer.encode(token_text)[:nr_tokens]
    
    if len(tokens) < nr_tokens:
        tokens += [tokenizer.pad_token_id] * (nr_tokens - len(tokens))
    
    one_batch = tokenizer.decode(tokens, skip_special_tokens=True)
    batches = [one_batch] * nr_batches
    return batches

def run_vllm_iteration(llm, prompt, nr_output):
    sampling_params = SamplingParams(max_tokens=nr_output, temperature=0)
    start = timer()
    outputs = llm.generate(prompt, sampling_params)
    end = timer()
    total_time = end-start

    tokens_generated = sum(len(output.outputs[0].token_ids) for output in outputs if output.outputs)
    throughput = tokens_generated / total_time if total_time > 0 else 0  # tokens/sec

    avg_time = total_time / len(outputs)

    return {"ttft": avg_time, "throughput": throughput, "total": total_time}

def run_vllm_cold_iteration(model_name, prompt, nr_output):
    start = timer()
    llm = LLM(model_name)
    end = timer()
    load_time = end - start

    result = run_vllm_iteration(llm, prompt, nr_output)
    result["load_time"] = load_time
    result["total_time"] = load_time + (result.get("ttft", 0))

    return result

def run_vllm_warm(llm, prompt, nr_output):
    ttfts = []
    throughputs = []
    totals = []
    torch.cuda.empty_cache()
    gc.collect()

    for i in range(NR_WARMUP_ITERATIONS):
            run_vllm_iteration(llm, prompt, nr_output)

    for i in range(NR_ITERATIONS):
        torch.cuda.empty_cache()
        gc.collect()
        res = run_vllm_iteration(llm, prompt, nr_output)
        ttfts.append(res["ttft"])
        throughputs.append(res["throughput"])
        totals.append(res["total"])

    
    return {"ttfts": ttfts, "throughputs": throughputs, "totals": totals}

def benchmark_changing_sm_percentage(model_name, nr_outputs=128, watcher= None):
    print("------ Running benchmark_changing_sm_percentage ------")

    cold_start = False
    exec_loc = "gpu"
    model_loc = "gpu"

    llm = LLM(model_name, gpu_memory_utilization=DEFAULT_MEM_UTIL)
    prompt = make_prompt(nr_tokens=DEFAULT_NR_INPUT_TOKENS, model_name=model_name, nr_batches=DEFAULT_NR_BATCHES)
    result = BenchmarkResult()

    for i in range(10, 101, 10):
        print(f"Percentage: {i}")
        try:
            os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = f"{i}"
            result_i = run_vllm_warm(llm, prompt, nr_outputs)
            result.add_raw_result(result_i, nr_input_tokens=DEFAULT_NR_INPUT_TOKENS, nr_batches=DEFAULT_NR_BATCHES, 
                        thread_percentage=i, memory_rate=DEFAULT_MEM_UTIL, 
                        cold_start=cold_start, model_loc=model_loc, exec_loc=exec_loc)
            del llm
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(0.5)
        except Exception as e:
            print(f"Failed")
            if 'llm' in locals():
                del llm
            torch.cuda.empty_cache()
            gc.collect()
            continue

    
    return result

def benchmark_changing_batch_size(model_name, nr_outputs=128, watcher= None):
    print("------ Running benchmark_changing_batch_size ------")

    llm = LLM(model_name, gpu_memory_utilization=DEFAULT_MEM_UTIL)
    result = BenchmarkResult()
    cold_start = False
    exec_loc = "gpu"
    model_loc = "gpu"
    
    for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        print(f"Size: {i}")
        try:
            prompt = make_prompt(nr_tokens=DEFAULT_NR_INPUT_TOKENS, model_name=model_name, nr_batches=i)
            result_i = run_vllm_warm(llm, prompt, nr_outputs)
            result.add_raw_result(result_i, nr_input_tokens=DEFAULT_NR_INPUT_TOKENS, nr_batches=i, 
                        thread_percentage=DEFAULT_THREAD_PERCENTAGE, memory_rate=DEFAULT_MEM_UTIL, 
                        cold_start=cold_start, model_loc=model_loc, exec_loc=exec_loc)
            del llm
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(0.5)
        except Exception as e:
            print(f"Failed")
            if 'llm' in locals():
                del llm
            torch.cuda.empty_cache()
            gc.collect()
            continue

    return result

def benchmark_changing_input_length(model_name, nr_outputs=128, watcher= None):
    print("------ Running benchmark_changing_input_length ------")
    llm = LLM(model_name, gpu_memory_utilization=DEFAULT_MEM_UTIL)
    result = BenchmarkResult()
    cold_start = False
    exec_loc = "gpu"
    model_loc = "gpu"

    for i in [64, 128, 256, 512, 1024, 2048]:
        print(f"Length: {i}")
        try:
            prompt = make_prompt(nr_tokens=i, model_name=model_name, nr_batches=DEFAULT_NR_BATCHES)
            result_i = run_vllm_warm(llm, prompt, nr_outputs)
            result.add_raw_result(result_i, nr_input_tokens=i, nr_batches=DEFAULT_NR_BATCHES, 
                        thread_percentage=DEFAULT_THREAD_PERCENTAGE, memory_rate=DEFAULT_MEM_UTIL, 
                        cold_start=cold_start, model_loc=model_loc, exec_loc=exec_loc)
            del llm
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(0.5)
        except Exception as e:
            print(f"Failed")
            if 'llm' in locals():
                del llm
            torch.cuda.empty_cache()
            gc.collect()
            continue

    return result


def benchmark_changing_gpu_memory_utilization(model_name, nr_outputs=128, watcher= None):
    print("------ Running benchmark_changing_gpu_memory_utilization ------")
    prompt = make_prompt(nr_tokens=DEFAULT_NR_INPUT_TOKENS, model_name=model_name, nr_batches=4)
    result = BenchmarkResult()
    cold_start = False
    exec_loc = "gpu"
    model_loc = "gpu"

    for i in range(20,91,10):
        print(f"Percentage: {i}")
        util = i/100
        try:
            llm = LLM(model_name, gpu_memory_utilization=util)
            result_i = run_vllm_warm(llm, prompt, nr_outputs)
            result.add_raw_result(result_i, nr_input_tokens=DEFAULT_NR_INPUT_TOKENS, nr_batches=DEFAULT_NR_BATCHES,
                      thread_percentage=DEFAULT_THREAD_PERCENTAGE, memory_rate=util,
                      cold_start=cold_start, model_loc=model_loc, exec_loc=exec_loc)

            del llm
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(0.5)
        except Exception as e:
            print(f"Failed at memory utilization {util}: {str(e)}")
            if 'llm' in locals():
                del llm
            torch.cuda.empty_cache()
            gc.collect()
            continue

    return result

def run(model_name, execution_loc="gpu", measure_memory=False, mode=1):
    nr_outputs = 128
    watcher = None
    dir_path = f"results/{model_name}"

    if measure_memory:
        if execution_loc == "gpu":
            watcher = GpuWatcher(gpu_id=0, save_loc=f"{dir_path}/gpu-{mode}-memory.csv")
            watcher.start()
        elif execution_loc == "cpu":
            watcher = CpuWatcher(id=0, save_loc=f"{dir_path}/cpu-{mode}-memory.csv")
            watcher.start()

    if mode == 1:
        res = benchmark_changing_sm_percentage(model_name=model_name, nr_outputs=nr_outputs, watcher=watcher)
        res.save_to_csv(dir_path, f"{execution_loc}-{mode}")
    elif mode == 2:
        res = benchmark_changing_batch_size(model_name=model_name, nr_outputs=nr_outputs, watcher=watcher)
        res.save_to_csv(dir_path, f"{execution_loc}-{mode}")
    elif mode == 3:
        res = benchmark_changing_input_length(model_name=model_name, nr_outputs=nr_outputs, watcher=watcher)
        res.save_to_csv(dir_path, f"{execution_loc}-{mode}")
    elif mode == 4:
        res = benchmark_changing_gpu_memory_utilization(model_name=model_name, nr_outputs=nr_outputs, watcher=watcher)
        res.save_to_csv(dir_path, f"{execution_loc}-{mode}")
    elif mode == 5:
        res = benchmark_changing_sm_percentage(model_name=model_name, nr_outputs=nr_outputs, watcher=watcher)
        res.save_to_csv(dir_path, f"{execution_loc}-1")
        res = benchmark_changing_batch_size(model_name=model_name, nr_outputs=nr_outputs, watcher=watcher)
        res.save_to_csv(dir_path, f"{execution_loc}-2")
        res = benchmark_changing_input_length(model_name=model_name, nr_outputs=nr_outputs, watcher=watcher)
        res.save_to_csv(dir_path, f"{execution_loc}-3")
        res = benchmark_changing_gpu_memory_utilization(model_name=model_name, nr_outputs=nr_outputs, watcher=watcher)
        res.save_to_csv(dir_path, f"{execution_loc}-4")
    
    if watcher:
        time.sleep(0.5)
        watcher.stop()

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Huggingface model name")
parser.add_argument("--execution_location", type=str, default="gpu", help="cpu/gpu")
parser.add_argument("--mode", type=int, default=5, help="Benchmark mode")
parser.add_argument("--measure_memory", action="store_true", help="measure memory usage")
parser.add_argument("--cold_start", action="store_true", help="Enable cold start")

args = parser.parse_args()

model_name = args.model
execution_loc = args.execution_location
mode = args.mode
measure_memory = args.measure_memory
is_cold_start = args.cold_start

run(model_name, execution_loc=execution_loc, measure_memory=measure_memory, mode=mode)

"""
if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Huggingface model name")
    parser.add_argument("--model_location", type=str, required=True, help="cpu/gpu")
    parser.add_argument("--execution_location", type=str, required=True, help="cpu/gpu")
    parser.add_argument("--thread_percentage", type=int, required=True, help="for file naming")
    parser.add_argument("--nr_input_tokens", type=int, default=10, help="nr. of input tokens")
    parser.add_argument("--nr_output_tokens", type=int, default=10, help="nr. of output tokens")
    parser.add_argument("--nr_batches", type=int, default=1, help="batch size")
    parser.add_argument("--memory_rate", type=float, default=0.85, help="rate of gpu memory used")
    parser.add_argument("--measure_memory", action="store_true", help="measure memory usage")
    parser.add_argument("--cold_start", action="store_true", help="Enable cold start")

    args = parser.parse_args()

    model_name = args.model
    is_cold_start = args.cold_start
    model_loc = args.model_location
    execution_loc = args.execution_location
    thread_percentage = args.thread_percentage
    nr_input_tokens = args.nr_input_tokens
    nr_output = args.nr_output_tokens
    measure_memory = args.measure_memory
    nr_batches = args.nr_batches
    memory_rate = args.memory_rate

    dir_path = f"results/{model_name}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_type = f"{model_loc}-{execution_loc}-{is_cold_start}-batch"

    prompt = make_prompt(nr_tokens=nr_input_tokens, nr_batches= nr_batches, model_name=model_name)
    
    

    n = len(ttfts)


    df = pd.DataFrame({
        "nr_input_tokens": [nr_input_tokens],
        "nr_batches": [nr_batches],
        "thread_percentage": [thread_percentage],
        "memory_rate": [memory_rate],
        "cold_start": [is_cold_start],
        "model_loc": [model_loc],
        "exec_loc": [execution_loc],
        "avg_ttft": [sum(ttfts) / n],
        "max_ttft": [max(ttfts)],
        "min_ttft": [min(ttfts)],
        "avg_throughput": [sum(throughputs) / n],
        "max_throughput": [max(throughputs)],
        "min_throughput": [min(throughputs)]
    })


    #csv_df_string = df.to_csv(index=False)
    #print(csv_df_string)

    file_path = f"{dir_path}/{file_type}.csv"
    df.to_csv(file_path, mode='a', header=False, index=False)
"""


