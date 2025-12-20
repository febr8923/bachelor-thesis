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



def make_prompt(nr_tokens, model_name):
    text = "just some text as input for benchmarking"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    text_token_length = len(tokenizer.encode(text))
    token_text = text * (nr_tokens // text_token_length + 1)

    tokens = tokenizer.encode(token_text)[:nr_tokens]

    if len(tokens) < nr_tokens:
        tokens += [tokenizer.pad_token_id] * (nr_tokens - len(tokens))

    return tokenizer.decode(tokens)

def run_vllm(llm, prompt, nr_output):
    sampling_params = SamplingParams(max_tokens=nr_output, temperature=0)
    start = timer()
    outputs = llm.generate(prompt, sampling_params)
    end = timer()
    return end - start

def run_vllm_cold(model_name, prompt, nr_output):
    #todo
    start = timer()
    llm = LLM(model_name)
    end = timer()
    time = run_vllm(llm, prompt, nr_output)

    return (end-start) + time

NR_ITERATIONS = 5
NR_WARMUP_ITERATIONS = 5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Huggingface model name")
    parser.add_argument("--cold_start", action="store_true", help="Enable cold start")
    parser.add_argument("--model_location", type=str, required=True, help="cpu/gpu")
    parser.add_argument("--execution_location", type=str, required=True, help="cpu/gpu")
    parser.add_argument("--thread_percentage", type=int, required=True, help="for file naming")
    parser.add_argument("--nr_input_tokens", type=int, default=10, help="nr. of input tokens")
    parser.add_argument("--nr_output_tokens", type=int, default=10, help="nr. of output tokens")
    parser.add_argument("--measure_memory", action="store_true", help="measure memory usage")

    args = parser.parse_args()

    model_name = args.model
    is_cold_start = args.cold_start
    model_loc = args.model_location
    execution_loc = args.execution_location
    thread_percentage = args.thread_percentage
    nr_input_tokens = args.nr_input_tokens
    nr_output = args.nr_output_tokens
    measure_memory = args.measure_memory

    dir_path = f"results/{model_name}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_type = f"{model_loc}-{execution_loc}-{is_cold_start}"

    prompt = make_prompt(nr_tokens=nr_input_tokens, model_name=model_name)
    
    times = []
    
    if measure_memory:
        if execution_loc == "gpu":
            watcher = GpuWatcher(gpu_id=0, save_loc=f"{dir_path}/gpu-memory-{file_type}-{thread_percentage}.csv")
        elif execution_loc == "cpu":
            watcher = CpuWatcher(cpu_id=0, save_loc=f"{dir_path}/cpu-memory-{file_type}-{thread_percentage}.csv")
        
    if is_cold_start:
        if measure_memory:
            watcher.start()
        
        res = run_vllm_cold(model_name,prompt, nr_output)
        times.append(res)
    else:
        if measure_memory:
            watcher.start()
        llm = LLM(model_name)
        
        for i in range(NR_WARMUP_ITERATIONS):
            run_vllm(llm, prompt, nr_output)

        for i in range(NR_ITERATIONS):
            res = run_vllm(llm, prompt, nr_output)
            times.append(res)
    
    if measure_memory:
        #stop tracking
        time.sleep(0.5)
        watcher.stop()

    n = len(times)


    df = pd.DataFrame([
        sum(times) / n, max(times), min(times)
    ])

    #csv_df_string = df.to_csv(index=False)
    #print(csv_df_string)

    file_path = f"{dir_path}/{file_type}.csv"
    df.to_csv(file_path, mode='a', header=False, index=False)

