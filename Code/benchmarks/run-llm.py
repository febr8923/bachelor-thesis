import torch
import os
from PIL import Image
from torchvision import transforms
from timeit import default_timer as timer
import sys
import itertools
import pandas as pd
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM
import argparse

#mode can be 1 (CPU,CPU), 2 (CPU,GPU), 3 (GPU,CPU), 4 (GPU,GPU) for (model location, execution location)
#model should be on gpu/cpu, execution should be on gpu/cpu 

os.environ['TRANSFORMERS_CACHE'] = '/iopsstor/scratch/cscs/fbrunne'
model = None

def run_llm(model, tokenizer, model_loc: str, exec_loc: str):
    print(f"model_loc: {model_loc}, exec_loc: {exec_loc}")

    # memory: {torch.cuda.memory_allocated() / (1024**2):.2f}MB")
    #print(f"Reserved memory: {torch.cuda.memory_reserved() / (1024**2):.2f}MB")
    text = "some kind of text"

    input_batch = tokenizer(text, return_tensors='pt')
    

    # Move model to model_loc first
    torch.cuda.empty_cache()  # Clear cache before moving
    if(model_loc == "cpu"):
        model.to('cpu')
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        input_batch = {k: v.to('cpu') for k, v in input_batch.items()}
    elif(model_loc == "gpu"):
        model.to('cuda')
        torch.cuda.synchronize()
        input_batch = {k: v.to('cuda') for k, v in input_batch.items()}
    else:
        raise ValueError(f"Wrong model_loc, found {model_loc}, should be 'cpu' or 'gpu'")

    start = timer()

    # Then move model to exec_loc and measure the time
    if(exec_loc == "cpu"):
        model.to('cpu')
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        input_batch = {k: v.to('cpu') for k, v in input_batch.items()}
    elif(exec_loc == "gpu"):
        model.to("cuda")
        torch.cuda.synchronize()
        input_batch = {k: v.to('cuda') for k, v in input_batch.items()}
    else:
        raise ValueError(f"Wrong exec_loc, found {exec_loc}, should be 'cpu' or 'gpu'")
    
    #print(f"Allocated memory: {torch.cuda.memory_allocated() / (1024**2):.2f}MB")
    #print(f"Reserved memory: {torch.cuda.memory_reserved() / (1024**2):.2f}MB")
    print("--------")
        
    torch.cuda.synchronize()
    load_time = timer() - start

    model.eval()
    with torch.no_grad():
        output = model.generate(**input_batch,  max_new_tokens=1, min_new_tokens=1, do_sample=False)

    torch.cuda.synchronize()
    inference_time = timer() - (load_time + start)

    #clean-up
    #del model
    torch.cuda.empty_cache()

    total = inference_time + load_time
    return [load_time, inference_time, total]

NR_ITERATIONS = 5
NR_WARMUP_ITERATIONS = 5
if __name__ == "__main__":
    #torch.backends.cudnn.benchmark = False                                                                                                                                                                                                                                                          

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Hugginface model name")
    parser.add_argument("--cold_start", action="store_true", help="Enable cold start")
    parser.add_argument("--model_location", type=str, help="cpu/gpu")
    parser.add_argument("--execution_location", type=str, help="cpu/gpu")
    parser.add_argument("--thread_percentage", type=int, help="for file naming")
    args = parser.parse_args()

    model_name = args.model
    is_cold_start = args.cold_start
    model_loc = args.model_location
    execution_loc = args.execution_location
    thread_percentage = args.thread_percentage

    #warm-up
    if not is_cold_start:
        for i in range(NR_WARMUP_ITERATIONS):
            run_llm(model_loc=model_loc, exec_loc=execution_loc, model_name=model_name)

        print("finished warm-up")

    torch.cuda.empty_cache()
    #measurements
    times_load = []
    times_inference = []
    times_total = []
    for i in range(NR_WARMUP_ITERATIONS):
        res = run_llm(model_loc=model_loc, exec_loc=execution_loc, model_name=model_name)

        times_load.append(res[0])
        times_inference.append(res[1])
        times_total.append(res[2])
        torch.cuda.empty_cache()

    n = len(times_load)

    # Create DataFrame with just the values
    df = pd.DataFrame([
        sum(times_load) / n, max(times_load), min(times_load),
        sum(times_inference) / n, max(times_inference), min(times_inference),
        sum(times_total) / n, max(times_total), min(times_total)
    ])

    #csv_df_string = df.to_csv(index=False)
    #print(csv_df_string)
    dir_path = f"results/{model_name}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_path = f"{dir_path}/{model_loc}-{execution_loc}-{is_cold_start}.csv"
    df.to_csv(file_path, mode='a', header=False, index=False)