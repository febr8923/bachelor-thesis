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
from watcher import GpuWatcher
from watcher import CpuWatcher

#mode can be 1 (CPU,CPU), 2 (CPU,GPU), 3 (GPU,CPU), 4 (GPU,GPU) for (model location, execution location)
#model should be on gpu/cpu, execution should be on gpu/cpu 
import gc
import time
os.environ['TRANSFORMERS_CACHE'] = '/iopsstor/scratch/cscs/fbrunne'

NR_ITERATIONS = 5
NR_WARMUP_ITERATIONS = 5

DEFAULT_MEM_UTIL = 0.85
DEFAULT_THREAD_PERCENTAGE = 100
DEFAULT_NR_BATCHES = 1
DEFAULT_NR_INPUT_TOKENS = 20


def run_image_iteration(model_loc: str, exec_loc: str, model):

    # Preprocess the input image
    input_image = Image.open('dog.jpg')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    #print(f"Allocated memory: {torch.cuda.memory_allocated() / (1024**2):.2f}MB")
    #print(f"Reserved memory: {torch.cuda.memory_reserved() / (1024**2):.2f}MB")

    if(model_loc == "cpu"):#start with model on cpu execution on cpu
        model.to('cpu')
        input_batch = input_batch.to('cpu')
    elif(model_loc == "gpu"):
        model.to('cuda')
        input_batch = input_batch.to('cuda')
    else:
        raise ValueError(f"Wrong model_loc, found {model_loc}, should be 'cpu' or 'gpu'")

    start = timer()

    if(exec_loc == "cpu"):
        model.to('cpu')
        input_batch = input_batch.to('cpu')
    elif(exec_loc == "gpu"):
        model.to("cuda")
        input_batch = input_batch.to("cuda")
    else:
        raise ValueError(f"Wrong exec_loc, found {exec_loc}, should be 'cpu' or 'gpu'")

    #print(f"Allocated memory: {torch.cuda.memory_allocated() / (1024**2):.2f}MB")
    #print(f"Reserved memory: {torch.cuda.memory_reserved() / (1024**2):.2f}MB")
    #print("--------")

    torch.cuda.synchronize()
    load_time = timer() - start

    model.eval()
    with torch.no_grad():
        model(input_batch)

    torch.cuda.synchronize()
    inference_time = timer() - (load_time + start)

    torch.cuda.empty_cache()

    total = inference_time + load_time
    return [load_time, inference_time, total]

def run_image_warm(model_loc, exec_loc, model, model_name):
    if model is None:
        model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
    times_load = []
    times_inference = []
    times_total = []

    for i in range(NR_WARMUP_ITERATIONS):
        run_image_iteration(model_loc=model_loc, exec_loc=exec_loc, model=model)
    for i in range(NR_ITERATIONS):
        
        res = run_image_iteration(model_loc=model_loc, exec_loc=exec_loc, model=model)

        times_load.append(res[0])
        times_inference.append(res[1])
        times_total.append(res[2])
        torch.cuda.empty_cache()
    
    return {"load": times_load, "execute": times_inference, "totals": times_total}

def run_image(model_loc, exec_loc, model_name, is_cold_start):
    model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
    
    if is_cold_start:
        # Single cold start run
        res = run_image_iteration(model_loc=model_loc, exec_loc=exec_loc, model=model)
        times_load = [res[0]]
        times_inference = [res[1]]
        times_total = [res[2]]
    else:
        # Warm start with multiple iterations
        result = run_image_warm(model_loc=model_loc, exec_loc=exec_loc, model=model, model_name=model_name)
        times_load = result["load"]
        times_inference = result["execute"]
        times_total = result["totals"]
    
    return times_load, times_inference, times_total


if __name__ == "__main__":
    #torch.backends.cudnn.benchmark = False

    for i in range(10, 101, 10):
        print(f"Percentage: {i}")
        try:
            os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = f"{i}"

            result_i = run_image_warm(model_loc = model_loc, exec_loc=exec_loc, model=model)
            result.add_raw_result(result_i, nr_input_tokens=DEFAULT_NR_INPUT_TOKENS, nr_batches=DEFAULT_NR_BATCHES,
                      thread_percentage=i, memory_rate=DEFAULT_MEM_UTIL,
                      cold_start=cold_start, model_loc=model_loc, exec_loc=exec_loc)

            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(0.5)
        except Exception as e:
            print(f"Failed")
            torch.cuda.empty_cache()
            gc.collect()
            continue
        
    return result


if __name__ == "__main__":
    print("")
    print("------ Running cpu_benchmark_changing_nr_threads ------")
    print("")

    cold_start = False
    exec_loc = "cpu"
    model_loc = "cpu"
    nr_outputs = 32

    #llm = LLM(model_name, max_num_batched_tokens=2048, max_model_len=2048)


    prompt = make_prompt(nr_tokens=DEFAULT_NR_INPUT_TOKENS, model_name=model_name, nr_batches=DEFAULT_NR_BATCHES)
    result = BenchmarkResult()

    for i in [16, 36, 72]:
        print(f"Nr threads: {i}")
        llm = LLM(model_name, max_num_batched_tokens=2048, max_model_len=2048)
        try:
            os.environ['OMP_NUM_THREADS'] = f"{i}"
            os.environ['MKL_NUM_THREADS'] = f"{i}"

            llm = LLM(model_name, max_num_batched_tokens=2048, max_model_len=2048)

            result_i = run_vllm_warm(llm, prompt, nr_outputs)
            result.add_raw_result(result_i, nr_input_tokens=DEFAULT_NR_INPUT_TOKENS, nr_batches=DEFAULT_NR_BATCHES, 
                        thread_percentage=i, memory_rate=100, 
                        cold_start=cold_start, model_loc=model_loc, exec_loc=exec_loc)

            del llm
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(0.5)
        except Exception as e:
            print(f"Failed with {e}")
            if 'llm' in locals():
                del llm
            torch.cuda.empty_cache()
            gc.collect()
            continue
    
    return result


if __name__ == "__main__":
    #torch.backends.cudnn.benchmark = False                                                                                                                                                                                                                                                          

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Hugginface model name")
    parser.add_argument("--cold_start", action="store_true", help="Enable cold start")
    parser.add_argument("--model_location", type=str, help="cpu/gpu")
    parser.add_argument("--execution_location", type=str, help="cpu/gpu")
    parser.add_argument("--thread_percentage", type=int, help="for file naming")
    parser.add_argument("--mode", type=int, help="benchark mode (1: changing batch size, 2: changing input length, 3: changing gpu memory utilization, 4: changing cpu nr threads")
    parser.add_argument("--measure_memory", action="store_true", help="Enable memory measurement")
    args = parser.parse_args()

    model_name = args.model
    is_cold_start = args.cold_start
    model_loc = args.model_location
    execution_loc = args.execution_location
    thread_percentage = args.thread_percentage
    mode = args.mode
    measure_memory = args.measure_memory

    dir_path = f"results/{model_name}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if measure_memory:
        if execution_loc == "gpu":
            watcher = GpuWatcher(gpu_id=0, save_loc=f"{dir_path}/gpu-{mode}-memory.csv")
        else:
            watcher = CpuWatcher(id=0, save_loc=f"{dir_path}/cpu-{mode}-memory.csv")
        watcher.start()

    # Run inference (handles both warm and cold start internally)
    times_load, times_inference, times_total = run_image(
        model_loc=model_loc, 
        exec_loc=execution_loc, 
        model_name=model_name,
        is_cold_start=is_cold_start
    )

    if measure_memory:
        watcher.stop()

    n = len(times_load)

    df = pd.DataFrame({
        "avg_load_time": [sum(times_load) / n],
        "max_load_time": [max(times_load)],
        "min_load_time": [min(times_load)],
        "avg_inference_time": [sum(times_inference) / n],
        "max_inference_time": [max(times_inference)],
        "min_inference_time": [min(times_inference)],
        "avg_total_time": [sum(times_total) / n],
        "max_total_time": [max(times_total)],
        "min_total_time": [min(times_total)]
    })


    #csv_df_string = df.to_csv(index=False)
    #print(csv_df_string)

    file_path = f"{dir_path}/{model_loc}-{execution_loc}-{mode}-{is_cold_start}.csv"
    df.to_csv(file_path, mode='a', header=False, index=False)