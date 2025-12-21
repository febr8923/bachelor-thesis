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

def run_image(model_loc: str, exec_loc: str, model_name: str):
    global model
    if model is None:
        model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)

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

    times_load = []
    times_inference = []
    times_total = []
    #warm-up
    if not is_cold_start:
        for i in range(NR_WARMUP_ITERATIONS):
            run_image(model_loc=model_loc, exec_loc=execution_loc, model_name=model_name)

        print("finished warm-up")

        torch.cuda.empty_cache()
        #measurements

        for i in range(NR_ITERATIONS):
            res = run_image(model_loc=model_loc, exec_loc=execution_loc, model_name=model_name)

            times_load.append(res[0])
            times_inference.append(res[1])
            times_total.append(res[2])
            torch.cuda.empty_cache()

    else:
        res = run_image(model_loc=model_loc, exec_loc=execution_loc, model_name=model_name)
        times_load.append(res[0])
        times_inference.append(res[1])
        times_total.append(res[2])
        torch.cuda.empty_cache()

    n = len(times_load)


    df = pd.DataFrame({
        "avg_load_time": sum(times_load) / n,
        "max_load_time": max(times_load),
        "min_load_time": min(times_load),
        "avg_inference_time": sum(times_inference) / n,
        "max_inference_time": max(times_inference),
        "min_inference_time": min(times_inference),
        "avg_total_time": sum(times_total) / n,
        "max_total_time": max(times_total),
        "min_total_time": min(times_total)
    })


    #csv_df_string = df.to_csv(index=False)
    #print(csv_df_string)
    dir_path = f"results/{model_name}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_path = f"{dir_path}/{model_loc}-{execution_loc}-{is_cold_start}.csv"
    df.to_csv(file_path, mode='a', header=False, index=False)