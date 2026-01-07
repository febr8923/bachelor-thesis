import torch
import os
from PIL import Image
from torchvision import transforms
from timeit import default_timer as timer
import pandas as pd
import argparse
from watcher import GpuWatcher
from watcher import CpuWatcher

os.environ['TRANSFORMERS_CACHE'] = '/iopsstor/scratch/cscs/fbrunne'

NR_ITERATIONS = 5
NR_WARMUP_ITERATIONS = 5


def run_image_iteration(model_loc: str, exec_loc: str, model):
    """Run a single inference iteration on an image."""
    # Preprocess the input image
    input_image = Image.open('dog.jpg')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    # Move model and data to initial location
    if model_loc == "cpu":
        model.to('cpu')
        input_batch = input_batch.to('cpu')
    elif model_loc == "gpu":
        model.to('cuda')
        input_batch = input_batch.to('cuda')
    else:
        raise ValueError(f"Wrong model_loc, found {model_loc}, should be 'cpu' or 'gpu'")

    start = timer()

    # Move to execution location
    if exec_loc == "cpu":
        model.to('cpu')
        input_batch = input_batch.to('cpu')
    elif exec_loc == "gpu":
        model.to("cuda")
        input_batch = input_batch.to("cuda")
    else:
        raise ValueError(f"Wrong exec_loc, found {exec_loc}, should be 'cpu' or 'gpu'")

    torch.cuda.synchronize()
    load_time = timer() - start

    # Run inference
    model.eval()
    with torch.no_grad():
        model(input_batch)

    torch.cuda.synchronize()
    inference_time = timer() - (load_time + start)

    torch.cuda.empty_cache()

    total = inference_time + load_time
    return [load_time, inference_time, total]


def run_image_warm(model_loc: str, exec_loc: str, model, model_name: str):
    """Run warm inference with warmup iterations."""
    times_load = []
    times_inference = []
    times_total = []

    # Warmup iterations
    for i in range(NR_WARMUP_ITERATIONS):
        run_image_iteration(model_loc=model_loc, exec_loc=exec_loc, model=model)
    
    # Measurement iterations
    for i in range(NR_ITERATIONS):
        res = run_image_iteration(model_loc=model_loc, exec_loc=exec_loc, model=model)
        times_load.append(res[0])
        times_inference.append(res[1])
        times_total.append(res[2])
        torch.cuda.empty_cache()
    
    return {"load": times_load, "execute": times_inference, "totals": times_total}


def run_image(model_loc: str, exec_loc: str, model_name: str, is_cold_start: bool):
    """Run image inference in either cold or warm mode."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Torchvision model name (e.g., resnet50, vgg19)")
    parser.add_argument("--cold_start", action="store_true", help="Enable cold start")
    parser.add_argument("--model_location", type=str, help="cpu/gpu")
    parser.add_argument("--execution_location", type=str, help="cpu/gpu")
    parser.add_argument("--mode", type=int, help="benchmark mode for file naming")
    parser.add_argument("--measure_memory", action="store_true", help="Enable memory measurement")
    args = parser.parse_args()

    model_name = args.model
    is_cold_start = args.cold_start
    model_loc = args.model_location
    execution_loc = args.execution_location
    mode = args.mode
    measure_memory = args.measure_memory

    dir_path = f"results/{model_name}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if measure_memory:
        if execution_loc == "gpu":
            watcher = GpuWatcher(gpu_id=0, save_loc=f"{dir_path}/gpu-{mode}-memory.csv")
        else:
            watcher = CpuWatcher(id=0, save_loc=f"{dir_path}/cpu-{mode}-memory.csv", interval=0.01)
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

    file_path = f"{dir_path}/{model_loc}-{execution_loc}-{mode}-{is_cold_start}.csv"
    df.to_csv(file_path, mode='a', header=False, index=False)
