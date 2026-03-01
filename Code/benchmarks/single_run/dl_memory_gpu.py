"""
Single-run DL (torchvision) GPU memory profiler.

Loads a torchvision model, runs one inference on the GPU, and captures GPU
memory usage over time via GpuWatcher (nvidia-smi, 10 ms intervals).

model_loc controls where the data starts (cpu = includes H2D transfer in
load_time, gpu = data already on GPU).

Usage (run from benchmarks/):
  python single_run/dl_memory_gpu.py --model resnet50
  python single_run/dl_memory_gpu.py --model vgg19 --model_loc cpu \
      --output_csv results/vgg19/single_gpu_memory.csv
"""

import argparse
import os
import sys
import time
from timeit import default_timer as timer

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from watcher import GpuWatcher


def load_input():
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(Image.open("dog.jpg")).unsqueeze(0)


def run(model_name, model_loc, output_csv):
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    watcher = GpuWatcher(gpu_id=0, save_loc=output_csv)
    watcher.start()

    print(f"Loading model: {model_name}")
    model = torch.hub.load("pytorch/vision:v0.10.0", model_name, pretrained=True)
    model.eval()

    input_batch = load_input()

    # place model and data at starting location
    if model_loc == "cpu":
        model.to("cpu")
        input_batch = input_batch.to("cpu")
    else:
        model.to("cuda")
        input_batch = input_batch.to("cuda")

    t0 = timer()
    model.to("cuda")
    input_batch = input_batch.to("cuda")
    torch.cuda.synchronize()
    load_time = timer() - t0
    print(f"  load/transfer time: {load_time:.4f} s")

    with torch.no_grad():
        t1 = timer()
        model(input_batch)
        torch.cuda.synchronize()
        infer_time = timer() - t1
    print(f"  inference time    : {infer_time:.4f} s")

    torch.cuda.empty_cache()
    time.sleep(0.5)
    watcher.stop()

    print(f"\nMemory trace saved to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single torchvision GPU inference with GPU memory profiling"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Torchvision model name (e.g. resnet50, vgg19, alexnet)")
    parser.add_argument("--model_loc", type=str, default="cpu", choices=["cpu", "gpu"],
                        help="Where data starts before GPU transfer (default: cpu)")
    parser.add_argument("--output_csv", type=str, default="dl_gpu_memory_trace.csv",
                        help="Output CSV path for GPU memory time-series")

    args = parser.parse_args()
    run(args.model, args.model_loc, args.output_csv)
