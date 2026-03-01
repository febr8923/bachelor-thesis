"""
Single-run DL (torchvision) CPU memory profiler.

Loads a torchvision model, runs one inference on the CPU, and captures RAM
usage over time via CpuWatcher (psutil, 0.5 s intervals).

Usage (run from benchmarks/):
  python single_run/dl_memory_cpu.py --model resnet50
  python single_run/dl_memory_cpu.py --model vgg19 \
      --output_csv results/vgg19/single_cpu_memory.csv
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
from watcher import CpuWatcher


def load_input():
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(Image.open("dog.jpg")).unsqueeze(0)


def run(model_name, output_csv):
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    watcher = CpuWatcher(id=0, save_loc=output_csv, interval=0.5)
    watcher.start()

    print(f"Loading model: {model_name} (CPU)")
    model = torch.hub.load("pytorch/vision:v0.10.0", model_name, pretrained=True)
    model.to("cpu")
    model.eval()

    input_batch = load_input().to("cpu")

    t0 = timer()
    # no-op transfer — data is already on CPU; timer mirrors the GPU script structure
    load_time = timer() - t0

    with torch.no_grad():
        t1 = timer()
        model(input_batch)
        infer_time = timer() - t1
    print(f"  load/transfer time: {load_time:.4f} s")
    print(f"  inference time    : {infer_time:.4f} s")

    time.sleep(1.0)
    watcher.stop()

    print(f"\nMemory trace saved to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single torchvision CPU inference with RAM profiling"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Torchvision model name (e.g. resnet50, vgg19, alexnet)")
    parser.add_argument("--output_csv", type=str, default="dl_cpu_memory_trace.csv",
                        help="Output CSV path for RAM time-series")

    args = parser.parse_args()
    run(args.model, args.output_csv)
