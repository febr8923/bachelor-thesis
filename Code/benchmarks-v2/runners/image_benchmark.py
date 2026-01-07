"""
Image Model Benchmark Runner
"""
import os
import time
import torch
import gc
from PIL import Image
from torchvision import transforms
from timeit import default_timer as timer
from typing import Dict, List, Optional

from core.config import *
from core.results import BenchmarkResult
from core.utils import cleanup_memory, start_memory_watcher, stop_memory_watcher


def run_image_iteration(model_loc: str, exec_loc: str, model, batch_size: int = 1) -> Dict:
    """Run single image inference iteration"""
    # Preprocess the input image
    input_image = Image.open('dog.jpg')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    # Create a batch by repeating the image
    input_batch = input_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    # Move model to initial location
    if model_loc == "cpu":
        model.to('cpu')
        input_batch = input_batch.to('cpu')
    elif model_loc == "gpu":
        model.to('cuda')
        input_batch = input_batch.to('cuda')
    else:
        raise ValueError(f"Wrong model_loc: {model_loc}, should be 'cpu' or 'gpu'")

    start = timer()

    # Move to execution location (may be same or different)
    if exec_loc == "cpu":
        model.to('cpu')
        input_batch = input_batch.to('cpu')
    elif exec_loc == "gpu":
        model.to("cuda")
        input_batch = input_batch.to("cuda")
    else:
        raise ValueError(f"Wrong exec_loc: {exec_loc}, should be 'cpu' or 'gpu'")

    if exec_loc == "gpu":
        torch.cuda.synchronize()
    load_time = timer() - start

    # Run inference
    model.eval()
    with torch.no_grad():
        model(input_batch)

    if exec_loc == "gpu":
        torch.cuda.synchronize()
    inference_time = timer() - (load_time + start)

    torch.cuda.empty_cache()

    total = inference_time + load_time
    return {"load": load_time, "execute": inference_time, "total": total}


def run_image_warm(model_loc: str, exec_loc: str, model, batch_size: int = 1) -> Dict:
    """Run warm benchmark with warmup iterations"""
    times_load = []
    times_inference = []
    times_total = []

    cleanup_memory()

    # Warmup
    for _ in range(NR_WARMUP_ITERATIONS):
        run_image_iteration(model_loc, exec_loc, model, batch_size)

    # Actual measurements
    for _ in range(NR_ITERATIONS):
        cleanup_memory()
        res = run_image_iteration(model_loc, exec_loc, model, batch_size)

        times_load.append(res["load"])
        times_inference.append(res["execute"])
        times_total.append(res["total"])

    return {"load": times_load, "execute": times_inference, "totals": times_total}


class ImageBenchmark:
    """Image Model Benchmark Runner"""

    def __init__(self, model_name: str, model_loc: str = "gpu",
                 exec_loc: str = "gpu", measure_memory: bool = False):
        self.model_name = model_name
        self.model_loc = model_loc
        self.exec_loc = exec_loc
        self.measure_memory = measure_memory
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

        result = BenchmarkResult("image")

        for pct in GPU_SM_PERCENTAGES:
            print(f"SM Percentage: {pct}%")
            try:
                os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = f"{pct}"
                model = torch.hub.load('pytorch/vision:v0.10.0', self.model_name, pretrained=True)
                result_i = run_image_warm(self.model_loc, self.exec_loc, model, batch_size=1)
                result.add_raw_result_image(
                    result_i, self.model_name, 1, pct, False, self.model_loc, self.exec_loc
                )
                del model
                cleanup_memory()
                time.sleep(0.5)
            except Exception as e:
                print(f"Failed: {str(e)}")
                if 'model' in locals():
                    del model
                cleanup_memory()
                continue

        return result

    def benchmark_batch_size(self) -> BenchmarkResult:
        """Benchmark with varying batch sizes"""
        print("\n------ Running benchmark_changing_batch_size ------\n")

        batch_sizes = BATCH_SIZES_CPU if self.exec_loc == "cpu" else BATCH_SIZES_GPU
        model = torch.hub.load('pytorch/vision:v0.10.0', self.model_name, pretrained=True)
        result = BenchmarkResult("image")

        for batch_size in batch_sizes:
            print(f"Batch size: {batch_size}")
            try:
                result_i = run_image_warm(self.model_loc, self.exec_loc, model, batch_size)
                result.add_raw_result_image(
                    result_i, self.model_name, batch_size, DEFAULT_THREAD_PERCENTAGE,
                    False, self.model_loc, self.exec_loc
                )
                cleanup_memory()
                time.sleep(0.5)
            except Exception as e:
                print(f"Failed at batch size {batch_size}: {str(e)}")
                cleanup_memory()
                continue

        del model
        return result

    def benchmark_cpu_threads(self) -> BenchmarkResult:
        """Benchmark with varying CPU thread counts"""
        print("\n------ Running cpu_benchmark_changing_nr_threads ------\n")

        result = BenchmarkResult("image")

        for thread_count in CPU_THREAD_COUNTS:
            print(f"Thread count: {thread_count}")
            try:
                os.environ['OMP_NUM_THREADS'] = f"{thread_count}"
                os.environ['MKL_NUM_THREADS'] = f"{thread_count}"

                model = torch.hub.load('pytorch/vision:v0.10.0', self.model_name, pretrained=True)
                result_i = run_image_warm(self.model_loc, self.exec_loc, model, batch_size=1)
                result.add_raw_result_image(
                    result_i, self.model_name, 1, thread_count, False, self.model_loc, self.exec_loc
                )
                del model
                cleanup_memory()
                time.sleep(0.5)
            except Exception as e:
                print(f"Failed: {str(e)}")
                if 'model' in locals():
                    del model
                cleanup_memory()
                continue

        return result

    def run(self, mode: int = 3) -> List[BenchmarkResult]:
        """
        Run benchmarks based on mode

        Args:
            mode: Benchmark mode (1-3, 5)

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
                res.save_to_csv(dir_path, f"{self.model_loc}-{self.exec_loc}-1")
                results.append(res)

            elif mode == 2:
                # Batch size
                res = self.benchmark_batch_size()
                res.save_to_csv(dir_path, f"{self.model_loc}-{self.exec_loc}-2")
                results.append(res)

            elif mode in [3, 5]:
                # All applicable benchmarks
                if self.exec_loc == "gpu":
                    res = self.benchmark_sm_percentage()
                    res.save_to_csv(dir_path, f"{self.model_loc}-{self.exec_loc}-1")
                    results.append(res)
                else:
                    res = self.benchmark_cpu_threads()
                    res.save_to_csv(dir_path, f"{self.model_loc}-{self.exec_loc}-1")
                    results.append(res)

                res = self.benchmark_batch_size()
                res.save_to_csv(dir_path, f"{self.model_loc}-{self.exec_loc}-2")
                results.append(res)

            else:
                raise ValueError(f"Mode {mode} not supported for image models")

        finally:
            self._stop_watcher()

        return results
