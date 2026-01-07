"""
Utility functions shared across benchmarks
"""
import torch
import gc
import time
import os
from typing import Optional
from core.watcher import GpuWatcher, CpuWatcher


def cleanup_memory():
    """Clean up GPU and system memory"""
    torch.cuda.empty_cache()
    gc.collect()


def setup_environment(transformers_cache: Optional[str] = None):
    """Setup environment variables for benchmarks"""
    if transformers_cache:
        os.environ['TRANSFORMERS_CACHE'] = transformers_cache
        os.environ['TORCH_HOME'] = transformers_cache
        os.environ['HF_HOME'] = transformers_cache
    os.environ['PYTHONWARNINGS'] = 'ignore'


def setup_gpu_mps():
    """Setup CUDA MPS for GPU benchmarks"""
    os.environ['CUDA_MPS_PIPE_DIRECTORY'] = '/tmp/nvidia-mps'
    os.environ['CUDA_MPS_LOG_DIRECTORY'] = f'/tmp/nvidia-log-{os.getuid()}'


def start_memory_watcher(exec_loc: str, model_name: str, mode: int,
                        suffix: str = "") -> Optional[object]:
    """
    Start memory monitoring watcher

    Args:
        exec_loc: Execution location ("gpu" or "cpu")
        model_name: Name of the model being benchmarked
        mode: Benchmark mode number
        suffix: Optional suffix for filename

    Returns:
        Watcher object or None
    """
    dir_path = f"results/{model_name}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    filename = f"{exec_loc}-{mode}{suffix}-memory.csv"
    save_loc = os.path.join(dir_path, filename)

    if exec_loc == "gpu":
        watcher = GpuWatcher(gpu_id=0, save_loc=save_loc)
    elif exec_loc == "cpu":
        watcher = CpuWatcher(id=0, save_loc=save_loc)
    else:
        return None

    watcher.start()
    return watcher


def stop_memory_watcher(watcher: Optional[object]):
    """Stop memory monitoring watcher"""
    if watcher:
        time.sleep(0.5)
        watcher.stop()


def validate_device(device: str) -> str:
    """Validate and return device string"""
    device = device.lower()
    if device not in ["cpu", "gpu"]:
        raise ValueError(f"Invalid device: {device}. Must be 'cpu' or 'gpu'")
    return device
