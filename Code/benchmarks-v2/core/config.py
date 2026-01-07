"""
Centralized configuration for all benchmarks
"""
import os

# Benchmark parameters
NR_ITERATIONS = 5
NR_WARMUP_ITERATIONS = 5
DEFAULT_THREAD_PERCENTAGE = 100
DEFAULT_BATCH_SIZE = 1

# Model cache locations
TRANSFORMERS_CACHE = os.environ.get('TRANSFORMERS_CACHE', '/iopsstor/scratch/cscs/fbrunne')
TORCH_HOME = os.environ.get('TORCH_HOME', TRANSFORMERS_CACHE)
HF_HOME = os.environ.get('HF_HOME', TRANSFORMERS_CACHE)

# vLLM specific
DEFAULT_VLLM_MEM_UTIL = 0.85
DEFAULT_NR_INPUT_TOKENS = 20
DEFAULT_NR_OUTPUT_TOKENS = 128

# Benchmark modes
BENCHMARK_MODES = {
    1: "resource_variation",  # SM% (GPU) or threads (CPU)
    2: "batch_size",          # Varying batch sizes
    3: "input_length",        # Varying input lengths (LLM only)
    4: "memory_util",         # GPU memory utilization (GPU only)
    5: "all",                 # Run all applicable benchmarks
}

# Device configurations
VALID_DEVICES = ["cpu", "gpu"]

# Thread configurations for CPU benchmarks
CPU_THREAD_COUNTS = [16, 36, 72]

# SM percentage configurations for GPU benchmarks
GPU_SM_PERCENTAGES = list(range(10, 101, 10))

# Batch size configurations
BATCH_SIZES_CPU = [1, 2, 4, 8, 16, 32, 64]
BATCH_SIZES_GPU = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# Input length configurations (for LLM benchmarks)
INPUT_LENGTHS = [64, 128, 256, 512, 1024, 2048]

# Memory utilization configurations
MEMORY_UTILS = [i/100 for i in range(20, 91, 10)]

# Results directory
RESULTS_DIR = "results"
PLOTS_DIR = "plots"
