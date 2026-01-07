
"""
Default parameters
"""
DEFAULT_IS_COLD_START = False
DEFAULT_NUM_INPUT_TOKENS = 128
DEFAULT_MAX_OUTPUT_TOKENS = 256
DEFAULT_BATCH_SIZE = 1
DEFAULT_SM_PERCENTAGE = 100
DEFAULT_THREAD_NR = 36
DEFAULT_MEMORY_UTIL = 0.85

NUM_INPUT_TOKENS_GPU = [64, 128, 256, 512, 1024, 2048]
NUM_INPUT_TOKENS_CPU = [32, 64, 128, 256]
BATCH_SIZES_CPU = [1, 2, 4, 8, 16, 32, 64]
BATCH_SIZES_GPU = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
THREAD_NRS = [1, 2, 4, 8, 16, 36, 72]
SM_PERCENTAGES = list(range(10, 101, 10))
MEMORY_UTILS = [i/100 for i in range(20, 91, 10)]

NR_ITERATIONS = 5
NR_WARUMUP_ITERATIONS = 5

"""
Benchmark modes
"""
BENCHMARK_MODES = {
    1: "resource_variation",  # SM% (GPU) or threads (CPU)
    2: "batch_size",          # Varying batch sizes
    3: "input_length",        # Varying input lengths (LLM only)
    4: "memory_util",         # GPU memory utilization (GPU only)
    5: "all",                 # Run all applicable benchmarks
}
 
RESULTS_DIR = "results"
PLOTS_DIR = "plots"


"""
Result schemas
"""
RESULTS_EXECUTION = [
    "TTFT",
    "ITL",
    "TPS",
    "e2e_inference_latency",
    "throughput",
    "load_duration",
]

RESULTS_CPU_MEMORY = [
    'timestamp', 
    'poi_type',
    'cpu_util_pct', 
    'mem_total_bytes', 
    'mem_used_bytes', 
    'mem_available_bytes'
]

RESULTS_GPU_MEMORY = [
    "timestamp", 
    "poi_type",
    "gpu_id", 
    "gpu_util%", 
    "mem_util%", 
    "mem_total_mb", 
    "mem_free_mb", 
    "mem_used_mb", 
    "mem_reserved_mb"
]

PARAMETERS_VLLM_GPU = [
    "model_name",
    "model_location",
    "execution_location",
    "is_cold_start",
    "num_input_tokens",
    "max_output_tokens",
    "batch_size",
    "thread_percentage",
    "memory_percentage",
]

PARAMETERS_VLLM_CPU = [
    "model_name",
    "model_location",
    "execution_location",
    "is_cold_start",
    "num_input_tokens",
    "max_output_tokens",
    "batch_size",
    "thread_nr",
    "memory_percentage",
]

PARAMETERS_DL = [
    "model_location",
    "execution_location",
    "is_cold_start",
    "num_input_tokens",
    "max_output_tokens",
    "batch_size",
    "thread_percentage",
    "memory_percentage",
]
