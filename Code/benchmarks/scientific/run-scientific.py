#!/usr/bin/env python3
"""
Scientific Benchmarks Runner

This script runs scientific benchmarks (BFS, NN, Leukocyte) with four execution modes:
- cpu-cpu: Data initially on CPU, execution on CPU
- cpu-gpu: Data initially on CPU, execution on GPU (includes transfer time)
- gpu-cpu: Data initially on GPU, execution on CPU (includes transfer time)
- gpu-gpu: Data initially on GPU, execution on GPU

The output format is compatible with the vLLM benchmark results.

Run from the 'scientific/' directory:
  python run-scientific.py --benchmark bfs --mode all
"""

import os
import subprocess
import sys
import time
import argparse
import re
import pandas as pd
from pathlib import Path

# Add parent directory to path for watcher imports
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from watcher import GpuWatcher, CpuWatcher
    WATCHER_AVAILABLE = True
except ImportError:
    WATCHER_AVAILABLE = False
    print("Warning: watcher module not available, memory monitoring disabled")

NR_ITERATIONS = 5
NR_WARMUP_ITERATIONS = 2

# Benchmark configurations
# Note: paths are relative to the benchmark's working directory
BENCHMARKS = {
    "bfs": {
        "cpu_dir": "bfs",
        "gpu_dir": "bfs_cuda",
        "cpu_executable": "bfs",
        "gpu_executable": "bfs",
        "gpu_cpu_executable": "bfs_gpu_cpu",  # GPU-CPU hybrid mode
        "cpu_args": ["4", "../data/bfs/graph1MW_6.txt"],  # num_threads, input_file
        "gpu_args": ["../data/bfs/graph1MW_6.txt"],  # input_file
        "gpu_cpu_args": ["../data/bfs/graph1MW_6.txt", "4"],  # input_file, num_threads
        "timing_patterns": {
            "total": r"Total time:\s+([\d.]+)\s*ms",
            "data_transfer": r"CPU->GPU data transfer:\s+([\d.]+)\s*ms",
            "gpu_cpu_transfer": r"GPU->CPU data transfer:\s+([\d.]+)\s*ms",
            "computation": r"Pure BFS computation:\s+([\d.]+)\s*ms",
            "cpu_computation": r"CPU computation \(OpenMP\):\s+([\d.]+)\s*ms",
            "cpu_compute": r"Compute time:\s+([\d.]+)"  # seconds for CPU version
        }
    },
    "nn": {
        "cpu_dir": "nn",
        "gpu_dir": "nn_cuda",
        "cpu_executable": "nn_walltime",
        "gpu_executable": "nn",
        "gpu_cpu_executable": "nn_gpu_cpu",  # GPU-CPU hybrid mode
        "cpu_args": ["filelist_4", "5", "30", "90"],  # filelist, k, lat, lng
        "gpu_args": ["filelist_4", "-r", "5", "-lat", "30", "-lng", "90"],
        "gpu_cpu_args": ["filelist_4", "-r", "5", "-lat", "30", "-lng", "90"],  # same args
        "timing_patterns": {
            "total": r"Total time:\s+([\d.]+)\s*ms",
            "data_transfer": r"CPU->GPU data transfer:\s+([\d.]+)\s*ms",
            "gpu_cpu_transfer": r"GPU->CPU data transfer:\s+([\d.]+)\s*ms",
            "computation": r"Pure NN computation:\s+([\d.]+)\s*ms",
            "cpu_computation": r"CPU computation \(OpenMP\):\s+([\d.]+)\s*ms",
            "cpu_total": r"total time\s*:\s*([\d.]+)\s*s",  # CPU version uses seconds
        }
    },
    "leukocyte": {
        "cpu_dir": "leukocyte",
        "gpu_dir": "leukocyte_cuda",
        "cpu_executable": "OpenMP/leukocyte",
        "gpu_executable": "CUDA/leukocyte",
        "gpu_cpu_executable": "CUDA/leukocyte_gpu_cpu",  # GPU-CPU hybrid mode
        "cpu_args": ["5", "4", "../data/leukocyte/testfile.avi"],  # num_frames, num_threads, input_file
        "gpu_args": ["../data/leukocyte/testfile.avi", "5"],  # input_file, num_frames
        "gpu_cpu_args": ["../data/leukocyte/testfile.avi", "5", "4"],  # input_file, num_frames, num_threads
        "timing_patterns": {
            "total": r"Total time:\s+([\d.]+)\s*ms",
            "data_transfer": r"CPU->GPU data transfer:\s+([\d.]+)\s*ms",
            "gpu_cpu_transfer": r"GPU->CPU data transfer:\s+([\d.]+)\s*ms",
            "computation": r"Pure computation:\s+([\d.]+)\s*ms",
            "cpu_computation": r"CPU computation \(OpenMP\):\s+([\d.]+)\s*ms",
            "cpu_total": r"Total application run time:\s+([\d.]+)\s*seconds",  # CPU version uses seconds
        }
    }
}


class BenchmarkResult:
    """Class to store and manage benchmark results in a format compatible with vLLM benchmarks."""

    def __init__(self):
        self.data = pd.DataFrame({
            "benchmark": [],
            "data_loc": [],      # where data starts: cpu or gpu
            "exec_loc": [],      # where execution happens: cpu or gpu
            "sm_percentage": [],  # GPU SM percentage (via CUDA_MPS_ACTIVE_THREAD_PERCENTAGE)
            "num_threads": [],    # CPU thread count (via OMP_NUM_THREADS)
            "iteration": [],
            "total_time_ms": [],
            "data_transfer_time_ms": [],
            "computation_time_ms": [],
            "avg_total_ms": [],
            "max_total_ms": [],
            "min_total_ms": [],
            "avg_transfer_ms": [],
            "avg_computation_ms": [],
        })

    def add_datapoint(self, datapoint):
        """Add a datapoint (DataFrame row) to the results."""
        if isinstance(datapoint, pd.DataFrame):
            self.data = pd.concat([self.data, datapoint], ignore_index=True)
        else:
            raise ValueError("Invalid datapoint type")

    def add_raw_results(self, benchmark_name, data_loc, exec_loc,
                        total_times, transfer_times, computation_times,
                        sm_percentage=100, num_threads=4):
        """Add raw benchmark results and compute statistics."""
        n = len(total_times)
        if n == 0:
            return

        # Handle None values in transfer and computation times
        transfer_times = [t if t is not None else 0.0 for t in transfer_times]
        computation_times = [c if c is not None else 0.0 for c in computation_times]

        datapoint = pd.DataFrame({
            "benchmark": [benchmark_name],
            "data_loc": [data_loc],
            "exec_loc": [exec_loc],
            "sm_percentage": [sm_percentage],
            "num_threads": [num_threads],
            "iteration": [n],
            "total_time_ms": [total_times[-1] if total_times else 0],  # last iteration
            "data_transfer_time_ms": [transfer_times[-1] if transfer_times else 0],
            "computation_time_ms": [computation_times[-1] if computation_times else 0],
            "avg_total_ms": [sum(total_times) / n],
            "max_total_ms": [max(total_times)],
            "min_total_ms": [min(total_times)],
            "avg_transfer_ms": [sum(transfer_times) / n if transfer_times else 0],
            "avg_computation_ms": [sum(computation_times) / n if computation_times else 0],
        })
        self.add_datapoint(datapoint)

    def save_to_csv(self, dir_path, name):
        """Save results to CSV file."""
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        file_path = f"{dir_path}/{name}.csv"
        self.data.to_csv(file_path, mode='w', header=True, index=False)
        print(f"Results saved to {file_path}")


def parse_output(output, patterns):
    """Parse benchmark output using regex patterns to extract timing information."""
    results = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            results[key] = float(match.group(1))
        else:
            results[key] = None
    return results


def run_benchmark_iteration(benchmark_name, data_loc, exec_loc, base_dir,
                            sm_percentage=100, num_threads=4):
    """
    Run a single benchmark iteration.

    Args:
        benchmark_name: Name of the benchmark (bfs, nn, leukocyte)
        data_loc: Where data initially resides (cpu or gpu)
        exec_loc: Where execution happens (cpu or gpu)
        base_dir: Base directory of the scientific benchmarks
        sm_percentage: GPU SM percentage (1-100) via CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
        num_threads: CPU thread count via OMP_NUM_THREADS

    Returns:
        Dictionary with timing results
    """
    config = BENCHMARKS[benchmark_name]

    # Determine which executable and arguments to use based on mode
    if data_loc == "gpu" and exec_loc == "cpu":
        # GPU-CPU mode: data on GPU, execution on CPU (hybrid executable)
        if "gpu_cpu_executable" not in config:
            print(f"Warning: {benchmark_name} does not support gpu-cpu mode")
            return None
        work_dir = os.path.join(base_dir, config["gpu_dir"])
        executable = os.path.join(work_dir, config["gpu_cpu_executable"])
        # For gpu-cpu mode, update the thread count in args if applicable
        args = list(config.get("gpu_cpu_args", config["gpu_args"]))
        # BFS gpu-cpu takes num_threads as second arg
        if benchmark_name == "bfs" and len(args) >= 2:
            args[1] = str(num_threads)
        # Leukocyte gpu-cpu takes num_threads as third arg
        elif benchmark_name == "leukocyte" and len(args) >= 3:
            args[2] = str(num_threads)
    elif exec_loc == "gpu":
        # CPU-GPU mode (standard CUDA): data on CPU, execution on GPU
        work_dir = os.path.join(base_dir, config["gpu_dir"])
        executable = os.path.join(work_dir, config["gpu_executable"])
        args = config["gpu_args"]
    else:
        # CPU-CPU mode: data on CPU, execution on CPU
        work_dir = os.path.join(base_dir, config["cpu_dir"])
        executable = os.path.join(work_dir, config["cpu_executable"])
        # Update thread count in args for BFS and leukocyte
        args = list(config["cpu_args"])
        if benchmark_name == "bfs" and len(args) >= 1:
            args[0] = str(num_threads)
        elif benchmark_name == "leukocyte" and len(args) >= 2:
            args[1] = str(num_threads)

    # Build command
    cmd = [executable] + args

    # Set environment variables for the subprocess
    # Also set in os.environ so they're inherited by subprocess
    if exec_loc == "gpu":
        os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = str(sm_percentage)

    if exec_loc == "cpu":
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        os.environ['MKL_NUM_THREADS'] = str(num_threads)

    try:
        start_time = time.perf_counter()
        result = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        end_time = time.perf_counter()

        wall_time_ms = (end_time - start_time) * 1000

        if result.returncode != 0:
            print(f"Warning: {benchmark_name} returned non-zero exit code {result.returncode}")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            print(f"command: {' '.join(cmd)}")
            print(f"cwd: {work_dir}")
            return None

        output = result.stdout + result.stderr

        # Parse timing from output
        parsed = parse_output(output, config["timing_patterns"])

        # Construct result dictionary
        timing_result = {
            "wall_time_ms": wall_time_ms,
            "total_ms": None,
            "transfer_ms": None,
            "computation_ms": None
        }

        # Extract timing based on mode (data_loc, exec_loc combination)
        if data_loc == "gpu" and exec_loc == "cpu":
            # GPU-CPU mode: GPU->CPU transfer + CPU computation
            timing_result["total_ms"] = parsed.get("total")
            timing_result["transfer_ms"] = parsed.get("gpu_cpu_transfer")
            timing_result["computation_ms"] = parsed.get("cpu_computation")
        elif benchmark_name == "bfs":
            if exec_loc == "gpu":
                timing_result["total_ms"] = parsed.get("total")
                timing_result["transfer_ms"] = parsed.get("data_transfer")
                timing_result["computation_ms"] = parsed.get("computation")
            else:
                # CPU version reports compute time in seconds
                cpu_compute = parsed.get("cpu_compute")
                if cpu_compute:
                    timing_result["total_ms"] = cpu_compute * 1000  # convert to ms
                    timing_result["computation_ms"] = cpu_compute * 1000
                else:
                    timing_result["total_ms"] = wall_time_ms
        elif benchmark_name == "nn":
            if exec_loc == "gpu":
                # GPU version reports in ms
                timing_result["total_ms"] = parsed.get("total")
                timing_result["transfer_ms"] = parsed.get("data_transfer")
                timing_result["computation_ms"] = parsed.get("computation")
            else:
                # CPU version reports total time in seconds
                cpu_total = parsed.get("cpu_total")
                if cpu_total:
                    timing_result["total_ms"] = cpu_total * 1000  # convert to ms
                else:
                    timing_result["total_ms"] = wall_time_ms
        elif benchmark_name == "leukocyte":
            if exec_loc == "gpu":
                # GPU version reports in ms
                timing_result["total_ms"] = parsed.get("total")
                timing_result["transfer_ms"] = parsed.get("data_transfer")
                timing_result["computation_ms"] = parsed.get("computation")
            else:
                # CPU version reports total time in seconds
                cpu_total = parsed.get("cpu_total")
                if cpu_total:
                    timing_result["total_ms"] = cpu_total * 1000  # convert to ms
                else:
                    timing_result["total_ms"] = wall_time_ms

        # If total wasn't parsed, use wall time
        if timing_result["total_ms"] is None:
            timing_result["total_ms"] = wall_time_ms

        return timing_result

    except subprocess.TimeoutExpired:
        print(f"Warning: {benchmark_name} timed out")
        return None
    except FileNotFoundError:
        print(f"Error: Executable not found: {executable}")
        print(f"Please build the benchmark first using 'make' in {work_dir}")
        return None
    except Exception as e:
        print(f"Error running {benchmark_name}: {e}")
        return None


def run_benchmark(benchmark_name, data_loc, exec_loc, base_dir,
                  nr_iterations=NR_ITERATIONS, nr_warmup=NR_WARMUP_ITERATIONS,
                  sm_percentage=100, num_threads=4):
    """
    Run benchmark multiple times with warmup iterations.

    Args:
        benchmark_name: Name of the benchmark
        data_loc: Where data initially resides (cpu or gpu)
        exec_loc: Where execution happens (cpu or gpu)
        base_dir: Base directory of scientific benchmarks
        nr_iterations: Number of measurement iterations
        nr_warmup: Number of warmup iterations
        sm_percentage: GPU SM percentage (1-100) via CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
        num_threads: CPU thread count via OMP_NUM_THREADS

    Returns:
        Dictionary with lists of timing results
    """
    config_str = f"SM={sm_percentage}%" if exec_loc == "gpu" else f"threads={num_threads}"
    print(f"\n--- Running {benchmark_name} (data: {data_loc}, exec: {exec_loc}, {config_str}) ---")

    # Warmup iterations
    print(f"Running {nr_warmup} warmup iterations...")
    for i in range(nr_warmup):
        result = run_benchmark_iteration(benchmark_name, data_loc, exec_loc, base_dir,
                                         sm_percentage=sm_percentage, num_threads=num_threads)
        if result is None:
            print(f"Warmup iteration {i+1} failed")

    # Measurement iterations
    print(f"Running {nr_iterations} measurement iterations...")
    total_times = []
    transfer_times = []
    computation_times = []

    for i in range(nr_iterations):
        result = run_benchmark_iteration(benchmark_name, data_loc, exec_loc, base_dir,
                                         sm_percentage=sm_percentage, num_threads=num_threads)
        if result:
            total_times.append(result["total_ms"])
            transfer_times.append(result.get("transfer_ms"))
            computation_times.append(result.get("computation_ms"))
            print(f"  Iteration {i+1}: total={result['total_ms']:.2f}ms")
        else:
            print(f"  Iteration {i+1}: FAILED")

    if total_times:
        print(f"  Average: {sum(total_times)/len(total_times):.2f}ms")

    return {
        "total_times": total_times,
        "transfer_times": transfer_times,
        "computation_times": computation_times
    }


def run_all_modes(benchmark_name, base_dir, results, watcher=None):  # noqa: ARG001 watcher
    """
    Run benchmark in all supported modes:
    - cpu-cpu: Data on CPU, execution on CPU
    - cpu-gpu: Data on CPU, execution on GPU (includes CPU->GPU transfer)
    - gpu-cpu: Data on GPU, execution on CPU (includes GPU->CPU transfer)
    - gpu-gpu: Data on GPU, execution on GPU (not yet implemented)

    Note: gpu-cpu mode uses special hybrid executables that allocate data on GPU,
    transfer to CPU, and then execute on CPU with OpenMP.
    """
    config = BENCHMARKS[benchmark_name]

    modes = [
        ("cpu", "cpu"),   # CPU benchmark, data starts and stays on CPU
        ("cpu", "gpu"),   # GPU benchmark with data transfer from CPU
    ]

    # Add gpu-cpu mode if the benchmark supports it
    if "gpu_cpu_executable" in config:
        modes.append(("gpu", "cpu"))

    # gpu-gpu mode would require pre-loading data to GPU (not yet implemented)
    # modes.append(("gpu", "gpu"))

    for data_loc, exec_loc in modes:
        try:
            raw_results = run_benchmark(benchmark_name, data_loc, exec_loc, base_dir)
            if raw_results["total_times"]:
                results.add_raw_results(
                    benchmark_name=benchmark_name,
                    data_loc=data_loc,
                    exec_loc=exec_loc,
                    total_times=raw_results["total_times"],
                    transfer_times=raw_results["transfer_times"],
                    computation_times=raw_results["computation_times"]
                )
        except Exception as e:
            print(f"Error running {benchmark_name} ({data_loc}->{exec_loc}): {e}")


def benchmark_changing_sm_percentage(benchmark_name, base_dir, results,
                                      nr_iterations=NR_ITERATIONS, nr_warmup=NR_WARMUP_ITERATIONS):
    """
    Run GPU benchmark with varying SM percentages (10%, 20%, ..., 100%).
    Uses CUDA_MPS_ACTIVE_THREAD_PERCENTAGE environment variable.

    This mode runs cpu-gpu configuration (data on CPU, execution on GPU)
    with different GPU SM utilization levels.
    """
    print("\n" + "=" * 60)
    print(f"Running {benchmark_name} with varying SM percentage (GPU)")
    print("=" * 60)

    data_loc = "cpu"
    exec_loc = "gpu"

    for sm_pct in range(10, 101, 10):
        print(f"\n--- SM Percentage: {sm_pct}% ---")
        try:
            raw_results = run_benchmark(
                benchmark_name, data_loc, exec_loc, base_dir,
                nr_iterations=nr_iterations, nr_warmup=nr_warmup,
                sm_percentage=sm_pct
            )

            if raw_results["total_times"]:
                results.add_raw_results(
                    benchmark_name=benchmark_name,
                    data_loc=data_loc,
                    exec_loc=exec_loc,
                    total_times=raw_results["total_times"],
                    transfer_times=raw_results["transfer_times"],
                    computation_times=raw_results["computation_times"],
                    sm_percentage=sm_pct
                )
        except Exception as e:
            print(f"Error at SM {sm_pct}%: {e}")
            continue


def benchmark_changing_num_threads(benchmark_name, base_dir, results,
                                    nr_iterations=NR_ITERATIONS, nr_warmup=NR_WARMUP_ITERATIONS,
                                    thread_counts=None):
    """
    Run CPU benchmark with varying thread counts.
    Uses OMP_NUM_THREADS and MKL_NUM_THREADS environment variables.

    This mode runs cpu-cpu configuration (data on CPU, execution on CPU)
    with different CPU thread counts.

    Args:
        thread_counts: List of thread counts to test. Default: [1, 2, 4, 8, 16, 32, 64]
    """
    if thread_counts is None:
        thread_counts = [1, 2, 4, 8, 16, 32, 64]

    print("\n" + "=" * 60)
    print(f"Running {benchmark_name} with varying thread count (CPU)")
    print("=" * 60)

    data_loc = "cpu"
    exec_loc = "cpu"

    for num_threads in thread_counts:
        print(f"\n--- Thread count: {num_threads} ---")
        try:
            raw_results = run_benchmark(
                benchmark_name, data_loc, exec_loc, base_dir,
                nr_iterations=nr_iterations, nr_warmup=nr_warmup,
                num_threads=num_threads
            )

            if raw_results["total_times"]:
                results.add_raw_results(
                    benchmark_name=benchmark_name,
                    data_loc=data_loc,
                    exec_loc=exec_loc,
                    total_times=raw_results["total_times"],
                    transfer_times=raw_results["transfer_times"],
                    computation_times=raw_results["computation_times"],
                    num_threads=num_threads
                )
        except Exception as e:
            print(f"Error at {num_threads} threads: {e}")
            continue


def benchmark_gpu_cpu_changing_num_threads(benchmark_name, base_dir, results,
                                            nr_iterations=NR_ITERATIONS, nr_warmup=NR_WARMUP_ITERATIONS,
                                            thread_counts=None):
    """
    Run GPU-CPU hybrid benchmark with varying thread counts.
    Uses OMP_NUM_THREADS for the CPU computation part.

    This mode runs gpu-cpu configuration (data on GPU, transferred to CPU, execution on CPU)
    with different CPU thread counts for the computation.

    Args:
        thread_counts: List of thread counts to test. Default: [1, 2, 4, 8, 16, 32, 64]
    """
    if thread_counts is None:
        thread_counts = [1, 2, 4, 8, 16, 32, 64]

    config = BENCHMARKS.get(benchmark_name)
    if not config or "gpu_cpu_executable" not in config:
        print(f"Warning: {benchmark_name} does not support gpu-cpu mode")
        return

    print("\n" + "=" * 60)
    print(f"Running {benchmark_name} GPU-CPU mode with varying thread count")
    print("=" * 60)

    data_loc = "gpu"
    exec_loc = "cpu"

    for num_threads in thread_counts:
        print(f"\n--- Thread count: {num_threads} ---")
        try:
            raw_results = run_benchmark(
                benchmark_name, data_loc, exec_loc, base_dir,
                nr_iterations=nr_iterations, nr_warmup=nr_warmup,
                num_threads=num_threads
            )

            if raw_results["total_times"]:
                results.add_raw_results(
                    benchmark_name=benchmark_name,
                    data_loc=data_loc,
                    exec_loc=exec_loc,
                    total_times=raw_results["total_times"],
                    transfer_times=raw_results["transfer_times"],
                    computation_times=raw_results["computation_times"],
                    num_threads=num_threads
                )
        except Exception as e:
            print(f"Error at {num_threads} threads: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Run scientific benchmarks with multiple execution modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Execution Modes:
  cpu-cpu: Data initially on CPU, execution on CPU
  cpu-gpu: Data initially on CPU, execution on GPU (includes transfer)
  gpu-cpu: Data initially on GPU, execution on CPU (includes transfer)
  gpu-gpu: Data initially on GPU, execution on GPU

Sweep Modes (similar to vLLM benchmarks):
  sweep-sm:      Vary GPU SM percentage (10%%, 20%%, ..., 100%%) - for GPU execution
  sweep-threads: Vary CPU thread count (1, 2, 4, 8, 16, 32, 64) - for CPU execution
  sweep-gpu-cpu-threads: Vary CPU thread count for GPU-CPU hybrid mode

Examples:
  %(prog)s --benchmark bfs --mode cpu-gpu
  %(prog)s --benchmark all --mode all
  %(prog)s --benchmark nn --data_loc cpu --exec_loc gpu
  %(prog)s --benchmark bfs --mode sweep-sm
  %(prog)s --benchmark nn --mode sweep-threads
  %(prog)s --benchmark bfs --mode sweep-gpu-cpu-threads
        """
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        default="all",
        choices=["bfs", "nn", "leukocyte", "all"],
        help="Benchmark to run (default: all)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["cpu-cpu", "cpu-gpu", "gpu-cpu", "gpu-gpu", "all",
                 "sweep-sm", "sweep-threads", "sweep-gpu-cpu-threads"],
        help="Execution mode (default: all). Use sweep-* modes for parameter sweeps."
    )
    parser.add_argument(
        "--data_loc",
        type=str,
        choices=["cpu", "gpu"],
        help="Override: where data initially resides"
    )
    parser.add_argument(
        "--exec_loc",
        type=str,
        choices=["cpu", "gpu"],
        help="Override: where execution happens"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=NR_ITERATIONS,
        help=f"Number of measurement iterations (default: {NR_ITERATIONS})"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=NR_WARMUP_ITERATIONS,
        help=f"Number of warmup iterations (default: {NR_WARMUP_ITERATIONS})"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)"
    )
    parser.add_argument(
        "--measure_memory",
        action="store_true",
        help="Enable memory monitoring during benchmarks"
    )
    parser.add_argument(
        "--thread_counts",
        type=str,
        default="1,2,4,8,16,32,64",
        help="Comma-separated list of thread counts for sweep-threads mode (default: 1,2,4,8,16,32,64)"
    )

    args = parser.parse_args()

    # Determine base directory - this script is now in scientific/ directly
    # so base_dir is the same as script_dir
    script_dir = Path(__file__).parent.resolve()
    base_dir = script_dir  # Script is now in scientific/, which contains bfs/, bfs_cuda/, etc.

    # Determine which benchmarks to run
    if args.benchmark == "all":
        benchmarks = list(BENCHMARKS.keys())
    else:
        benchmarks = [args.benchmark]

    # Parse thread counts for sweep modes
    thread_counts = [int(x.strip()) for x in args.thread_counts.split(",")]

    # Setup memory watcher if requested
    watcher = None
    if args.measure_memory and WATCHER_AVAILABLE:
        output_dir = os.path.join(base_dir, args.output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Will be started per-benchmark

    # Run benchmarks
    results = BenchmarkResult()

    print("=" * 60)
    print("Scientific Benchmarks Runner")
    print("=" * 60)
    print(f"Benchmarks: {', '.join(benchmarks)}")
    print(f"Mode: {args.mode}")
    print(f"Iterations: {args.iterations} (warmup: {args.warmup})")
    print(f"Base directory: {base_dir}")
    if args.mode in ["sweep-threads", "sweep-gpu-cpu-threads"]:
        print(f"Thread counts: {thread_counts}")
    print("=" * 60)

    # Handle sweep modes
    if args.mode == "sweep-sm":
        # Sweep SM percentage for GPU execution
        for benchmark in benchmarks:
            config = BENCHMARKS[benchmark]
            gpu_dir = os.path.join(base_dir, config["gpu_dir"])
            gpu_exec = os.path.join(gpu_dir, config["gpu_executable"])
            if not os.path.exists(gpu_exec):
                print(f"\nSkipping {benchmark} sweep-sm mode - executable not found: {gpu_exec}")
                print(f"Build with 'make' in {gpu_dir}")
                continue

            # Setup watcher
            if args.measure_memory and WATCHER_AVAILABLE:
                output_dir = os.path.join(base_dir, args.output_dir)
                mem_file = f"{output_dir}/{benchmark}-sweep-sm-memory.csv"
                watcher = GpuWatcher(gpu_id=0, save_loc=mem_file)
                watcher.start()

            try:
                benchmark_changing_sm_percentage(
                    benchmark, base_dir, results,
                    nr_iterations=args.iterations, nr_warmup=args.warmup
                )
            except Exception as e:
                print(f"Error running {benchmark} sweep-sm: {e}")
            finally:
                if watcher:
                    time.sleep(0.5)
                    watcher.stop()
                    watcher = None

    elif args.mode == "sweep-threads":
        # Sweep thread count for CPU execution
        for benchmark in benchmarks:
            config = BENCHMARKS[benchmark]
            cpu_dir = os.path.join(base_dir, config["cpu_dir"])
            cpu_exec = os.path.join(cpu_dir, config["cpu_executable"])
            if not os.path.exists(cpu_exec):
                print(f"\nSkipping {benchmark} sweep-threads mode - executable not found: {cpu_exec}")
                print(f"Build with 'make' in {cpu_dir}")
                continue

            # Setup watcher
            if args.measure_memory and WATCHER_AVAILABLE:
                output_dir = os.path.join(base_dir, args.output_dir)
                mem_file = f"{output_dir}/{benchmark}-sweep-threads-memory.csv"
                watcher = CpuWatcher(id=0, save_loc=mem_file)
                watcher.start()

            try:
                benchmark_changing_num_threads(
                    benchmark, base_dir, results,
                    nr_iterations=args.iterations, nr_warmup=args.warmup,
                    thread_counts=thread_counts
                )
            except Exception as e:
                print(f"Error running {benchmark} sweep-threads: {e}")
            finally:
                if watcher:
                    time.sleep(0.5)
                    watcher.stop()
                    watcher = None

    elif args.mode == "sweep-gpu-cpu-threads":
        # Sweep thread count for GPU-CPU hybrid execution
        for benchmark in benchmarks:
            config = BENCHMARKS[benchmark]
            if "gpu_cpu_executable" not in config:
                print(f"\nSkipping {benchmark} sweep-gpu-cpu-threads mode - not supported")
                continue
            gpu_dir = os.path.join(base_dir, config["gpu_dir"])
            gpu_cpu_exec = os.path.join(gpu_dir, config["gpu_cpu_executable"])
            if not os.path.exists(gpu_cpu_exec):
                print(f"\nSkipping {benchmark} sweep-gpu-cpu-threads mode - executable not found: {gpu_cpu_exec}")
                print(f"Build with 'make' in {gpu_dir}")
                continue

            # Setup watcher
            if args.measure_memory and WATCHER_AVAILABLE:
                output_dir = os.path.join(base_dir, args.output_dir)
                mem_file = f"{output_dir}/{benchmark}-sweep-gpu-cpu-threads-memory.csv"
                watcher = CpuWatcher(id=0, save_loc=mem_file)
                watcher.start()

            try:
                benchmark_gpu_cpu_changing_num_threads(
                    benchmark, base_dir, results,
                    nr_iterations=args.iterations, nr_warmup=args.warmup,
                    thread_counts=thread_counts
                )
            except Exception as e:
                print(f"Error running {benchmark} sweep-gpu-cpu-threads: {e}")
            finally:
                if watcher:
                    time.sleep(0.5)
                    watcher.stop()
                    watcher = None

    else:
        # Standard modes (cpu-cpu, cpu-gpu, gpu-cpu, all)
        if args.data_loc and args.exec_loc:
            modes = [(args.data_loc, args.exec_loc)]
        elif args.mode == "all":
            modes = [
                ("cpu", "cpu"),
                ("cpu", "gpu"),
                ("gpu", "cpu"),  # GPU-CPU mode (requires hybrid executable)
                # ("gpu", "gpu"),  # Would need pre-loaded GPU data (not implemented)
            ]
        else:
            data_loc, exec_loc = args.mode.split("-")
            modes = [(data_loc, exec_loc)]

        for benchmark in benchmarks:
            for data_loc, exec_loc in modes:
                config = BENCHMARKS[benchmark]

                # Check if mode is valid for benchmark
                if data_loc == "gpu" and exec_loc == "cpu":
                    # GPU-CPU mode requires special hybrid executable
                    if "gpu_cpu_executable" not in config:
                        print(f"\nSkipping {benchmark} gpu-cpu mode - not supported")
                        continue
                    gpu_dir = os.path.join(base_dir, config["gpu_dir"])
                    gpu_cpu_exec = os.path.join(gpu_dir, config["gpu_cpu_executable"])
                    if not os.path.exists(gpu_cpu_exec):
                        print(f"\nSkipping {benchmark} gpu-cpu mode - executable not found: {gpu_cpu_exec}")
                        print(f"Build with 'make' in {gpu_dir}")
                        continue
                elif exec_loc == "gpu":
                    # Check if GPU executable exists
                    gpu_dir = os.path.join(base_dir, config["gpu_dir"])
                    gpu_exec = os.path.join(gpu_dir, config["gpu_executable"])
                    if not os.path.exists(gpu_exec):
                        print(f"\nSkipping {benchmark} cpu-gpu mode - executable not found: {gpu_exec}")
                        print(f"Build with 'make' in {gpu_dir}")
                        continue
                elif exec_loc == "cpu" and data_loc == "cpu":
                    # Check if CPU executable exists
                    cpu_dir = os.path.join(base_dir, config["cpu_dir"])
                    cpu_exec = os.path.join(cpu_dir, config["cpu_executable"])
                    if not os.path.exists(cpu_exec):
                        print(f"\nSkipping {benchmark} cpu-cpu mode - executable not found: {cpu_exec}")
                        print(f"Build with 'make' in {cpu_dir}")
                        continue

                # Setup watcher for this run
                if args.measure_memory and WATCHER_AVAILABLE:
                    output_dir = os.path.join(base_dir, args.output_dir)
                    mem_file = f"{output_dir}/{benchmark}-{data_loc}-{exec_loc}-memory.csv"
                    if exec_loc == "gpu":
                        watcher = GpuWatcher(gpu_id=0, save_loc=mem_file)
                    else:
                        watcher = CpuWatcher(id=0, save_loc=mem_file)
                    watcher.start()

                try:
                    raw_results = run_benchmark(
                        benchmark, data_loc, exec_loc, base_dir,
                        nr_iterations=args.iterations,
                        nr_warmup=args.warmup
                    )

                    if raw_results["total_times"]:
                        results.add_raw_results(
                            benchmark_name=benchmark,
                            data_loc=data_loc,
                            exec_loc=exec_loc,
                            total_times=raw_results["total_times"],
                            transfer_times=raw_results["transfer_times"],
                            computation_times=raw_results["computation_times"]
                        )
                except Exception as e:
                    print(f"Error running {benchmark} ({data_loc}->{exec_loc}): {e}")
                finally:
                    if watcher:
                        time.sleep(0.5)
                        watcher.stop()
                        watcher = None

    # Save results with mode-specific filename
    output_dir = os.path.join(base_dir, args.output_dir)
    if args.mode.startswith("sweep-"):
        results.save_to_csv(output_dir, f"scientific-benchmarks-{args.mode}")
    else:
        results.save_to_csv(output_dir, "scientific-benchmarks")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if not results.data.empty:
        print(results.data.to_string(index=False))
    else:
        print("No results collected")
    print("=" * 60)


if __name__ == "__main__":
    main()
