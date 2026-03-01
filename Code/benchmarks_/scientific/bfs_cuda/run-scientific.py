#!/usr/bin/env python3
"""
Scientific Benchmarks Runner

This script runs scientific benchmarks (BFS, NN, Leukocyte) with four execution modes:
- cpu-cpu: Data initially on CPU, execution on CPU
- cpu-gpu: Data initially on CPU, execution on GPU (includes transfer time)
- gpu-cpu: Data initially on GPU, execution on CPU (includes transfer time)
- gpu-gpu: Data initially on GPU, execution on GPU

The output format is compatible with the vLLM benchmark results.
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
try:
    from watcher import GpuWatcher, CpuWatcher
    WATCHER_AVAILABLE = True
except ImportError:
    WATCHER_AVAILABLE = False
    print("Warning: watcher module not available, memory monitoring disabled")

NR_ITERATIONS = 5
NR_WARMUP_ITERATIONS = 2

# Benchmark configurations
BENCHMARKS = {
    "bfs": {
        "cpu_dir": "bfs",
        "gpu_dir": "bfs_cuda",
        "cpu_executable": "bfs",
        "gpu_executable": "bfs",
        "cpu_args": ["4", "../../data/bfs/graph1MW_6.txt"],  # num_threads, input_file
        "gpu_args": ["../../data/bfs/graph1MW_6.txt"],  # input_file
        "timing_patterns": {
            "total": r"Total time:\s+([\d.]+)\s*ms",
            "data_transfer": r"CPU->GPU data transfer:\s+([\d.]+)\s*ms",
            "computation": r"Pure BFS computation:\s+([\d.]+)\s*ms",
            "cpu_compute": r"Compute time:\s+([\d.]+)"  # seconds for CPU version
        }
    },
    "nn": {
        "cpu_dir": "nn",
        "gpu_dir": "nn_cuda",
        "cpu_executable": "nn",
        "gpu_executable": "nn",
        "cpu_args": ["filelist_4", "5", "30", "90"],  # filelist, k, lat, lng
        "gpu_args": ["filelist_4", "-r", "5", "-lat", "30", "-lng", "90"],
        "timing_patterns": {
            "total": r"Total time:\s+([\d.]+)\s*ms",
            "data_transfer": r"CPU->GPU data transfer:\s+([\d.]+)\s*ms",
            "computation": r"Pure NN computation:\s+([\d.]+)\s*ms",
            "cpu_total": r"total time\s*:\s*([\d.]+)\s*s",  # CPU version uses seconds
        }
    },
    "leukocyte": {
        "cpu_dir": "leukocyte",
        "gpu_dir": "leukocyte_cuda",
        "cpu_executable": "OpenMP/leukocyte",
        "gpu_executable": "CUDA/leukocyte",
        "cpu_args": ["5", "4", "../../data/leukocyte/testfile.avi"],  # num_frames, num_threads, input_file
        "gpu_args": ["../../data/leukocyte/testfile.avi", "5"],  # input_file, num_frames
        "timing_patterns": {
            "total": r"Total time:\s+([\d.]+)\s*ms",
            "data_transfer": r"CPU->GPU data transfer:\s+([\d.]+)\s*ms",
            "computation": r"Pure computation:\s+([\d.]+)\s*ms",
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
                        total_times, transfer_times, computation_times):
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


def run_benchmark_iteration(benchmark_name, data_loc, exec_loc, base_dir):  # noqa: ARG001 data_loc
    """
    Run a single benchmark iteration.

    Args:
        benchmark_name: Name of the benchmark (bfs, nn, leukocyte)
        data_loc: Where data initially resides (cpu or gpu) - used for result tracking
        exec_loc: Where execution happens (cpu or gpu)
        base_dir: Base directory of the scientific benchmarks

    Returns:
        Dictionary with timing results
    """
    config = BENCHMARKS[benchmark_name]

    # Determine which executable and arguments to use
    if exec_loc == "gpu":
        work_dir = os.path.join(base_dir, config["gpu_dir"])
        executable = os.path.join(work_dir, config["gpu_executable"])
        args = config["gpu_args"]
    else:
        work_dir = os.path.join(base_dir, config["cpu_dir"])
        executable = os.path.join(work_dir, config["cpu_executable"])
        args = config["cpu_args"]

    # Build command
    cmd = [executable] + args

    # For modes involving data transfer between CPU and GPU,
    # we need special handling. The CUDA benchmarks already measure
    # transfer time internally when data starts on CPU.
    # For gpu-cpu mode (data on GPU, exec on CPU), we'd need to modify
    # the C code to support this, which is complex. For now, we note
    # that this mode measures the full execution including any necessary transfers.

    env = os.environ.copy()

    # Set thread count for CPU execution
    if exec_loc == "cpu" and benchmark_name == "bfs":
        # BFS uses command line arg for thread count
        pass
    elif exec_loc == "cpu":
        env['OMP_NUM_THREADS'] = '4'

    try:
        start_time = time.perf_counter()
        result = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            env=env,
            timeout=300  # 5 minute timeout
        )
        end_time = time.perf_counter()

        wall_time_ms = (end_time - start_time) * 1000

        if result.returncode != 0:
            print(f"Warning: {benchmark_name} returned non-zero exit code")
            print(f"stderr: {result.stderr}")
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

        # Extract timing based on benchmark type
        if benchmark_name == "bfs":
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
                  nr_iterations=NR_ITERATIONS, nr_warmup=NR_WARMUP_ITERATIONS):
    """
    Run benchmark multiple times with warmup iterations.

    Args:
        benchmark_name: Name of the benchmark
        data_loc: Where data initially resides (cpu or gpu)
        exec_loc: Where execution happens (cpu or gpu)
        base_dir: Base directory of scientific benchmarks
        nr_iterations: Number of measurement iterations
        nr_warmup: Number of warmup iterations

    Returns:
        Dictionary with lists of timing results
    """
    print(f"\n--- Running {benchmark_name} (data: {data_loc}, exec: {exec_loc}) ---")

    # Warmup iterations
    print(f"Running {nr_warmup} warmup iterations...")
    for i in range(nr_warmup):
        result = run_benchmark_iteration(benchmark_name, data_loc, exec_loc, base_dir)
        if result is None:
            print(f"Warmup iteration {i+1} failed")

    # Measurement iterations
    print(f"Running {nr_iterations} measurement iterations...")
    total_times = []
    transfer_times = []
    computation_times = []

    for i in range(nr_iterations):
        result = run_benchmark_iteration(benchmark_name, data_loc, exec_loc, base_dir)
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
    Run benchmark in all four modes:
    - cpu-cpu: Data on CPU, execution on CPU
    - cpu-gpu: Data on CPU, execution on GPU
    - gpu-cpu: Data on GPU, execution on CPU (note: requires data already on GPU)
    - gpu-gpu: Data on GPU, execution on GPU

    Note: The gpu-cpu and gpu-gpu modes assume data is already on the GPU.
    For scientific benchmarks, the CUDA versions handle CPU->GPU transfer internally.
    """
    modes = [
        ("cpu", "cpu"),   # CPU benchmark, data starts and stays on CPU
        ("cpu", "gpu"),   # GPU benchmark with data transfer from CPU
        # ("gpu", "cpu"),   # Would require pre-loading data to GPU, then transferring back
        # ("gpu", "gpu"),   # Would require pre-loading data to GPU
    ]

    # For GPU benchmarks, the data transfer is measured internally
    # The "gpu-gpu" mode would be the GPU benchmark with data pre-loaded,
    # but current implementations always start with data on CPU

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

Examples:
  %(prog)s --benchmark bfs --mode cpu-gpu
  %(prog)s --benchmark all --mode all
  %(prog)s --benchmark nn --data_loc cpu --exec_loc gpu
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
        choices=["cpu-cpu", "cpu-gpu", "gpu-cpu", "gpu-gpu", "all"],
        help="Execution mode (default: all)"
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

    args = parser.parse_args()

    # Determine base directory (parent of this script's directory)
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent

    # Determine which benchmarks to run
    if args.benchmark == "all":
        benchmarks = list(BENCHMARKS.keys())
    else:
        benchmarks = [args.benchmark]

    # Determine which modes to run
    if args.data_loc and args.exec_loc:
        modes = [(args.data_loc, args.exec_loc)]
    elif args.mode == "all":
        modes = [
            ("cpu", "cpu"),
            ("cpu", "gpu"),
            # ("gpu", "cpu"),  # Not easily supported by current benchmark implementations
            # ("gpu", "gpu"),  # Would need pre-loaded GPU data
        ]
    else:
        data_loc, exec_loc = args.mode.split("-")
        modes = [(data_loc, exec_loc)]

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
    print(f"Modes: {', '.join([f'{d}-{e}' for d, e in modes])}")
    print(f"Iterations: {args.iterations} (warmup: {args.warmup})")
    print("=" * 60)

    for benchmark in benchmarks:
        for data_loc, exec_loc in modes:
            # Check if mode is valid for benchmark
            if exec_loc == "gpu":
                # Check if GPU executable exists
                config = BENCHMARKS[benchmark]
                gpu_dir = os.path.join(base_dir, config["gpu_dir"])
                gpu_exec = os.path.join(gpu_dir, config["gpu_executable"])
                if not os.path.exists(gpu_exec):
                    print(f"\nSkipping {benchmark} gpu mode - executable not found: {gpu_exec}")
                    print(f"Build with 'make' in {gpu_dir}")
                    continue

            if exec_loc == "cpu":
                # Check if CPU executable exists
                config = BENCHMARKS[benchmark]
                cpu_dir = os.path.join(base_dir, config["cpu_dir"])
                cpu_exec = os.path.join(cpu_dir, config["cpu_executable"])
                if not os.path.exists(cpu_exec):
                    print(f"\nSkipping {benchmark} cpu mode - executable not found: {cpu_exec}")
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

    # Save results
    output_dir = os.path.join(base_dir, args.output_dir)
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
