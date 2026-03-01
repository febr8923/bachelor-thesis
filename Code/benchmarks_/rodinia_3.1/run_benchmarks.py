#!/usr/bin/env python3
"""
Rodinia Benchmark Runner

Runs BFS, NN, and Leukocyte benchmarks in four execution modes:
  - cpu:     OpenMP (data on CPU, execution on CPU)
  - gpu-cpu: Data moved to GPU then back, computation on CPU (OpenMP)
  - gpu-gpu: CUDA with timer starting after alloc + H2D transfer
  - gpu:     Full CUDA (timer includes alloc + H2D + kernel + D2H)

Parses timing output and appends results to a CSV file.

Usage:
  python run_benchmarks.py --benchmark bfs --model_location cpu --execution_location cpu \
      --thread_percentage 4 [--cold_start] [--runs 5] [--warmup 3]
  python run_benchmarks.py --benchmark all --model_location gpu --execution_location gpu \
      --thread_percentage 50 [--cold_start] [--runs 5] [--warmup 3]
"""

import argparse
import subprocess
import re
import csv
import os
import sys
import statistics

# ============================================================================
# Configuration: paths relative to RODINIA_ROOT
# ============================================================================
RODINIA_ROOT = os.path.dirname(os.path.abspath(__file__))

# Input data paths (relative to RODINIA_ROOT)
BFS_INPUT = os.path.join(RODINIA_ROOT, "data", "bfs", "graph1MW_6.txt")
NN_FILELIST = os.path.join(RODINIA_ROOT, "data", "nn", "filelist.txt")
NN_DATA_DIR = os.path.join(RODINIA_ROOT, "data", "nn")
LEUKOCYTE_INPUT = os.path.join(RODINIA_ROOT, "data", "leukocyte", "testfile.avi")
LEUKOCYTE_FRAMES = "5"

# Default CSV output
DEFAULT_CSV = os.path.join(RODINIA_ROOT, "benchmark_results.csv")

# Executable paths per (model_location, execution_location) → mode
# mode: (executable_relative_path, lambda threads: [args])
BENCHMARKS = {
    "bfs": {
        ("cpu", "cpu"): {
            "exe": "openmp/bfs/bfs",
            "args": lambda t: [str(t), BFS_INPUT],
            "mode": "cpu",
        },
        ("gpu", "cpu"): {
            "exe": "gpu-cpu/bfs/bfs_gpu_cpu",
            "args": lambda t: [str(t), BFS_INPUT],
            "mode": "gpu-cpu",
        },
        ("gpu", "gpu"): {
            "exe": "gpu-gpu/bfs/bfs_gpu_gpu",
            "args": lambda _: [BFS_INPUT],
            "mode": "gpu-gpu",
        },
        ("cpu", "gpu"): {
            "exe": "cuda/bfs/bfs",
            "args": lambda _: [BFS_INPUT],
            "mode": "gpu",
        },
    },
    "nn": {
        ("cpu", "cpu"): {
            "exe": "openmp/nn/nn",
            "args": lambda _: [NN_FILELIST, "5", "30", "90"],
            "mode": "cpu",
            "cwd": NN_DATA_DIR,
        },
        ("gpu", "cpu"): {
            "exe": "gpu-cpu/nn/nn_gpu_cpu",
            "args": lambda _: [NN_FILELIST, "5", "30", "90"],
            "mode": "gpu-cpu",
            "cwd": NN_DATA_DIR,
        },
        ("gpu", "gpu"): {
            "exe": "gpu-gpu/nn/nn_gpu_gpu",
            "args": lambda _: [NN_FILELIST, "-r", "5", "-lat", "30", "-lng", "90"],
            "mode": "gpu-gpu",
            "cwd": NN_DATA_DIR,
        },
        ("cpu", "gpu"): {
            "exe": "cuda/nn/nn",
            "args": lambda _: [NN_FILELIST, "-r", "5", "-lat", "30", "-lng", "90"],
            "mode": "gpu",
            "cwd": NN_DATA_DIR,
        },
    },
    "leukocyte": {
        ("cpu", "cpu"): {
            "exe": "openmp/leukocyte/OpenMP/leukocyte",
            "args": lambda t: [LEUKOCYTE_FRAMES, str(t), LEUKOCYTE_INPUT],
            "mode": "cpu",
        },
        ("gpu", "cpu"): {
            "exe": "gpu-cpu/leukocyte/leukocyte_gpu_cpu",
            "args": lambda t: [LEUKOCYTE_FRAMES, str(t), LEUKOCYTE_INPUT],
            "mode": "gpu-cpu",
        },
        ("gpu", "gpu"): {
            "exe": "gpu-gpu/leukocyte/leukocyte_gpu_gpu",
            "args": lambda _: [LEUKOCYTE_INPUT, LEUKOCYTE_FRAMES],
            "mode": "gpu-gpu",
        },
        ("cpu", "gpu"): {
            "exe": "cuda/leukocyte/CUDA/leukocyte",
            "args": lambda _: [LEUKOCYTE_INPUT, LEUKOCYTE_FRAMES],
            "mode": "gpu",
        },
    },
}

# ============================================================================
# Timing parsers
# ============================================================================

# Regex patterns for extracting timing values from benchmark output
TIMING_PATTERNS = {
    "total_time": [
        re.compile(r"Total time[^:]*:\s*([\d.]+)\s*(?:seconds|s)?", re.IGNORECASE),
        re.compile(r"Total application run time:\s*([\d.]+)\s*seconds", re.IGNORECASE),
        re.compile(r"total time\s*:\s*([\d.]+)\s*s", re.IGNORECASE),
    ],
    "compute_time": [
        re.compile(r"Compute time[^:]*:\s*([\d.]+)", re.IGNORECASE),
        re.compile(r"Computation time[^:]*:\s*([\d.]+)\s*seconds", re.IGNORECASE),
        re.compile(r"GICOV computation:\s*([\d.]+)\s*seconds", re.IGNORECASE),
    ],
    "transfer_time": [
        re.compile(r"Data transfer time[^:]*:\s*([\d.]+)\s*seconds", re.IGNORECASE),
    ],
}


def parse_timing(output):
    """Parse timing values from benchmark stdout."""
    result = {
        "total_time": None,
        "compute_time": None,
        "transfer_time": None,
    }

    for key, patterns in TIMING_PATTERNS.items():
        for pattern in patterns:
            match = pattern.search(output)
            if match:
                result[key] = float(match.group(1))
                break

    return result


# ============================================================================
# Runner
# ============================================================================


def run_once(exe_path, args, env=None, cwd=None):
    """Run a benchmark once and return parsed timing dict + raw output."""
    cmd = [exe_path] + args
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    if cwd is None:
        cwd = os.path.dirname(exe_path) or "."

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=600,
            env=merged_env,
            cwd=cwd,
        )
        output = proc.stdout.decode("utf-8", errors="replace")
        timing = parse_timing(output)
        return timing, output
    except subprocess.TimeoutExpired:
        print(f"  WARNING: Timeout running {' '.join(cmd)}", file=sys.stderr)
        return None, ""
    except FileNotFoundError:
        print(f"  ERROR: Executable not found: {exe_path}", file=sys.stderr)
        print(f"  Make sure you have compiled the benchmark first.", file=sys.stderr)
        return None, ""


def run_benchmark(benchmark, model_location, execution_location, thread_percentage,
                  cold_start, runs, warmup, verbose):
    """Run a single benchmark configuration and return averaged results."""
    key = (model_location, execution_location)
    if benchmark not in BENCHMARKS:
        print(f"ERROR: Unknown benchmark '{benchmark}'", file=sys.stderr)
        return None
    if key not in BENCHMARKS[benchmark]:
        print(f"ERROR: Unknown location combo ({model_location}, {execution_location}) "
              f"for benchmark '{benchmark}'", file=sys.stderr)
        return None

    config = BENCHMARKS[benchmark][key]
    exe_path = os.path.join(RODINIA_ROOT, config["exe"])
    args = config["args"](thread_percentage)
    mode = config["mode"]
    run_cwd = config.get("cwd", None)

    if not os.path.isfile(exe_path):
        print(f"ERROR: Executable not found: {exe_path}", file=sys.stderr)
        print(f"  Compile it first, then re-run.", file=sys.stderr)
        return None

    env = {}
    if execution_location == "cpu":
        env["OMP_NUM_THREADS"] = str(thread_percentage)
        env["MKL_NUM_THREADS"] = str(thread_percentage)

    # Warmup runs (skipped for cold start)
    actual_warmup = 0 if cold_start else warmup
    for i in range(actual_warmup):
        if verbose:
            print(f"  Warmup {i+1}/{actual_warmup}...")
        run_once(exe_path, args, env, cwd=run_cwd)

    # Measurement runs
    timings = []
    for i in range(runs):
        if verbose:
            print(f"  Run {i+1}/{runs}...")
        timing, output = run_once(exe_path, args, env, cwd=run_cwd)
        if timing is None:
            continue
        if verbose and timing["total_time"] is not None:
            print(f"    total={timing['total_time']:.6f}s", end="")
            if timing["compute_time"] is not None:
                print(f"  compute={timing['compute_time']:.6f}s", end="")
            if timing["transfer_time"] is not None:
                print(f"  transfer={timing['transfer_time']:.6f}s", end="")
            print()
        timings.append(timing)

    if not timings:
        print(f"ERROR: No successful runs for {benchmark} ({mode})", file=sys.stderr)
        return None

    # Compute averages
    def avg(key):
        vals = [t[key] for t in timings if t[key] is not None]
        return statistics.mean(vals) if vals else None

    result = {
        "benchmark": benchmark,
        "mode": mode,
        "model_location": model_location,
        "execution_location": execution_location,
        "is_coldstart": cold_start,
        "avg_total_time": avg("total_time"),
        "avg_computation_time": avg("compute_time"),
        "avg_transfer_time": avg("transfer_time"),
        "thread_percentage": thread_percentage,
        "num_runs": len(timings),
    }

    return result


def write_csv(result, csv_path):
    """Append a result row to the CSV file."""
    fieldnames = [
        "benchmark", "mode", "model_location", "execution_location",
        "is_coldstart", "avg_total_time", "avg_computation_time",
        "avg_transfer_time", "thread_percentage", "num_runs",
    ]

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists or os.path.getsize(csv_path) == 0:
            writer.writeheader()
        writer.writerow(result)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Rodinia Benchmark Runner with timing measurement"
    )
    parser.add_argument(
        "--benchmark", required=True,
        choices=["bfs", "nn", "leukocyte", "all"],
        help="Which benchmark to run (or 'all')"
    )
    parser.add_argument(
        "--model_location", required=True,
        choices=["cpu", "gpu"],
        help="Where the data/model resides"
    )
    parser.add_argument(
        "--execution_location", required=True,
        choices=["cpu", "gpu"],
        help="Where computation is executed"
    )
    parser.add_argument(
        "--thread_percentage", required=True, type=int,
        help="Number of CPU threads (cpu mode) or SM percentage (gpu mode)"
    )
    parser.add_argument(
        "--cold_start", action="store_true",
        help="Cold start mode (skip warmup iterations)"
    )
    parser.add_argument(
        "--runs", type=int, default=5,
        help="Number of measurement runs (default: 5)"
    )
    parser.add_argument(
        "--warmup", type=int, default=3,
        help="Number of warmup iterations before measurement (default: 3)"
    )
    parser.add_argument(
        "--csv", type=str, default=DEFAULT_CSV,
        help=f"Output CSV file (default: {DEFAULT_CSV})"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-run timing details"
    )

    args = parser.parse_args()

    benchmarks = ["bfs", "nn", "leukocyte"] if args.benchmark == "all" else [args.benchmark]

    for bench in benchmarks:
        cold_str = "cold" if args.cold_start else "warm"
        print(f"[{bench}] mode=({args.model_location},{args.execution_location}) "
              f"threads={args.thread_percentage} {cold_str} "
              f"runs={args.runs} warmup={args.warmup}")

        result = run_benchmark(
            benchmark=bench,
            model_location=args.model_location,
            execution_location=args.execution_location,
            thread_percentage=args.thread_percentage,
            cold_start=args.cold_start,
            runs=args.runs,
            warmup=args.warmup,
            verbose=args.verbose,
        )

        if result:
            write_csv(result, args.csv)
            total = f"{result['avg_total_time']:.6f}s" if result['avg_total_time'] else "N/A"
            comp = f"{result['avg_computation_time']:.6f}s" if result['avg_computation_time'] else "N/A"
            xfer = f"{result['avg_transfer_time']:.6f}s" if result['avg_transfer_time'] else "N/A"
            print(f"  => avg_total={total}  avg_compute={comp}  avg_transfer={xfer}")
            print(f"  => Written to {args.csv}")
        else:
            print(f"  => SKIPPED (errors)")

    print("Done.")


if __name__ == "__main__":
    main()
