#!/usr/bin/env python3
"""
Rodinia Benchmark Runner

Runs BFS, NN, and Hotspot3D benchmarks in four execution modes:
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
HOTSPOT3D_POWER = os.path.join(RODINIA_ROOT, "data", "hotspot3D", "power_512x8")
HOTSPOT3D_TEMP = os.path.join(RODINIA_ROOT, "data", "hotspot3D", "temp_512x8")

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
    "hotspot3D": {
        ("cpu", "cpu"): {
            "exe": "openmp/hotspot3D/3D",
            "args": lambda _: ["512", "8", "100", HOTSPOT3D_POWER, HOTSPOT3D_TEMP, "output.out"],
            "mode": "cpu",
        },
        ("gpu", "cpu"): {
            "exe": "gpu-cpu/hotspot3D/3D_gpu_cpu",
            "args": lambda _: ["512", "8", "100", HOTSPOT3D_POWER, HOTSPOT3D_TEMP, "output.out"],
            "mode": "gpu-cpu",
        },
        ("gpu", "gpu"): {
            "exe": "gpu-gpu/hotspot3D/3D_gpu_gpu",
            "args": lambda _: ["512", "8", "100", HOTSPOT3D_POWER, HOTSPOT3D_TEMP, "output.out"],
            "mode": "gpu-gpu",
        },
        ("cpu", "gpu"): {
            "exe": "cuda/hotspot3D/3D",
            "args": lambda _: ["512", "8", "100", HOTSPOT3D_POWER, HOTSPOT3D_TEMP, "output.out"],
            "mode": "gpu",
        },
    },
}

# ============================================================================
# Timing parsers
# ============================================================================

# Regex patterns for extracting timing values from benchmark output
TIMING_PATTERNS = {
    "execution_time": [
        re.compile(r"Execution time[^:]*:\s*([\d.]+)\s*(?:seconds|s)?", re.IGNORECASE),
    ],
    "transfer_time_h2d": [
        re.compile(r"Data transfer time \(H2D\):\s*([\d.]+)\s*(?:seconds|s)?", re.IGNORECASE),
    ],
    "transfer_time_d2h": [
        re.compile(r"Data transfer time \(D2H\):\s*([\d.]+)\s*(?:seconds|s)?", re.IGNORECASE),
    ],
    "total_time": [
        re.compile(r"Total time[^:]*:\s*([\d.]+)\s*(?:seconds|s)?", re.IGNORECASE),
        re.compile(r"Total application run time:\s*([\d.]+)\s*seconds", re.IGNORECASE),
    ],
}


def parse_timing(output):
    """Parse timing values from benchmark stdout."""
    result = {
        "execution_time": None,
        "transfer_time_h2d": None,
        "transfer_time_d2h": None,
        "total_time": None,
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
        if verbose and timing["execution_time"] is not None:
            print(f"    execution={timing['execution_time']:.6f}s", end="")
            if timing["transfer_time_h2d"] is not None:
                print(f"  h2d={timing['transfer_time_h2d']:.6f}s", end="")
            if timing["transfer_time_d2h"] is not None:
                print(f"  d2h={timing['transfer_time_d2h']:.6f}s", end="")
            if timing["total_time"] is not None:
                print(f"  total={timing['total_time']:.6f}s", end="")
            print()
        timings.append(timing)

    if not timings:
        print(f"ERROR: No successful runs for {benchmark} ({mode})", file=sys.stderr)
        return None

    # Compute statistics
    def compute_stats(key):
        vals = [t[key] for t in timings if t[key] is not None]
        if not vals:
            return None, None, None, None
        avg = statistics.mean(vals)
        mn = min(vals)
        mx = max(vals)
        sd = statistics.stdev(vals) if len(vals) >= 2 else 0.0
        return avg, mn, mx, sd

    exec_avg, exec_min, exec_max, exec_std = compute_stats("execution_time")
    h2d_avg, h2d_min, h2d_max, h2d_std = compute_stats("transfer_time_h2d")
    d2h_avg, d2h_min, d2h_max, d2h_std = compute_stats("transfer_time_d2h")
    total_avg, total_min, total_max, total_std = compute_stats("total_time")

    result = {
        "benchmark": benchmark,
        "mode": mode,
        "model_location": model_location,
        "execution_location": execution_location,
        "is_coldstart": cold_start,
        "avg_execution_time": exec_avg,
        "min_execution_time": exec_min,
        "max_execution_time": exec_max,
        "std_execution_time": exec_std,
        "avg_transfer_time_h2d": h2d_avg,
        "min_transfer_time_h2d": h2d_min,
        "max_transfer_time_h2d": h2d_max,
        "std_transfer_time_h2d": h2d_std,
        "avg_transfer_time_d2h": d2h_avg,
        "min_transfer_time_d2h": d2h_min,
        "max_transfer_time_d2h": d2h_max,
        "std_transfer_time_d2h": d2h_std,
        "avg_total_time": total_avg,
        "min_total_time": total_min,
        "max_total_time": total_max,
        "std_total_time": total_std,
        "thread_percentage": thread_percentage,
        "num_runs": len(timings),
    }

    return result


def write_csv(result, csv_path):
    """Append a result row to the CSV file."""
    fieldnames = [
        "benchmark", "mode", "model_location", "execution_location",
        "is_coldstart",
        "avg_execution_time", "min_execution_time", "max_execution_time", "std_execution_time",
        "avg_transfer_time_h2d", "min_transfer_time_h2d", "max_transfer_time_h2d", "std_transfer_time_h2d",
        "avg_transfer_time_d2h", "min_transfer_time_d2h", "max_transfer_time_d2h", "std_transfer_time_d2h",
        "avg_total_time", "min_total_time", "max_total_time", "std_total_time",
        "thread_percentage", "num_runs",
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
        choices=["bfs", "nn", "hotspot3D", "all"],
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

    benchmarks = ["bfs", "nn", "hotspot3D"] if args.benchmark == "all" else [args.benchmark]

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
            def fmt_stat(avg_key, std_key, min_key, max_key):
                a = result[avg_key]
                if a is None:
                    return "N/A"
                s = result[std_key] or 0.0
                lo = result[min_key] or a
                hi = result[max_key] or a
                return f"{a:.6f}s (±{s:.6f}, min={lo:.6f}, max={hi:.6f})"
            exe = fmt_stat("avg_execution_time", "std_execution_time", "min_execution_time", "max_execution_time")
            h2d = fmt_stat("avg_transfer_time_h2d", "std_transfer_time_h2d", "min_transfer_time_h2d", "max_transfer_time_h2d")
            d2h = fmt_stat("avg_transfer_time_d2h", "std_transfer_time_d2h", "min_transfer_time_d2h", "max_transfer_time_d2h")
            total = fmt_stat("avg_total_time", "std_total_time", "min_total_time", "max_total_time")
            print(f"  => execution={exe}")
            print(f"  => h2d={h2d}  d2h={d2h}")
            print(f"  => total={total}")
            print(f"  => Written to {args.csv}")
        else:
            print(f"  => SKIPPED (errors)")

    print("Done.")


if __name__ == "__main__":
    main()
