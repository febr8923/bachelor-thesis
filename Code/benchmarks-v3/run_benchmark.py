#!/usr/bin/env python3
"""
Example script for running vLLM benchmarks.

Usage:
    python run_benchmark.py --model <model_path> --mode <1-5> --device <gpu|cpu>

Example:
    python run_benchmark.py --model meta-llama/Llama-2-7b-hf --mode 1 --device gpu
"""

import argparse
from pathlib import Path
from util.benchmark import VLLMBenchmark
from util.config import BENCHMARK_MODES


def main():
    parser = argparse.ArgumentParser(description="Run vLLM benchmarks")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model or HuggingFace model ID"
    )
    parser.add_argument(
        "--mode",
        type=int,
        choices=[1, 2, 3, 4, 5],
        required=True,
        help=(
            "Benchmark mode: "
            "1=resource_variation, "
            "2=batch_size, "
            "3=input_length, "
            "4=memory_util (GPU only), "
            "5=all"
        )
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["gpu", "cpu"],
        default="gpu",
        help="Execution device (default: gpu)"
    )
    parser.add_argument(
        "--cold-start",
        action="store_true",
        help="Perform cold start for each iteration"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save results (default: results)"
    )
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="Disable resource monitoring"
    )

    args = parser.parse_args()

    # Validate mode for device
    if args.mode == 4 and args.device != "gpu":
        parser.error("Mode 4 (memory_util) is only supported on GPU")

    print(f"Starting vLLM benchmark...")
    print(f"  Model: {args.model}")
    print(f"  Mode: {args.mode} ({BENCHMARK_MODES[args.mode]})")
    print(f"  Device: {args.device}")
    print(f"  Cold start: {args.cold_start}")
    print(f"  Results dir: {args.results_dir}")
    print()

    # Create benchmark instance
    benchmark = VLLMBenchmark(
        model_name=args.model,
        execution_location=args.device,
        results_dir=args.results_dir,
    )

    # Run benchmark
    try:
        result = benchmark.run_benchmark_mode(
            mode=args.mode,
            is_cold_start=args.cold_start,
            save_results=True,
            monitor_resources=not args.no_monitor,
        )

        print("Benchmark completed successfully!")

        if args.mode == 5:
            print(f"Results saved for modes: {', '.join(result.keys())}")
        else:
            print(f"Results saved to: {args.results_dir}/{BENCHMARK_MODES[args.mode]}_results.csv")

        if not args.no_monitor:
            print(f"Memory monitoring saved to: {args.results_dir}/{BENCHMARK_MODES[args.mode]}_memory.csv")

    except Exception as e:
        print(f"Error running benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
