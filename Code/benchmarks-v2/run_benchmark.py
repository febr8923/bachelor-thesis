#!/usr/bin/env python3
"""
Unified Benchmark Runner
Run all benchmarks with a single command
"""
import argparse
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import *
from core.utils import setup_environment, validate_device
from runners.vllm_benchmark import VLLMBenchmark
from runners.image_benchmark import ImageBenchmark
from plots.plotter import BenchmarkPlotter


def main():
    parser = argparse.ArgumentParser(
        description="Unified Benchmark Runner for ML Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run vLLM benchmark on GPU with all modes
  python run_benchmark.py --type vllm --model meta-llama/Llama-2-7b-hf --device gpu --mode 5

  # Run image benchmark on CPU with batch size variations
  python run_benchmark.py --type image --model resnet50 --device cpu --mode 2

  # Run with memory monitoring
  python run_benchmark.py --type vllm --model gpt2 --device gpu --mode 1 --measure-memory

  # Generate plots after benchmarking
  python run_benchmark.py --type vllm --model gpt2 --device gpu --mode 5 --plot

Modes:
  1: Resource variation (SM% for GPU, thread count for CPU)
  2: Batch size variation
  3: Input length variation (vLLM only)
  4: Memory utilization variation (GPU only, vLLM only)
  5: All applicable benchmarks
        """
    )

    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["vllm", "image"],
        help="Type of benchmark to run"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (HuggingFace model for vLLM, PyTorch Hub model for image)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["cpu", "gpu"],
        help="Device to run on (default: gpu)"
    )

    parser.add_argument(
        "--model-location",
        type=str,
        default=None,
        choices=["cpu", "gpu"],
        help="Initial model location (image benchmarks only, defaults to --device)"
    )

    parser.add_argument(
        "--mode",
        type=int,
        default=5,
        choices=[1, 2, 3, 4, 5],
        help="Benchmark mode (default: 5 - all benchmarks)"
    )

    parser.add_argument(
        "--measure-memory",
        action="store_true",
        help="Enable memory usage monitoring"
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots after benchmarking"
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=TRANSFORMERS_CACHE,
        help=f"Model cache directory (default: {TRANSFORMERS_CACHE})"
    )

    args = parser.parse_args()

    # Validate inputs
    try:
        device = validate_device(args.device)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Setup environment
    setup_environment(args.cache_dir)

    # Determine model location for image benchmarks
    model_loc = args.model_location if args.model_location else device

    print("="*60)
    print(f"Benchmark Configuration")
    print("="*60)
    print(f"Type:           {args.type}")
    print(f"Model:          {args.model}")
    print(f"Device:         {device}")
    if args.type == "image":
        print(f"Model Location: {model_loc}")
    print(f"Mode:           {args.mode} - {BENCHMARK_MODES.get(args.mode, 'Unknown')}")
    print(f"Memory Monitor: {args.measure_memory}")
    print(f"Generate Plots: {args.plot}")
    print("="*60)
    print()

    # Run benchmark
    try:
        if args.type == "vllm":
            benchmark = VLLMBenchmark(
                model_name=args.model,
                exec_loc=device,
                measure_memory=args.measure_memory
            )
        elif args.type == "image":
            benchmark = ImageBenchmark(
                model_name=args.model,
                model_loc=model_loc,
                exec_loc=device,
                measure_memory=args.measure_memory
            )
        else:
            print(f"Error: Unknown benchmark type: {args.type}")
            return 1

        # Run benchmarks
        print(f"\nStarting {args.type.upper()} benchmarks...")
        results = benchmark.run(mode=args.mode)
        print(f"\n✓ Completed {len(results)} benchmark(s)")

        # Generate plots if requested
        if args.plot:
            print("\nGenerating plots...")
            plotter = BenchmarkPlotter(output_dir=PLOTS_DIR)
            results_dir = os.path.join(RESULTS_DIR, args.model)

            if os.path.exists(results_dir):
                plotter.plot_all_results(results_dir, args.model, args.type)
                print(f"✓ Plots saved to {PLOTS_DIR}/")
            else:
                print(f"Warning: Results directory not found: {results_dir}")

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        return 130
    except Exception as e:
        print(f"\nError running benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "="*60)
    print("Benchmark completed successfully!")
    print(f"Results saved to: {RESULTS_DIR}/{args.model}/")
    if args.plot:
        print(f"Plots saved to: {PLOTS_DIR}/")
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
