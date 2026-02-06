#!/usr/bin/env bash

set -e

ITERATIONS="${1:-5}"

export TRANSFORMERS_CACHE="$SCRATCH"
export TORCH_HOME="$SCRATCH"
export PYTHONWARNINGS=ignore

# Cleanup function to ensure MPS daemon is stopped
cleanup() {
  echo "Stopping MPS daemon..."
  echo quit | nvidia-cuda-mps-control 2>/dev/null || true
}

# Set trap to call cleanup on EXIT (success, failure, or interruption)
trap cleanup EXIT

# Remove previous results CSV
rm -f results/npbench_results.csv

CPU_THREADS=(1 2 4 8 16 36 72)

# CPU benchmarks
echo "=== CPU benchmarks ==="
for i in "${CPU_THREADS[@]}"; do
  echo "Running with $i threads..."
  export OMP_NUM_THREADS="$i"
  export MKL_NUM_THREADS="$i"

  python benchmarks/deep_learning/conv2d_bias/conv2d_cpu_exec.py --iterations "$ITERATIONS" --data-loc cpu
  python benchmarks/nbody/nbody_cpu_exec.py --iterations "$ITERATIONS" --data-loc cpu
done

# GPU benchmarks
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-$(id -un)
nvidia-cuda-mps-control -d

echo "=== GPU benchmarks ==="
for ((i=10; i<=100; i+=10)); do
  echo "Running with SM $i%..."
  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$i"

  python benchmarks/deep_learning/conv2d_bias/conv2d_gpu_exec.py --iterations "$ITERATIONS" --data-loc cpu
  python benchmarks/nbody/nbody_gpu_exec.py --iterations "$ITERATIONS" --data-loc cpu
done

echo quit | nvidia-cuda-mps-control

echo ""
echo "=== Results saved to results/npbench_results.csv ==="