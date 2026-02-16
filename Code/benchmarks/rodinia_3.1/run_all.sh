#!/bin/bash
# ============================================================================
# Rodinia Benchmark Runner Script
#
# Runs all benchmarks (bfs, nn, leukocyte) across all execution modes:
#   - cpu/cpu:  OpenMP (data on CPU, execution on CPU)
#   - gpu/cpu:  GPU-CPU (data on GPU, execution on CPU)
#   - cpu/gpu:  Full CUDA (data on CPU, moved to GPU, execution on GPU)
#   - gpu/gpu:  GPU-GPU (data already on GPU, execution on GPU)
#
# CPU tests iterate over thread counts.
# GPU tests iterate over SM percentages via CUDA MPS.
#
# Usage: ./run_all.sh [benchmark]
#   benchmark: bfs, nn, leukocyte, or all (default: all)
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK="${1:-all}"

# Number of measurement runs per configuration
RUNS=5
# Number of warmup iterations (skipped for cold start)
WARMUP=3

# CPU thread counts to test
CPU_THREADS=(1 2 4 8 16 32 64 72)

# GPU SM percentages to test (via CUDA MPS)
GPU_SM_PERCENTAGES=$(seq 10 10 100)

# Output CSV file
CSV="${SCRIPT_DIR}/benchmark_results.csv"

# Remove old results if desired (comment out to append)
# rm -f "$CSV"

echo "============================================"
echo "Rodinia Benchmark Runner"
echo "============================================"
echo "Benchmark:  ${BENCHMARK}"
echo "Runs:       ${RUNS}"
echo "Warmup:     ${WARMUP}"
echo "Output CSV: ${CSV}"
echo "============================================"
echo ""

# ============================================================================
# Build all benchmarks
# ============================================================================

BENCHMARKS_TO_BUILD=()
if [ "$BENCHMARK" = "all" ]; then
    BENCHMARKS_TO_BUILD=(bfs nn leukocyte)
else
    BENCHMARKS_TO_BUILD=("$BENCHMARK")
fi

echo "=============================="
echo "Building benchmarks (clean + make)"
echo "=============================="

BUILD_FAILED=0
for bench in "${BENCHMARKS_TO_BUILD[@]}"; do
    for impl in openmp cuda gpu-cpu gpu-gpu; do
        dir="${SCRIPT_DIR}/${impl}/${bench}"
        if [ -f "${dir}/Makefile" ]; then
            echo "  [${impl}/${bench}] make clean && make ..."
            make -C "$dir" clean || true
            if ! make -C "$dir"; then
                echo "  ERROR: Build failed for ${impl}/${bench}"
                BUILD_FAILED=1
            fi
        else
            echo "  [${impl}/${bench}] skipped (no Makefile)"
        fi
    done
done

echo ""
if [ "$BUILD_FAILED" -ne 0 ]; then
    echo "WARNING: Some builds failed (see above). Continuing with available benchmarks."
fi
echo "Build complete."
echo ""

# ============================================================================
# CPU Tests (OpenMP): model_location={cpu,gpu}, execution_location=cpu
# ============================================================================

echo "=============================="
echo "CPU Tests"
echo "=============================="

for i in "${CPU_THREADS[@]}"; do
    echo ""
    echo "--- CPU threads: $i ---"
    export OMP_NUM_THREADS="$i"
    export MKL_NUM_THREADS="$i"

    # Warm start: cpu/cpu (pure OpenMP)
    python3 "${SCRIPT_DIR}/run_benchmarks.py" \
        --benchmark="$BENCHMARK" \
        --model_location="cpu" --execution_location="cpu" \
        --thread_percentage="$i" \
        --runs="$RUNS" --warmup="$WARMUP" \
        --csv="$CSV"

    # Warm start: gpu/cpu (GPU-CPU hybrid)
    python3 "${SCRIPT_DIR}/run_benchmarks.py" \
        --benchmark="$BENCHMARK" \
        --model_location="gpu" --execution_location="cpu" \
        --thread_percentage="$i" \
        --runs="$RUNS" --warmup="$WARMUP" \
        --csv="$CSV"

    # Cold start: cpu/cpu
    python3 "${SCRIPT_DIR}/run_benchmarks.py" \
        --benchmark="$BENCHMARK" \
        --model_location="cpu" --execution_location="cpu" \
        --thread_percentage="$i" \
        --cold_start \
        --runs="$RUNS" \
        --csv="$CSV"

    # Cold start: gpu/cpu
    python3 "${SCRIPT_DIR}/run_benchmarks.py" \
        --benchmark="$BENCHMARK" \
        --model_location="gpu" --execution_location="cpu" \
        --thread_percentage="$i" \
        --cold_start \
        --runs="$RUNS" \
        --csv="$CSV"
done

# ============================================================================
# GPU Tests (CUDA): model_location={gpu,cpu}, execution_location=gpu
# Uses CUDA MPS to control SM percentage
# ============================================================================

echo ""
echo "=============================="
echo "GPU Tests"
echo "=============================="

# Start CUDA MPS daemon
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-$(id -un)

echo "Starting CUDA MPS control daemon..."
nvidia-cuda-mps-control -d 2>/dev/null || echo "  (MPS daemon may already be running or not available)"

for i in $GPU_SM_PERCENTAGES; do
    echo ""
    echo "--- GPU SM percentage: $i% ---"
    export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$i"

    # Warm start: gpu/gpu (timer after alloc + H2D)
    python3 "${SCRIPT_DIR}/run_benchmarks.py" \
        --benchmark="$BENCHMARK" \
        --model_location="gpu" --execution_location="gpu" \
        --thread_percentage="$i" \
        --runs="$RUNS" --warmup="$WARMUP" \
        --csv="$CSV"

    # Warm start: cpu/gpu (full CUDA including alloc + H2D)
    python3 "${SCRIPT_DIR}/run_benchmarks.py" \
        --benchmark="$BENCHMARK" \
        --model_location="cpu" --execution_location="gpu" \
        --thread_percentage="$i" \
        --runs="$RUNS" --warmup="$WARMUP" \
        --csv="$CSV"

    # Cold start: gpu/gpu
    python3 "${SCRIPT_DIR}/run_benchmarks.py" \
        --benchmark="$BENCHMARK" \
        --model_location="gpu" --execution_location="gpu" \
        --thread_percentage="$i" \
        --cold_start \
        --runs="$RUNS" \
        --csv="$CSV"

    # Cold start: cpu/gpu
    python3 "${SCRIPT_DIR}/run_benchmarks.py" \
        --benchmark="$BENCHMARK" \
        --model_location="cpu" --execution_location="gpu" \
        --thread_percentage="$i" \
        --cold_start \
        --runs="$RUNS" \
        --csv="$CSV"
done

# Stop CUDA MPS daemon
echo ""
echo "Stopping CUDA MPS control daemon..."
echo quit | nvidia-cuda-mps-control 2>/dev/null || echo "  (MPS daemon was not running)"

echo ""
echo "============================================"
echo "All benchmarks complete!"
echo "Results written to: ${CSV}"
echo "============================================"
