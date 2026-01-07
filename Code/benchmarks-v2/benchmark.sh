#!/usr/bin/env bash
#
# Unified benchmark shell wrapper
# Handles environment setup and CUDA MPS for GPU benchmarks
#

set -e  # Exit on error

# Usage function
usage() {
    cat << EOF
Usage: $0 <type> <model> <device> <mode> [options]

Arguments:
    type       Benchmark type (vllm or image)
    model      Model name (HuggingFace for vLLM, PyTorch Hub for image)
    device     Device to use (cpu or gpu)
    mode       Benchmark mode (1-5)

Options:
    --measure-memory    Enable memory monitoring
    --plot             Generate plots after benchmarking
    --model-location   Initial model location (image benchmarks only)

Examples:
    $0 vllm gpt2 gpu 5 --plot
    $0 image resnet50 gpu 2 --measure-memory
    $0 vllm meta-llama/Llama-2-7b-hf cpu 1

Modes:
    1: Resource variation (SM% for GPU, threads for CPU)
    2: Batch size variation
    3: Input length variation (vLLM only)
    4: Memory utilization (GPU only, vLLM only)
    5: All applicable benchmarks
EOF
    exit 1
}

# Check arguments
if [ $# -lt 4 ]; then
    usage
fi

TYPE="$1"
MODEL="$2"
DEVICE="$3"
MODE="$4"
shift 4

# Validate type
if [[ "$TYPE" != "vllm" && "$TYPE" != "image" ]]; then
    echo "Error: Type must be 'vllm' or 'image'"
    usage
fi

# Validate device
if [[ "$DEVICE" != "cpu" && "$DEVICE" != "gpu" ]]; then
    echo "Error: Device must be 'cpu' or 'gpu'"
    usage
fi

# Validate mode
if [[ ! "$MODE" =~ ^[1-5]$ ]]; then
    echo "Error: Mode must be between 1 and 5"
    usage
fi

# Set environment variables
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${SCRATCH:-/tmp}}"
export TORCH_HOME="$TRANSFORMERS_CACHE"
export HF_HOME="$TRANSFORMERS_CACHE"
export PYTHONWARNINGS=ignore

echo "================================"
echo "Benchmark Configuration"
echo "================================"
echo "Type:   $TYPE"
echo "Model:  $MODEL"
echo "Device: $DEVICE"
echo "Mode:   $MODE"
echo "Cache:  $TRANSFORMERS_CACHE"
echo "================================"
echo ""

# MPS cleanup function
cleanup_mps() {
    if [ "$DEVICE" == "gpu" ] && [ -n "$MPS_STARTED" ]; then
        echo ""
        echo "Stopping CUDA MPS daemon..."
        echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    fi
}

# Set trap for cleanup
trap cleanup_mps EXIT INT TERM

# Start CUDA MPS for GPU benchmarks
if [ "$DEVICE" == "gpu" ]; then
    echo "Starting CUDA MPS daemon..."
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
    export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-$(id -un)

    # Stop any existing MPS daemon
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    sleep 1

    # Start MPS daemon
    nvidia-cuda-mps-control -d
    MPS_STARTED=1
    echo "CUDA MPS started"
    echo ""
fi

# Build Python command
PYTHON_CMD="python run_benchmark.py --type $TYPE --model $MODEL --device $DEVICE --mode $MODE"

# Add optional arguments
for arg in "$@"; do
    PYTHON_CMD="$PYTHON_CMD $arg"
done

# Run the benchmark
echo "Running benchmark..."
echo "Command: $PYTHON_CMD"
echo ""

$PYTHON_CMD

# MPS cleanup will be called by trap
exit $?
