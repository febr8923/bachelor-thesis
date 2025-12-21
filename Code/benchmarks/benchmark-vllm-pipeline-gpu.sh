#!/usr/bin/env bash

MODEL="$1"

if [ -z "$MODEL" ]; then
  echo "Usage: $0 <model_name>"
  exit 1
fi

export TRANSFORMERS_CACHE="$SCRATCH"
export TORCH_HOME="$SCRATCH"
export HF_HOME="$SCRATCH"
export PYTHONWARNINGS=ignore

pip install pandas
pip install matplotlib


# Cleanup function to ensure MPS daemon is stopped
cleanup() {
  echo "Stopping MPS daemon..."
  echo quit | nvidia-cuda-mps-control
}

# Set trap to call cleanup on EXIT (success, failure, or interruption)
trap cleanup EXIT

# remove previous results:
#rm -rf "results/$MODEL"

# gpu

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-$(id -un)
nvidia-cuda-mps-control -d

python run-vllm.py --model="$MODEL" --mode=4 --measure_memory

echo quit | nvidia-cuda-mps-control