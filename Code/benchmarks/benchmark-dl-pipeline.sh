#!/usr/bin/env bash

MODEL="$1"

if [ -z "$MODEL" ]; then
  echo "Usage: $0 <model_name>"
  exit 1
fi

export TRANSFORMERS_CACHE="$SCRATCH"
export TORCH_HOME="$SCRATCH"
export PYTHONWARNINGS=ignore


# Cleanup function to ensure MPS daemon is stopped
cleanup() {
  echo "Stopping MPS daemon..."
  echo quit | nvidia-cuda-mps-control
}

# Set trap to call cleanup on EXIT (success, failure, or interruption)
trap cleanup EXIT

# remove previous results:
rm -rf "results/$MODEL"

CPU_THREADS=(1 2 4 8 16 36 72)

# cpu
echo "cpu tests"
for i in "${CPU_THREADS[@]}"; do
  echo "$i"
  export OMP_NUM_THREADS="$i"
  export MKL_NUM_THREADS="$i"
  
  python run-dl.py --model="$MODEL" --model_location="cpu" --execution_location="cpu" --thread_percentage="$i"
  python run-dl.py --model="$MODEL" --model_location="gpu" --execution_location="cpu" --thread_percentage="$i"
  
  python run-dl.py --model="$MODEL" --model_location="cpu" --cold_start --execution_location="cpu" --thread_percentage="$i"
  python run-dl.py --model="$MODEL" --model_location="gpu" --cold_start --execution_location="cpu" --thread_percentage="$i"
done

# gpu

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-$(id -un)
nvidia-cuda-mps-control -d

echo "gpu tests"
for ((i=10; i<=100; i+=10)); do
  echo "$i"
  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$i"
  #echo "$CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
  python run-dl.py --model="$MODEL" --model_location="gpu" --execution_location="gpu" --thread_percentage="$i"
  python run-dl.py --model="$MODEL" --model_location="cpu" --execution_location="gpu" --thread_percentage="$i"

  python run-dl.py --model="$MODEL" --model_location="gpu" --cold_start --execution_location="gpu" --thread_percentage="$i"
  python run-dl.py --model="$MODEL" --model_location="cpu" --cold_start --execution_location="gpu" --thread_percentage="$i"
done
python make-graph.py --model="$MODEL"
echo quit | nvidia-cuda-mps-control