#!/usr/bin/env bash

pip install pandas matplotlib seaborn
MODEL="$1"
WATCH_MEMORY=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --watch-memory)
      WATCH_MEMORY=true
      shift
      ;;
    *)
      if [ -z "$MODEL" ]; then
        MODEL="$1"
      fi
      shift
      ;;
  esac
done

if [ -z "$MODEL" ]; then
  echo "Usage: $0 <model_name> [--watch-memory]"
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

# Start CPU watcher if requested
CPU_WATCHER_PID=""
if [ "$WATCH_MEMORY" = true ]; then
  CPU_WATCHER_OUTPUT="results/${MODEL}_cpu_memory_usage.csv"
  echo "Starting CPU watcher, output: $CPU_WATCHER_OUTPUT"
  python start_watcher.py cpu 0 "$CPU_WATCHER_OUTPUT" &
  CPU_WATCHER_PID=$!
  echo "CPU watcher started (PID: $CPU_WATCHER_PID)"
  sleep 2  # Give watcher time to initialize
fi

for i in "${CPU_THREADS[@]}"; do
  echo "$i"
  export OMP_NUM_THREADS="$i"
  export MKL_NUM_THREADS="$i"
  
  python run-dl-clean.py --model="$MODEL" --model_location="cpu" --execution_location="cpu" --mode=1
  python run-dl-clean.py --model="$MODEL" --model_location="gpu" --execution_location="cpu" --mode=1
  
  python run-dl-clean.py --model="$MODEL" --model_location="cpu" --cold_start --execution_location="cpu" --mode=1
  python run-dl-clean.py --model="$MODEL" --model_location="gpu" --cold_start --execution_location="cpu" --mode=1
done

# Stop CPU watcher
if [ -n "$CPU_WATCHER_PID" ]; then
  echo "Stopping CPU watcher (PID: $CPU_WATCHER_PID)..."
  kill $CPU_WATCHER_PID 2>/dev/null
  wait $CPU_WATCHER_PID 2>/dev/null
  echo "CPU watcher stopped"
fi

# gpu

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-$(id -un)
nvidia-cuda-mps-control -d

echo "gpu tests"

# Start GPU watcher if requested
GPU_WATCHER_PID=""
if [ "$WATCH_MEMORY" = true ]; then
  GPU_WATCHER_OUTPUT="results/${MODEL}_gpu_memory_usage.csv"
  echo "Starting GPU watcher, output: $GPU_WATCHER_OUTPUT"
  python start_watcher.py gpu 0 "$GPU_WATCHER_OUTPUT" &
  GPU_WATCHER_PID=$!
  echo "GPU watcher started (PID: $GPU_WATCHER_PID)"
  sleep 2  # Give watcher time to initialize
fi

for ((i=10; i<=100; i+=10)); do
  echo "$i"
  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$i"
  #echo "$CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
  python run-dl-clean.py --model="$MODEL" --model_location="gpu" --execution_location="gpu" --mode=1
  python run-dl-clean.py --model="$MODEL" --model_location="cpu" --execution_location="gpu" --mode=1

  python run-dl-clean.py --model="$MODEL" --model_location="gpu" --cold_start --execution_location="gpu" --mode=1
  python run-dl-clean.py --model="$MODEL" --model_location="cpu" --cold_start --execution_location="gpu" --mode=1
done

# Stop GPU watcher
if [ -n "$GPU_WATCHER_PID" ]; then
  echo "Stopping GPU watcher (PID: $GPU_WATCHER_PID)..."
  kill $GPU_WATCHER_PID 2>/dev/null
  wait $GPU_WATCHER_PID 2>/dev/null
  echo "GPU watcher stopped"
fi

#python make-plot.py --model="$MODEL"
echo quit | nvidia-cuda-mps-control