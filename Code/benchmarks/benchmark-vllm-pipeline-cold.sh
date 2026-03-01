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

NR_COLD_ITERATIONS=5

pip install pandas
pip install matplotlib
pip install seaborn

# Cleanup function to ensure MPS daemon is stopped
cleanup() {
  echo "Stopping MPS daemon..."
  echo quit | nvidia-cuda-mps-control
}

# Set trap to call cleanup on EXIT (success, failure, or interruption)
trap cleanup EXIT

# ============================================================================
# GPU cold-start
# ============================================================================

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-$(id -un)
nvidia-cuda-mps-control -d

echo "=== GPU cold-start benchmarks ==="

# Remove stale cold-start CSVs so appends start fresh
rm -f "results/$MODEL"/gpu-*-cold.csv

for ((c=0; c<NR_COLD_ITERATIONS; c++)); do
  echo "--- Cold iteration $((c+1))/$NR_COLD_ITERATIONS ---"
  python run-vllm-cold.py --model="$MODEL" --execution_location="gpu" --mode=1
  python run-vllm-cold.py --model="$MODEL" --execution_location="gpu" --mode=2
  #python run-vllm-cold.py --model="$MODEL" --execution_location="gpu" --mode=3
  #python run-vllm-cold.py --model="$MODEL" --execution_location="gpu" --mode=4
done


#echo quit | nvidia-cuda-mps-control

# ============================================================================
# CPU cold-start
# ============================================================================

#=(1 2 4 8 16 36 72)

#echo "=== CPU cold-start benchmarks ==="
#for i in "${CPU_THREADS[@]}"; do
#  echo "OMP_NUM_THREADS=$i"
#  export OMP_NUM_THREADS="$i"
#  export MKL_NUM_THREADS="$i"
#
#  python run-vllm-cold.py --model="$MODEL" --execution_location="cpu" --mode=1
#done
