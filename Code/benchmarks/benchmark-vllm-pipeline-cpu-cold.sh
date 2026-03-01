#!/usr/bin/env bash

MODEL="$1"

if [ -z "$MODEL" ]; then
  echo "Usage: $0 <model_name>"
  exit 1
fi

export TRANSFORMERS_CACHE="$SCRATCH"
export TORCH_HOME="$SCRATCH"
export PYTHONWARNINGS=ignore

CPU_THREADS=(1 2 4 8 16 36 72)

echo "=== CPU cold-start benchmarks ==="
for i in "${CPU_THREADS[@]}"; do
  echo "OMP_NUM_THREADS=$i"
  export OMP_NUM_THREADS="$i"
  export MKL_NUM_THREADS="$i"

  python run-vllm-cold.py --model="$MODEL" --execution_location="cpu" --mode=1
done
