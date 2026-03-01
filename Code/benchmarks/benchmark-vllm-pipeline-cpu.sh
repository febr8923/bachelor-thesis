#!/usr/bin/env bash

MODEL="$1"

if [ -z "$MODEL" ]; then
  echo "Usage: $0 <model_name>"
  exit 1
fi

export TRANSFORMERS_CACHE="$SCRATCH"
export TORCH_HOME="$SCRATCH"
export PYTHONWARNINGS=ignore

# remove previous results:
#rm -rf "results/$MODEL"

CPU_THREADS=(1 2 4 8 16 36 72)
KV_SIZES=(4 8 16 32 64 96 128)

# ============================================================================
# Sweep 1: varying thread count (mode 1)
# ============================================================================
echo "=== CPU tests: sweeping thread count ==="
for i in "${CPU_THREADS[@]}"; do
  echo "OMP_NUM_THREADS=$i"
  export OMP_NUM_THREADS="$i"
  export MKL_NUM_THREADS="$i"
  export VLLM_CPU_KVCACHE_SPACE="16"

  python run-vllm.py --model="$MODEL" --execution_location="cpu" --mode=1 --measure_memory
done

# ============================================================================
# Sweep 2: varying KV cache size (mode 6)
# ============================================================================
echo "=== CPU tests: sweeping KV cache size ==="
export OMP_NUM_THREADS="72"
export MKL_NUM_THREADS="72"
for kv in "${KV_SIZES[@]}"; do
  echo "VLLM_CPU_KVCACHE_SPACE=$kv"
  export VLLM_CPU_KVCACHE_SPACE="$kv"

  python run-vllm.py --model="$MODEL" --execution_location="cpu" --mode=6
done

# ============================================================================
# Standard benchmarks: batch size, input length
# ============================================================================
echo "=== CPU tests: batch size & input length ==="
export OMP_NUM_THREADS="72"
export MKL_NUM_THREADS="72"
export VLLM_CPU_KVCACHE_SPACE="16"

python run-vllm.py --model="$MODEL" --execution_location="cpu" --mode=2
python run-vllm.py --model="$MODEL" --execution_location="cpu" --mode=3

#python make-plot.py --model="$MODEL" --vllm --cpu_only
