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

# cpu
#echo "cpu tests"
#for i in "${CPU_THREADS[@]}"; do
#  echo "$i"
#  export OMP_NUM_THREADS="$i"
#  export MKL_NUM_THREADS="$i"
#  
#  python run-vllm.py --model="$MODEL" --execution_location="cpu"
#done

#KV_SIZE=(4 8 16 32 64 96 128)
#for in in "${KV_SIZE[@]}"; do
#  
#  python run-vllm.py --model="$MODEL" --execution_location="cpu" --measure_memory --mode="6"
#
#done

#python make-plot.py --model="$MODEL" --vllm --cpu_only
