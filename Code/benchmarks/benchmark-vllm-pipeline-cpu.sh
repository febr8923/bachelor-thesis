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
echo "cpu tests"
for i in "${CPU_THREADS[@]}"; do
  echo "$i"
  export OMP_NUM_THREADS="$i"
  export MKL_NUM_THREADS="$i"
  
  python run-vllm.py --model="$MODEL" --model_location="cpu" --execution_location="cpu" --thread_percentage="$i"
  python run-vllm.py --model="$MODEL" --model_location="cpu" --cold_start --execution_location="cpu" --thread_percentage="$i"
done

python make-plot.py --model="$MODEL" --vllm --cpu_only
