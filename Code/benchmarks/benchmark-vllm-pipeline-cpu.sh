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

python run-vllm.py --model="$MODEL" --mode=1 --execution_location=cpu --measure_memory
python run-vllm.py --model="$MODEL" --mode=2 --execution_location=cpu
python run-vllm.py --model="$MODEL" --mode=3 --execution_location=cpu

#python make-plot.py --model="$MODEL" --vllm --cpu_only
