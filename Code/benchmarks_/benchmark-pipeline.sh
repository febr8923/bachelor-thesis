#!/usr/bin/env bash

MODEL = "$1"
TYPE = "$2" #dl, llm, vllm or scientifc
EXEC_LOC = "$3" #gpu, cpu, or both (gpu is default)
MODE = "$4" #0: variable available resources; 1: monitor memory and compute usage, 3: both

if [ -z "$MODEL" ]; then
  echo "Usage: $0 <model_name> <model_type> (<exec_loc>) (<mode>)"
  exit 1
fi

if [ -z "$TYPE" ]; then
  echo "Usage: $0 <model_name> <model_type> (<exec_loc>) (<mode>)"
  exit 1
fi

if [ -z "$EXEC_LOC" ]; then
  MODEL = "gpu"
fi

if [ -z "$MODE" ]; then
  MODEL = 0
fi