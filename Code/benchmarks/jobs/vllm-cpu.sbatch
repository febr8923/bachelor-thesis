#!/usr/bin/env bash
#SBATCH --job-name=bench-vllm-cpu
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/vllm-cpu-%j.out
#SBATCH --error=logs/vllm-cpu-%j.err

MODEL="meta-llama/Llama-3.1-8B"
CONTAINER="$SCRATCH/ce-images/vllm-cuda-12.8.sqsh"
WORKDIR="$SCRATCH/benchmarks"

srun --container-image="$CONTAINER" \
     --container-workdir="$WORKDIR" \
     --container-mounts="$SCRATCH:$SCRATCH" \
     bash benchmark-vllm-pipeline-cpu.sh "$MODEL"
