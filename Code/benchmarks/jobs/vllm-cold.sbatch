#!/usr/bin/env bash
#SBATCH --job-name=bench-vllm-cold
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/vllm-cold-%j.out
#SBATCH --error=logs/vllm-cold-%j.err

MODEL="meta-llama/Llama-3.1-8B"
CONTAINER="$SCRATCH/ce-images/vllm-cuda-12.8.sqsh"
WORKDIR="$SCRATCH/benchmarks"

srun --container-image="$CONTAINER" \
     --container-workdir="$WORKDIR" \
     --container-mounts="$SCRATCH:$SCRATCH" \
     bash benchmark-vllm-pipeline-cold.sh "$MODEL"
