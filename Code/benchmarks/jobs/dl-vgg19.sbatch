#!/usr/bin/env bash
#SBATCH --job-name=bench-dl-vgg19
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/dl-vgg19-%j.out
#SBATCH --error=logs/dl-vgg19-%j.err

MODEL="vgg19"
CONTAINER="$SCRATCH/ce-images/vllm-cuda-12.8.sqsh"
WORKDIR="$SCRATCH/benchmarks"

srun --container-image="$CONTAINER" \
     --container-workdir="$WORKDIR" \
     --container-mounts="$SCRATCH:$SCRATCH" \
     bash benchmark-dl-pipeline.sh "$MODEL"
