#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00

#SBATCH --uenv=pytorch/v2.6.0:/user-environment
#SBATCH --view=default

#################################
# OpenMP environment variables #
#################################
export OMP_NUM_THREADS=8 

#################################
# PyTorch environment variables #
#################################
export MASTER_ADDR=$(hostname) 
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NPROCS
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 
export TRITON_HOME=/dev/shm/ 

#################################
# MPICH environment variables   #
#################################
export MPICH_GPU_SUPPORT_ENABLED=0 

#################################
# CUDA environment variables    #
#################################
export CUDA_CACHE_DISABLE=1 

############################################
# NCCL and Fabric environment variables    #
############################################

# This forces NCCL to use the libfabric plugin, enabling full use of the
# Slingshot network. If the plugin can not be found, applications will fail to
# start. With the default value, applications would instead fall back to e.g.
# TCP, which would be significantly slower than with the plugin. More information
# about `NCCL_NET` can be found at:
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-net
export NCCL_NET="AWS Libfabric"
# Use GPU Direct RDMA when GPU and NIC are on the same NUMA node. More
# information about `NCCL_NET_GDR_LEVEL` can be found at:
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-net-gdr-level-formerly-nccl-ib-gdr-level
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
# These `FI` (libfabric) environment variables have been found to give the best
# performance on the Alps network across a wide range of applications. Specific
# applications may perform better with other values.
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=32768
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_RX_MATCH_MODE=software
export FI_MR_CACHE_MONITOR=userfaultfd




srun bash -c "
    export LD_LIBRARY_PATH=/users/fbrunne/miniconda3/envs/rmm_dev/lib:\$LD_LIBRARY_PATH
    export RANK=\\\$SLURM_PROCID
    export LOCAL_RANK=\\\$SLURM_LOCALID
    ~/miniconda3/envs/rmm_dev/bin/python run-test.py"
"