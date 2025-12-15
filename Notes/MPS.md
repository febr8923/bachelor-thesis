# How MPS works

### Overview

→ build to use with multi-process CUDA applications (or MPI processes accessing the same gpu)

- Why use GPU fully?
    - More efficient
    - Usually performs better
- Clients submit jobs to gpu and MPS then “splits” gpu (=HyperQ: process cuda kernels concurrently on one gpu)

![Volta MPS](How%20MPS%20works/image.png)

Volta MPS

### Why MPS

- Inefficient use of GPU (e.g. mutli-process system on CPU submits MPI processes when necessary (to give maximal parallelization), this may result in MPI processes that underutilize the GPU)
- → Inter MPI rank parallelism

### What is MPS

- Client-server model of CUDA API, consisting of:
    - Clients: MPS client runtime (built into CUDA driver library, can be used by any CUDA application)
    - Server: Clients share connection to GPU and get provides concurrency through server
    - (+ Control deamon process (starting/stopping server, coordinating connections)
- `man nvidia-cuda-mps-control` or `man nvidia-smi`

### Benefits of MPS

- GPU Utilization
    - Allows kernel and memcopy operations from different processes to overlap on GPU → higher utilization and shorter running times
- Reduced On-GPU Cotext Storage
    - Can use one shared copy of GPU storage and scheduling resources for all clients
    - Volta MPS has memory isolation so not there
- Reduced GPU context switching
    - Shared one set of scheduling resources between all clients → no overhead of swapping between clients

### When useful?

- One process not enough work for GPU (Small nr. of blocks-per-grid)
    - Threads → blocks → grid
    - GPU has multiple processing units (SMs), each block is assigned to an SM)
    - There is no direct way to control nr. of SMs used ([https://forums.developer.nvidia.com/t/fixing-sms-for-a-kernel/44619](https://forums.developer.nvidia.com/t/fixing-sms-for-a-kernel/44619))
    - Setting MPS Limits
    Start MPS server: `nvidia-cuda-mps-control -d` (daemon mode).
    Set per-client SM limit via env var `CUDA_MPS_PIPE_DIRECTORY` or config file `/tmp/nvidia-mps-control.conf`—but standard MPS doesn’t directly cap SMs; use nvidia-smi or CUDA_VISIBLE_DEVICES for coarse limits.
    For precise limits, use MIG (Multi-Instance GPU) on A100/H100: `nvidia-smi mig -cgi 15,3` creates partitions (e.g., 3 SMs), then run in that instance.
    Active MPS client: `export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50` (limits compute ~50%, indirect SM cap). Clients exit with `echo quit | nvidia-cuda-mps-control`.

### MPS compute modes

- PROHIBITED
- EXCLUSIVE_PROCESS: Assinged to only one process at a time, threads of one process can submit concurrently
- DEFAULT: multiple processes can use GPU simulaneously, Threads my submit work simultaneously
- Using MPS: EXCLUSIVE_PROCESS behaves like DEFAULT for all MPS clients (Multiple clients can always submit work concurrently) → use EXCLUSIVE_PROCESS to ensure only one server is using GPU.

### Memory protection

# How MPS works on CSCS - clariden

referring to https://docs.cscs.ch/software/container-engine/resource-hook/#nvidia-cuda-mps-hook