# How to install vllm (or do inference)

Referring to: [https://docs.cscs.ch/tutorials/ml/llm-inference/](https://docs.cscs.ch/tutorials/ml/llm-inference/) and Mattermost chat

# **vLLM on Alps (gh200 nodes)**

While you follow this guide, also take a look at:
[https://docs.cscs.ch/tutorials/ml/llm-inference/](https://docs.cscs.ch/tutorials/ml/llm-inference/) .

# **Prepare the image**

### **Dockerfile**

The available NVIDIA image for vLLM is compatible only with CUDA 13, so we need to build a custom one. Create the following Dockerfile:

```
FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ccache git curl wget ca-certificates gcc-12 g++-12 libtcmalloc-minimal4 libnuma-dev ffmpeg libsm6 libxext6 libgl1 jq lsof python3 git python3-pip python3.12-venv python3-dev python3.12-dev && \
   rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12

WORKDIR /workspace

RUN git clone https://github.com/vllm-project/vllm.git
WORKDIR /workspace/vllm

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip

RUN pip install --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchvision torchaudio
RUN python use_existing_torch.py
RUN pip install -r requirements/build.txt
ENV MAX_JOBS=48
RUN pip install --no-build-isolation .

ENV LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libtcmalloc_minimal.so.4"
```

### Configure Storage

```bash
[storage]
driver = "overlay"
runroot = "/dev/shm/$USER/runroot"
graphroot = "/dev/shm/$USER/root"

[storage.options.overlay]
mount_program = "/usr/bin/fuse-overlayfs-1.13"
```

### Configure sth else

```bash
mkdir -p $SCRATCH/ce-images
lfs setstripe -E 4M -c 1 -E 64M -c 4 -E -1 -c -1 -S 4M $SCRATCH/ce-images
```

### **Build the image**

Request an interactive allocation on Clariden/Daint:

```bash
srun -A <ACCOUNT> --pty bash
```

Build the image by running:

```
podman build -t <insert_the_name_of_the_image> -f Dockerfile .
nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

podman build -t nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04 -f Dockerfile .
```

Then import the image:

```
enroot import -x mount \
  -o $SCRATCH/ce-images/<insert_the_name_of_the_image>.sqsh \
  podman://<insert_the_name_of_the_image>
  
  
  
 enroot import -x mount \
  -o $SCRATCH/ce-images/cuda-12.8.0-cudnn-devel-ubuntu24.04.sqsh \
  podman://nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

```

Now you can exit the interactive session and set up the EDF configuration.
Create a `<insert_the_name_of_the_image>.toml` file with the following content:

```bash
image = "${SCRATCH}/ce-images/vllm-cuda-12.8.sqsh"

mounts = [
    "/capstor",
    "/iopsstor"
] 

workdir = "${SCRATCH}/" 

[annotations]
com.hooks.aws_ofi_nccl.enabled = "true" 
com.hooks.aws_ofi_nccl.variant = "cuda12"

[env]
NCCL_DEBUG = "WARN" 
CUDA_CACHE_DISABLE = "1" 
TORCH_NCCL_ASYNC_ERROR_HANDLING = "1" 
MPICH_GPU_SUPPORT_ENABLED = "0"
CUDA_VISIBLE_DEVICES= "0,1,2,3"
LOG_LEVEL = "WARN"
```

# **Use the image**

Simply run the following command to allocate the required resources and use the newly created image:

```basic
srun -A <ACCOUNT> \
  --environment=./<insert_the_name_of_the_image>.toml --pty bash
  
srun -A <ACCOUNT> \
--environment=./vllm-cuda-12.8.toml --pty bash
  
srun -A a-g200 --environment=./cuda:12.8.0-cudnn-devel-ubuntu24.04.toml --pty bash
```