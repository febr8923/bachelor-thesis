FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ccache git curl wget ca-certificates \
    gcc-12 g++-12 \
    libtcmalloc-minimal4 libnuma-dev \
    ffmpeg libsm6 libxext6 libgl1 jq lsof \
    python3 python3-pip python3.12-venv python3-dev python3.12-dev \
 && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12

WORKDIR /workspace
RUN git clone https://github.com/vllm-project/vllm.git
WORKDIR /workspace/vllm

RUN git checkout v0.11.0

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip

# PyTorch CPU
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Build deps
RUN pip install "cmake>=3.26" wheel packaging ninja "setuptools-scm>=8" numpy

# vLLM CPU deps
RUN pip install -v -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Build vLLM for CPU
ENV VLLM_TARGET_DEVICE=cpu
ENV MAX_JOBS=48
RUN python setup.py install

# Adjust this path for your arch; for x86_64 usually:
# ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"
ENV LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libtcmalloc_minimal.so.4"
