FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ccache git curl wget ca-certificates gcc-12 g++-12 libtcmalloc-minimal4 libnuma-dev ffmpeg libsm6 libxext6 libgl1 jq lsof python3 git python3-pip python3.12-venv python3-dev python3.12-dev && \
   rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12

WORKDIR /workspace


RUN git clone https://github.com/vllm-project/vllm.git
WORKDIR /workspace/vllm


RUN git checkout v0.11.0

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip

#RUN pip install --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchvision torchaudio

# NEW (CUDA for GH200):
RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

RUN python use_existing_torch.py
RUN pip install -r requirements/build.txt
ENV MAX_JOBS=48
RUN pip install --no-build-isolation .

ENV LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libtcmalloc_minimal.so.4"
