# single_run

Single-execution scripts for targeted profiling (one iteration, no sweep).

---

## Scripts

### vllm_memory.py

Loads a vLLM model, runs one inference, saves GPU memory time-series via `nvidia-smi` (10 ms intervals).

```bash
# run from benchmarks/
python single_run/vllm_memory.py --model EleutherAI/gpt-j-6b \
    --output_csv results/EleutherAI/gpt-j-6b/single_memory.csv \
    --nr_input_tokens 128 --nr_output_tokens 128 --memory_rate 0.85
```

| Argument | Default | Description |
|---|---|---|
| `--model` | *(required)* | HuggingFace model name |
| `--output_csv` | `memory_trace.csv` | Output CSV path |
| `--nr_input_tokens` | `128` | Input prompt length |
| `--nr_output_tokens` | `128` | Max tokens to generate |
| `--memory_rate` | `0.85` | vLLM `gpu_memory_utilization` |

Plot output: `python plot_memory_gpu.py --csv <output_csv>`

---

### vllm_memory_cpu.py

Same as above but forces CPU-only execution (`CUDA_VISIBLE_DEVICES=""`). Tracks RAM via `psutil` (0.5 s intervals) using `CpuWatcher`.

```bash
python single_run/vllm_memory_cpu.py --model EleutherAI/gpt-j-6b \
    --output_csv results/EleutherAI/gpt-j-6b/single_cpu_memory.csv \
    --nr_input_tokens 128 --nr_output_tokens 32
```

| Argument | Default | Description |
|---|---|---|
| `--model` | *(required)* | HuggingFace model name |
| `--output_csv` | `cpu_memory_trace.csv` | Output CSV path |
| `--nr_input_tokens` | `128` | Input prompt length |
| `--nr_output_tokens` | `32` | Max tokens to generate (keep low — CPU is slow) |

Plot output: `python plot_memory_cpu.py --csv <output_csv>`

---

### vllm_cold_clear_cache.py

Cold-start vLLM benchmark that evicts model weights from the OS page cache between iterations via `posix_fadvise(POSIX_FADV_DONTNEED)` — no sudo required. Guarantees every iteration reads from disk, not RAM.

Output CSV has one row per iteration: `iteration, load_time, infer_time, total_time, throughput, ttft`.

```bash
python single_run/vllm_cold_clear_cache.py --model EleutherAI/gpt-j-6b \
    --iterations 5 \
    --output_csv results/EleutherAI/gpt-j-6b/true_cold.csv
```

| Argument | Default | Description |
|---|---|---|
| `--model` | *(required)* | HuggingFace model name |
| `--iterations` | `5` | Number of cold-start iterations |
| `--output_csv` | `true_cold.csv` | Output CSV path |
| `--nr_input_tokens` | `128` | Input prompt length |
| `--nr_output_tokens` | `128` | Max tokens to generate |
| `--memory_rate` | `0.85` | vLLM `gpu_memory_utilization` |
| `--cache_dir` | auto-detect | HF cache dir (`HF_HOME` → `TRANSFORMERS_CACHE` → `~/.cache/huggingface/hub`) |

---

### dl_memory_gpu.py

Loads a torchvision model, runs one GPU inference, saves GPU memory time-series via `nvidia-smi` (10 ms intervals). `--model_loc` sets where data starts (cpu = includes H2D transfer, gpu = already on device).

Requires `dog.jpg` in the working directory.

```bash
python single_run/dl_memory_gpu.py --model resnet50 --model_loc cpu \
    --output_csv results/resnet50/single_gpu_memory.csv
```

| Argument | Default | Description |
|---|---|---|
| `--model` | *(required)* | Torchvision model name (`resnet50`, `vgg19`, `alexnet`) |
| `--model_loc` | `cpu` | Starting data location before GPU transfer |
| `--output_csv` | `dl_gpu_memory_trace.csv` | Output CSV path |

Plot output: `python plot_memory_gpu.py --csv <output_csv>`

---

### dl_memory_cpu.py

Same as above but runs entirely on CPU. Tracks RAM via `psutil` (0.5 s intervals).

Requires `dog.jpg` in the working directory.

```bash
python single_run/dl_memory_cpu.py --model resnet50 \
    --output_csv results/resnet50/single_cpu_memory.csv
```

| Argument | Default | Description |
|---|---|---|
| `--model` | *(required)* | Torchvision model name (`resnet50`, `vgg19`, `alexnet`) |
| `--output_csv` | `dl_cpu_memory_trace.csv` | Output CSV path |

Plot output: `python plot_memory_cpu.py --csv <output_csv>`
