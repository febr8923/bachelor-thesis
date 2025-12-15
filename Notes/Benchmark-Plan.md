
# How to run benchmarks

### Evaluation methodology

Goal: measure times of different DL kernels on GH200 with different parameters.

Parameters:

- Cold vs warm start
- Model location (disk, cpu/memory, gpu)
- Execution location (cpu, gpu)
    - If GPU: changing percentage of SMs (Streaming Multiprocessors) w. MPS-> to find optimum
        - To find dependency on the SMs we set the CUDA_MPS_ACTIVE_THREAD_PERCENTAGE environment variable to the elements from the set {10, 20, …, 100}. Additionally setting CUDA_VISIBLE_DEVICES might be able to simulate different availability scenarios. ([https://stackoverflow.com/questions/69531995/how-to-control-the-resource-of-each-client-in-nvidia-mps](https://stackoverflow.com/questions/69531995/how-to-control-the-resource-of-each-client-in-nvidia-mps), [https://docs.nvidia.com/deploy/mps/index.html](https://docs.nvidia.com/deploy/mps/index.html))
    - If cpu: changing nr of threads

Following initial times can be measured 1) before moving to cpu/memory 2) after moving to cpu/memory 3) after moving to gpu.

### Evaluation categories

- Scientific
    - Rodina (one from each category) ([https://rodinia.cs.virginia.edu/](https://rodinia.cs.virginia.edu/))
        - Leucocyte (medical imaging, structured grid
        - Breadth-first search (Graph traversal)
        - kNN (Dense Linear Algebra)
    - Npbench ([https://github.com/spcl/npbench/tree/main/npbench/benchmarks](https://github.com/spcl/npbench/tree/main/npbench/benchmarks))
        - DL: conv2d_bias
        - N-body
- ML
    - Vgg
    - Alexnet
    - Resnet (dgsf, pask)
    - Bert
- LLM (w. vLLM)
    - GPT-j 6B mid-size llm tests
    - Llama3 8B: different model
    - Llama3 405B: large-size (if possible)

### Changing SMs