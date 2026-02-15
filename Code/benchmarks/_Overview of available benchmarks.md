# Overview of available benchmarks

## vLLM

available for gpt-j-6b, llama-8b

|  | warm | cold | thread-sweep | sm-sweep | gpu-memory/-util | cpu-memory/-util | kv-cache-size sweep |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gpu exec | total |  | total | total | total | total | total |
| cpu exec | total |  | total | total | total | total | total |

## DL

available for alexnet, resnet50, vgg19, bert

|  | warm | cold | thread-sweep | sm-sweep | gpu-memory/-util | cpu-memory/-util | kv-cache-size sweep |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cpu-cpu | (load, exec) | (load, exec) | (load, exec) | (load, exec) |  | x (not bert) | n.a. |
| gpu-cpu | (load, exec) | (load, exec) | (load, exec) | (load, exec) |  |  | n.a. |
| cpu-gpu | (load, exec) | (load, exec) | (load, exec) | (load, exec) | x (not bert) |  | n.a. |
| gpu-gpu | (load, exec) | (load, exec) | (load, exec) | (load, exec) |  |  | n.a. |

## Scientific

leukocyte, nn, bfs

|  | warm | cold | thread-sweep | sm-sweep | gpu-memory/-util | cpu-memory/-util | kv-cache-size sweep |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cpu-cpu |  |  | x | x |  | x | n.a. |
| gpu-cpu | x |  | x | x |  |  | n.a. |
| cpu-gpu | x |  | x | x | x |  | n.a. |
| gpu-gpu | x |  | x | x |  |  | n.a. |

conv2d, nbody

|  | warm | cold | thread-sweep | sm-sweep | gpu-memory/-util | cpu-memory/-util | kv-cache-size sweep |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cpu-cpu | (load, exec) needs to be checked |  | (load, exec) needs to be checked | (load, exec) needs to be checked |  |  | n.a. |
| gpu-cpu | (load, exec) needs to be checked |  |  |  |  |  | n.a. |
| cpu-gpu | (load, exec) needs to be checked |  | (load, exec) needs to be checked | (load, exec) needs to be checked |  |  | n.a. |
| gpu-gpu |  |  |  |  |  |  | n.a. |