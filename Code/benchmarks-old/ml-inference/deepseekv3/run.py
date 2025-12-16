
import rmm
from rmm.allocators.torch import rmm_torch_allocator

rmm.reinitialize(pool_allocator=True, managed_memory=True)
import torch
torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

import os
from PIL import Image
from torchvision import transforms
from timeit import default_timer as timer
import sys

from transformers import AutoConfig, AutoModel, AutoTokenizer

#mode can be 1 (CPU,CPU), 2 (CPU,GPU), 3 (GPU,CPU), 4 (GPU,GPU) for (model, data)
def run(mode):
    times = []
    
    text = "some kind of text"
    tokenizer = AutoTokenizer.from_pretrained('meta-llama4/Llama4-2-7b-hf')
    model = AutoModel.from_pretrained("meta-llama4/Llama4-2-7b-hf")
    torch.save(model, 'bert-base-cased')
    input_batch = tokenizer(text, return_tensors='pt')

    # Preprocess the input image
    input = ""
    start = 0
    load_time = 0
    inference_time = 0
    total = 0
    if(mode == 1):#start with model on cpu, data on cpu
        start = timer()
        model.to('cuda')
        input_batch = input_batch.to('cuda')
    if(mode == 2):#start with model on cpu, data on gpu
        input_batch = input_batch.to('cuda')
        start = timer()
        model.to('cuda')
    elif(mode == 3):#start with model on gpu, data on cpu
        model.to('cuda')
        start = timer()
        input_batch = input_batch.to('cuda')
    elif(mode == 4):#start with model on gpu, data on gpu
        model.to('cuda')
        input_batch = input_batch.to('cuda')
        start = timer()
    load_time = timer() - start
    model.eval()
    # Run inference
    with torch.no_grad():
        generate_ids = model.generate(input_batch.input_ids, max_length=30)
        tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    #print(output.)
    inference_time = timer() - start - load_time
    total = timer() - start
    return [load_time, inference_time, total]


NR_ITERATIONS = 11
if __name__ == "__main__":
    times = []
    #warm-up
    for j in range(NR_ITERATIONS):
        run(1)

    #measurements
    for i in range(1, 5):
        times_mode_i = []
        for j in range(NR_ITERATIONS):
            times_mode_i.append(run(i))#list of [[load_time_0, inference_time_0, total_0], [load_time_1, inference_time_1, total_1], ...]
        times.append(times_mode_i)#list of mode, each mode is list of [[load_time_0, inference_time_0, total_0], [load_time_1, inference_time_1, total_1], ...]

    print("(model, data) on (CPU,CPU), 2 (CPU,GPU), 3 (GPU,CPU), 4 (GPU,GPU) no unified memory")

    print("-----")

    print("Average load (model & data) times: ")
    print([sum([x[0] for x in m]) / NR_ITERATIONS for m in times])

    print("-----")

    print("Average inference time: ")
    print([sum([x[1] for x in m]) / NR_ITERATIONS for m in times])

    print("-----")

    print("Average total time: ")
    print([sum([x[2] for x in m]) / NR_ITERATIONS for m in times])


