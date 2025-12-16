

#Llama3 8B, GPT-J 6B, Falcon 7B, Baichuan2 7B, and Qwen 7B
#beam search (find parameter to disable random sampling)


import torch

use_managed_mem = True

import rmm
from rmm.allocators.torch import rmm_torch_allocator
rmm.reinitialize(pool_allocator=True, managed_memory=use_managed_mem)
torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

import os
from PIL import Image
from torchvision import transforms
from timeit import default_timer as timer
import sys
import itertools
import pandas as pd

from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM

#mode can be 1 (CPU,CPU), 2 (CPU,GPU), 3 (GPU,CPU), 4 (GPU,GPU) for (model location, execution location)
#model should be on gpu/cpu, execution should be on gpu/cpu 

os.environ['TRANSFORMERS_CACHE'] = '/iopsstor/scratch/cscs/fbrunne'
model = None

def run_other(model_loc: str, exec_loc: str, model_name: str):
    print(f"model_loc: {model_loc}, exec_loc: {exec_loc}, Name: {model_name}")
    times = []
    global model
   
    if(model_type == "llm"):
        
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="cpu")

        text = "some kind of text"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
    
        input_batch = tokenizer(text, return_tensors='pt')
    
    elif(model_type == "image"):
        if model is None:
            model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)

        # Preprocess the input image
        input_image = Image.open('/users/fbrunne/projects/benchmarks/dog.jpg')
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    elif(model_type == "other"):
        if model is None:
            model = AutoModel.from_pretrained(model_name)

        text = "some kind of text"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        input_batch = tokenizer(text, return_tensors='pt')
    else:
        print("Error, wrong model type")
        return


    #torch.save(model, model_name)
    #model.to('cpu')

    #print(f"Allocated memory: {torch.cuda.memory_allocated() / (1024**2):.2f}MB")
    #print(f"Reserved memory: {torch.cuda.memory_reserved() / (1024**2):.2f}MB")
    print("run")

    if(model_loc == "cpu"):#start with model on cpu execution on cpu
        #model.to('cpu')
        #input_batch = input_batch.to('cpu')
        input_batch = {k: v.to('cpu') for k, v in input_batch.items()}

    elif(model_loc == "gpu"):
        #model.to('cuda')
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="cuda")
        #input_batch = input_batch.to('cuda')
        input_batch = {k: v.to('cuda') for k, v in input_batch.items()}
    else:
         raise ValueError(f"Wrong model_loc, found {model_loc}, should be 'cpu' or 'gpu'")

    start = timer()

    if(exec_loc == "cpu"):
        #model.to('cpu')
        #input_batch = input_batch.to('cpu')
        pass
    elif(exec_loc == "gpu"):
        #model.to("cuda")
        #input_batch = input_batch.to("cuda")
        pass
    else:
         raise ValueError(f"Wrong exec_loc, found {exec_loc}, should be 'cpu' or 'gpu'")
    #input_batch = {k: v.to(next(model.parameters()).device) for k, v in input_batch.items()}
    
    #print(f"Allocated memory: {torch.cuda.memory_allocated() / (1024**2):.2f}MB")
    #print(f"Reserved memory: {torch.cuda.memory_reserved() / (1024**2):.2f}MB")
    print("--------")
        
    torch.cuda.synchronize()
    load_time = timer() - start

    model.eval()
    with torch.no_grad():
        if model_type == "llm":
            output = model.generate(**input_batch, max_length=1)
        elif model_type == "image":
            output = model(input_batch)
        else: 
            output = model(**input_batch)

    torch.cuda.synchronize()
    inference_time = timer() - (load_time + start)

    #clean-up
    #del model
    torch.cuda.empty_cache()

    total = inference_time + load_time
    return [load_time, inference_time, total]

def run_image(model_loc: str, exec_loc: str, model_name: str):
    global model
    if model is None:
        model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)

    # Preprocess the input image
    input_image = Image.open('/users/fbrunne/projects/benchmarks/dog.jpg')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    
    #print(f"Allocated memory: {torch.cuda.memory_allocated() / (1024**2):.2f}MB")
    #print(f"Reserved memory: {torch.cuda.memory_reserved() / (1024**2):.2f}MB")

    if(model_loc == "cpu"):#start with model on cpu execution on cpu
        model.to('cpu')
        input_batch = input_batch.to('cpu')
    elif(model_loc == "gpu"):
        model.to('cuda')
        input_batch = input_batch.to('cuda')
    else:
        raise ValueError(f"Wrong model_loc, found {model_loc}, should be 'cpu' or 'gpu'")

    start = timer()

    if(exec_loc == "cpu"):
        model.to('cpu')
        input_batch = input_batch.to('cpu')
    elif(exec_loc == "gpu"):
        model.to("cuda")
        input_batch = input_batch.to("cuda")
    else:
        raise ValueError(f"Wrong exec_loc, found {exec_loc}, should be 'cpu' or 'gpu'")
    
    #print(f"Allocated memory: {torch.cuda.memory_allocated() / (1024**2):.2f}MB")
    #print(f"Reserved memory: {torch.cuda.memory_reserved() / (1024**2):.2f}MB")
    print("--------")
        
    torch.cuda.synchronize()
    load_time = timer() - start

    model.eval()
    with torch.no_grad():
        output = model(input_batch)

    torch.cuda.synchronize()
    inference_time = timer() - (load_time + start)

    torch.cuda.empty_cache()

    total = inference_time + load_time
    return [load_time, inference_time, total]

def run_llm(model, tokenizer, model_loc: str, exec_loc: str):
    print(f"model_loc: {model_loc}, exec_loc: {exec_loc}")

    # memory: {torch.cuda.memory_allocated() / (1024**2):.2f}MB")
    #print(f"Reserved memory: {torch.cuda.memory_reserved() / (1024**2):.2f}MB")
    text = "some kind of text"

    input_batch = tokenizer(text, return_tensors='pt')
    

    # Move model to model_loc first
    torch.cuda.empty_cache()  # Clear cache before moving
    if(model_loc == "cpu"):
        model.to('cpu')
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        input_batch = {k: v.to('cpu') for k, v in input_batch.items()}
    elif(model_loc == "gpu"):
        model.to('cuda')
        torch.cuda.synchronize()
        input_batch = {k: v.to('cuda') for k, v in input_batch.items()}
    else:
        raise ValueError(f"Wrong model_loc, found {model_loc}, should be 'cpu' or 'gpu'")

    start = timer()

    # Then move model to exec_loc and measure the time
    if(exec_loc == "cpu"):
        model.to('cpu')
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        input_batch = {k: v.to('cpu') for k, v in input_batch.items()}
    elif(exec_loc == "gpu"):
        model.to("cuda")
        torch.cuda.synchronize()
        input_batch = {k: v.to('cuda') for k, v in input_batch.items()}
    else:
        raise ValueError(f"Wrong exec_loc, found {exec_loc}, should be 'cpu' or 'gpu'")
    
    #print(f"Allocated memory: {torch.cuda.memory_allocated() / (1024**2):.2f}MB")
    #print(f"Reserved memory: {torch.cuda.memory_reserved() / (1024**2):.2f}MB")
    print("--------")
        
    torch.cuda.synchronize()
    load_time = timer() - start

    model.eval()
    with torch.no_grad():
        output = model.generate(**input_batch,  max_new_tokens=1, min_new_tokens=1, do_sample=False)

    torch.cuda.synchronize()
    inference_time = timer() - (load_time + start)

    #clean-up
    #del model
    torch.cuda.empty_cache()

    total = inference_time + load_time
    return [load_time, inference_time, total]



NR_ITERATIONS = 5
if __name__ == "__main__":
    #torch.backends.cudnn.benchmark = False                                                                                                                                                                                                                                                          

    df = pd.DataFrame(columns=['gpu_cpu', 'cpu_gpu', 'gpu_gpu', 'cpu_cpu'], index=range(9))

    model_name = sys.argv[1]
    model_type = sys.argv[2]
    if model_type == "llm":
        print(f"Loading model {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            #bf16=False,#only Qwen
            #fp16=False,#
            #fp32=True#
            #device_map="cpu" #device_map pins model to cpu (prevents .to() from working?)
        )
        model.to('cpu')
        print("Model loaded successfully")
    #warm-up
    for j in range(2):
        if model_type == "llm":
            run_llm(model=model, tokenizer=tokenizer, model_loc="gpu", exec_loc="gpu")
        elif model_type == "image":
            run_image(model_loc="gpu", exec_loc="gpu", model_name=model_name)
        elif model_type == "other":
            run_other(model_loc="gpu", exec_loc="gpu", model_name=model_name)
        else:
            raise ValueError("Invalid model type")
    
    print("finished warm-up")
    torch.cuda.empty_cache()

    #measurements
    for mode in itertools.product(["cpu", "gpu"], repeat=2):
        times_mode_i_load = []
        times_mode_i_inference = []
        times_mode_i_total = []
        
        for i in range(NR_ITERATIONS):
            res = []
            if model_type == "llm":
                res = run_llm(model=model, tokenizer=tokenizer, model_loc=mode[0], exec_loc=mode[1])
            elif model_type == "image":
                res = run_image(model_loc=mode[0], exec_loc=mode[1], model_name=model_name)
            elif model_type == "other":
                res = run_other(model_loc=mode[0], exec_loc=mode[1], model_name=model_name)
            else:
                raise ValueError("Invalid model type")
            
            times_mode_i_load.append(res[0])
            times_mode_i_inference.append(res[1])
            times_mode_i_total.append(res[2])
            torch.cuda.empty_cache()

        col_name = f"{mode[0]}_{mode[1]}"

        n = len(times_mode_i_load)
 
        df.at[0, col_name] = sum(times_mode_i_load) / n
        df.at[1, col_name] = max(times_mode_i_load)
        df.at[2, col_name] = min(times_mode_i_load)

        df.at[3, col_name] = sum(times_mode_i_inference) / n
        df.at[4, col_name] = max(times_mode_i_inference)
        df.at[5, col_name] = min(times_mode_i_inference)

        df.at[6, col_name] = sum(times_mode_i_total) / n
        df.at[7, col_name] = max(times_mode_i_total)
        df.at[8, col_name] = min(times_mode_i_total)

    #csv_df_string = df.to_csv(index=False)
    #print(csv_df_string)
    df.to_csv(f"results/results-{model_name}-{use_managed_mem}.csv", mode='a', header=True, index=True)
