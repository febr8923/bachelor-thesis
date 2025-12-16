

#Llama3 8B, GPT-J 6B, Falcon 7B, Baichuan2 7B, and Qwen 7B
#beam search (find parameter to disable random sampling)
import torch

import os
#from PIL import Image
#from torchvision import transforms
from timeit import default_timer as timer
import sys
import itertools
import pandas as pd

from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM

#mode can be 1 (CPU,CPU), 2 (CPU,GPU), 3 (GPU,CPU), 4 (GPU,GPU) for (model location, execution location)
#model should be on gpu/cpu, execution should be on gpu/cpu 

os.environ['TRANSFORMERS_CACHE'] = '/iopsstor/scratch/cscs/fbrunne'
model = None

NR_ITERATIONS = 5
NR_TOKENS = 10
import torch

def move_model_and_hidden_tensors(model, device):
    """
    Recursively moves a model and all its tensors, including non-registered
    cached tensors, to the specified device.

    Args:
        model (torch.nn.Module): The model to move.
        device (torch.device or str): The target device.
    """
    # 1. First, use the default .to() method to move all registered parameters and buffers.
    model.to(device)

    # 2. Create a set of IDs for all parameters and buffers to avoid moving them again.
    param_ids = {id(p) for p in model.parameters()}
    buffer_ids = {id(b) for b in model.buffers()}

    # 3. Recursively traverse all submodules.
    for module in model.modules():
        # 4. For each module, iterate over its attributes.
        for name, value in module.__dict__.items():
            # 5. Check if the attribute is a tensor.
            if isinstance(value, torch.Tensor):
                # 6. Check if it's NOT a parameter or buffer (we already moved them).
                if id(value) not in param_ids and id(value) not in buffer_ids:
                    # 7. This is a hidden/cached tensor. Move it and overwrite the attribute.
                    try:
                        # Overwrite the attribute with the tensor on the new device
                        setattr(module, name, value.to(device))
                    except Exception as e:
                        print(f"Could not move attribute '{name}' from module {module.__class__.__name__}: {e}")





def run_llm(model, tokenizer, model_loc: str, exec_loc: str):
    print(f"model_loc: {model_loc}, exec_loc: {exec_loc}")

    # memory: {torch.cuda.memory_allocated() / (1024**2):.2f}MB")
    #print(f"Reserved memory: {torch.cuda.memory_reserved() / (1024**2):.2f}MB")
    text = "some kind of text"

    input_batch = tokenizer(text, return_tensors='pt')
    

    # Move model to model_loc first
    torch.cuda.empty_cache()  # Clear cache before moving
    if(model_loc == "cpu"):
        #model.to('cpu')
        move_model_and_hidden_tensors(model, 'cpu')
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        input_batch = {k: v.to('cpu') for k, v in input_batch.items()}
    elif(model_loc == "gpu"):
        #model.to('cuda')
        move_model_and_hidden_tensors(model, 'cuda')
        torch.cuda.synchronize()
        input_batch = {k: v.to('cuda') for k, v in input_batch.items()}
    else:
        raise ValueError(f"Wrong model_loc, found {model_loc}, should be 'cpu' or 'gpu'")

    start = timer()

    # Then move model to exec_loc and measure the time
    if(exec_loc == "cpu"):
        #model.to('cpu')
        move_model_and_hidden_tensors(model, 'cpu')
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        input_batch = {k: v.to('cpu') for k, v in input_batch.items()}

    elif(exec_loc == "gpu"):
        #model.to("cuda")
        move_model_and_hidden_tensors(model, 'cuda')
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



if __name__ == "__main__":
    #torch.backends.cudnn.benchmark = False

    df = pd.DataFrame(columns=['gpu_cpu', 'cpu_gpu', 'gpu_gpu', 'cpu_cpu'], index=range(9))

    model_name = sys.argv[1]
    model_type = sys.argv[2]

    # Load model and tokenizer once at the beginning
    if model_type == "llm":
        print(f"Loading model {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            bf16=True,#only Qwen
            fp16=False,#only Qwen
            fp32=False#only Qwen
            #device_map="cpu" #device_map pins model to cpu (prevents .to() from working?)
        )

        # very important notice!!! I changed the Qwen model modeling_qwen.py", line 1345, in apply_rotary_pos_emb to move tensor to device of t!! 
        # this will obviously also increase the measured execution times when execution and placement location are different.
        # otherwise it breaks; might need to monkey-patch for more general use
        # might happen with other models as well! -> not sure what to do

        model.to('cpu')
        print("Model loaded successfully")
    else:
        raise ValueError("Invalid model type")

    #warm-up
    for j in range(2):
        if model_type == "llm":
            run_llm(model, tokenizer, model_loc="cpu", exec_loc="cpu")
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
                res = run_llm(model, tokenizer, model_loc=mode[0], exec_loc=mode[1])
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
    df.to_csv(f"results/results-{model_name}-patch.csv", mode='a', header=True, index=True)
