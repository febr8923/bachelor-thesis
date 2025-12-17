import argparse
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import os



'''
Plots a line graph with 
- x values: lines, 
- y-axis values: y
- line labels: labels
- title: title
- at dir 
'''
def line_graph(lines, y, labels, title, dir):
    plt.clf()
    for i, line in enumerate(lines):
        plt.plot(y, line, label=labels[i])

    plt.title(title)
    plt.legend()
    plt.savefig(f"{dir}/plots/{title}.png")
    plt.clf()

'''
Returns a list of the times stored in a csv-file by model_loc, exec_loc and is_cold
'''
def times_by_thread(model_loc, exec_loc, is_cold, dir):
    if model_loc in ["cpu", "gpu"] and exec_loc in ["cpu", "gpu"]:
        df = pd.read_csv(f"{dir}/{model_loc}-{exec_loc}-{is_cold}.csv")
        return df.iloc[:,0].tolist()
    else:
        raise ValueError("Wrong location")


def plot_vllm(exec_loc, is_cold, dir):
    l = times_by_thread(model_loc=exec_loc, exec_loc=exec_loc, is_cold=is_cold, dir=dir)
    lines = [l[0::3]]

    if exec_loc == "cpu": 
        y = [1, 2, 4, 8, 16, 36, 72]
    elif exec_loc == "gpu":
        y = range(10,110,10)
    else:
        raise ValueError("Wrong location")
    
    line_graph(lines=lines, y=y,labels=[f"{exec_loc}-{exec_loc}"], title=f"time-{exec_loc}-{exec_loc}", dir=dir)
    
'''
Plots a line where y-axis is nr./percentage of threads and x-axis is execution time
'''
def plot_line_graph_by(info_type, model_loc, exec_loc, is_cold, dir):
    l = times_by_thread(model_loc=model_loc, exec_loc=exec_loc, is_cold=is_cold, dir=dir)

    if exec_loc == "cpu": 
        y = [1, 2, 4, 8, 16, 36, 72]
    elif exec_loc == "gpu":
        y = range(10,110,10)
    else:
        raise ValueError("Wrong location")
    
    if info_type == "load":
        lines = [l[0::9]]
    elif info_type == "execution":
        lines = [ l[3::9]]
    elif info_type == "total":
        lines = [ l[6::9]]
    else:
        raise ValueError("info_type must be load, execution, or total")
    
    line_graph(lines=lines, y=y,labels=[f"{model_loc}-{exec_loc}"], title=f"{info_type}-{model_loc}-{exec_loc}", dir=dir)
    
'''
Plots a lines where y-axis is nr./percentage of threads and x-axis is execution time
for a particular execution location and model location in [cpu, gpu]
'''
def plot_line_graph_by_exec_loc(info_type, exec_loc, is_cold, dir):

    if exec_loc == "cpu": 
        y = [1, 2, 4, 8, 16, 36, 72]
    elif exec_loc == "gpu":
        y = range(10,110,10)
    else:
        raise ValueError("Wrong location")

    l1 = times_by_thread(model_loc="cpu", exec_loc=exec_loc, is_cold=is_cold, dir=dir)
    l2 = times_by_thread(model_loc="gpu", exec_loc=exec_loc, is_cold=is_cold, dir=dir)

    lines = []
    if info_type == "load":
        lines = [l1[0::9], l2[0::9]]
    elif info_type == "execution":
        lines = [l1[3::9], l2[3::9]]
    elif info_type == "total":
        lines = [l1[3::9], l2[3::9]]
    else:
        raise ValueError("info_type must be load, execution, or all")
    
    line_graph(lines=lines, y=y,labels=[f"cpu-{exec_loc}", f"gpu-{exec_loc}"], title=f"{info_type}-{exec_loc}", dir=dir)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="huggingface model name")
    parser.add_argument("--cpu_only", action="store_true", help="Enable cold start")
    parser.add_argument("--gpu_only", action="store_true", help="Enable cold start")
    parser.add_argument("--vllm", action="store_true", help="Enable cold start")
    args = parser.parse_args()
    model_name = args.model
    cpu_only = args.cpu_only
    gpu_only = args.gpu_only
    vllm = args.vllm

    prefix = f"results/{model_name}"
    if not os.path.exists(f"{prefix}/plots"):
        os.makedirs(f"{prefix}/plots")

    
    if cpu_only:
        if vllm:
            plot_vllm(exec_loc="cpu",is_cold=False,dir=prefix)
    elif gpu_only:
        if vllm:
            plot_vllm(exec_loc="gpu",is_cold=False,dir=prefix)

    else:
        plot_line_graph_by_exec_loc(info_type="load", exec_loc="cpu", is_cold=False, dir=prefix)
        plot_line_graph_by_exec_loc(info_type="load", exec_loc="gpu", is_cold=False, dir=prefix)

        plot_line_graph_by_exec_loc(info_type="execute", exec_loc="cpu", is_cold=False, dir=prefix)
        plot_line_graph_by_exec_loc(info_type="execute", exec_loc="gpu", is_cold=False, dir=prefix)

        plot_line_graph_by_exec_loc(info_type="total", exec_loc="cpu", is_cold=False, dir=prefix)
        plot_line_graph_by_exec_loc(info_type="total", exec_loc="gpu", is_cold=False, dir=prefix)

    

    
