import argparse
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import os


def line_graph(lines, y, labels, title):
    plt.clf()
    for i, line in enumerate(lines):
        plt.plot(y, line, label=labels[i])

    plt.title(title)
    plt.legend()
    plt.savefig(f"{title}.png")
    plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="huggingface model name")
    args = parser.parse_args()
    model_name = args.model

    prefix = f"results/{model_name}"
    if not os.path.exists(f"{prefix}/plots"):
        os.makedirs(f"{prefix}/plots")

    
    df_cpu_cpu_t = pd.read_csv(f"{prefix}/cpu-cpu-True.csv")
    df_gpu_cpu_t = pd.read_csv(f"{prefix}/gpu-cpu-True.csv")
    df_cpu_cpu_f = pd.read_csv(f"{prefix}/cpu-cpu-False.csv")
    df_gpu_cpu_f = pd.read_csv(f"{prefix}/gpu-cpu-False.csv")

    df_cpu_gpu_t = pd.read_csv(f"{prefix}/cpu-gpu-True.csv")
    df_gpu_gpu_t = pd.read_csv(f"{prefix}/gpu-gpu-True.csv")
    df_cpu_gpu_f = pd.read_csv(f"{prefix}/cpu-gpu-False.csv")
    df_gpu_gpu_f = pd.read_csv(f"{prefix}/gpu-gpu-False.csv")

    print(df_cpu_cpu_t.head())

    ##load 
    #cpu
    l1 = df_cpu_cpu_f.iloc[0::9].iloc[:,0].tolist()
    l2 = df_gpu_cpu_f.iloc[0::9].iloc[:,0].tolist()
    line_graph([l1,l2], [1, 2, 4, 8, 16, 36, 72], ["cpu, cpu", "gpu, cpu"], f"{prefix}/plots/load-cpu")
    #gpu
    l3 = df_cpu_gpu_f.iloc[0::9].iloc[:,0].tolist()
    l4 = df_gpu_gpu_f.iloc[0::9].iloc[:,0].tolist()
    line_graph([l3,l4], range(10,110,10), ["cpu, gpu", "gpu, gpu"], f"{prefix}/plots/load-gpu")
    
    ##execute
    l1 = df_cpu_cpu_f.iloc[3::9].iloc[:,0].tolist()
    l2 = df_gpu_cpu_f.iloc[3::9].iloc[:,0].tolist()
    line_graph([l1,l2], [1, 2, 4, 8, 16, 36, 72], ["cpu, cpu", "gpu, cpu"], f"{prefix}/plots/execute-cpu")
    l3 = df_cpu_gpu_f.iloc[3::9].iloc[:,0].tolist()
    l4 = df_gpu_gpu_f.iloc[3::9].iloc[:,0].tolist()
    line_graph([l3,l4], range(10,110,10), ["cpu, gpu", "gpu, gpu"], f"{prefix}/plots/execute-gpu")

    ##total
    l1 = df_cpu_cpu_f.iloc[6::9].iloc[:,0].tolist()
    l2 = df_gpu_cpu_f.iloc[6::9].iloc[:,0].tolist()
    line_graph([l1,l2], [1, 2, 4, 8, 16, 36, 72], ["cpu, cpu", "gpu, cpu"], f"{prefix}/plots/total-cpu")
    l3 = df_cpu_gpu_f.iloc[6::9].iloc[:,0].tolist()
    l4 = df_gpu_gpu_f.iloc[6::9].iloc[:,0].tolist()
    line_graph([l3,l4], range(10,110,10), ["cpu, gpu", "gpu, gpu"], f"{prefix}/plots/total-gpu")

    
    
