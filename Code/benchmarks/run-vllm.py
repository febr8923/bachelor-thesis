from vllm import LLM, SamplingParams
import argparse
from timeit import default_timer as timer
import pandas as pd
import os


def run_vllm(llm):
    
    prompt = "some kind of text"
    sampling_params = SamplingParams(max_tokens=1, temperature=0)
    start = timer()
    outputs = llm.generate([prompt], sampling_params)
    end = timer()
    return end-start

NR_ITERATIONS = 5
NR_WARMUP_ITERATIONS = 5
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Hugginface model name")
    parser.add_argument("--cold_start", action="store_true", help="Enable cold start")
    parser.add_argument("--model_location", type=str, help="cpu/gpu")
    parser.add_argument("--execution_location", type=str, help="cpu/gpu")
    parser.add_argument("--thread_percentage", type=int, help="for file naming")
    args = parser.parse_args()

    model_name = args.model
    is_cold_start = args.cold_start
    model_loc = args.model_location
    execution_loc = args.execution_location
    thread_percentage = args.thread_percentage

    llm = LLM(model_name)
    times = []
    if not is_cold_start:
        for i in range(NR_WARMUP_ITERATIONS):
            run_vllm(llm)

        for i in range(NR_ITERATIONS):
            res = run_vllm(llm)
            times.append(res)
    

    n = len(times)


    df = pd.DataFrame([
        sum(times) / n, max(times), min(times)
    ])

    #csv_df_string = df.to_csv(index=False)
    #print(csv_df_string)
    dir_path = f"results/{model_name}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_path = f"{dir_path}/{model_loc}-{execution_loc}-{is_cold_start}.csv"
    df.to_csv(file_path, mode='a', header=False, index=False)

