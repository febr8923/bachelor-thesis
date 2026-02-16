#!/usr/bin/env python3
"""
Plot Rodinia benchmark results as stacked bar charts.

For each benchmark, produces two subplots:
  - Left:  CPU thread sweep (modes: cpu/cpu, gpu/cpu)
  - Right: GPU SM% sweep (modes: cpu/gpu, gpu/gpu)

Each bar is stacked: execution time (bottom) + load/transfer overhead (top).
Load time = total_time - computation_time.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.csv")

# Mode display names and colors
MODE_STYLE = {
    "cpu":    {"label": "cpu/cpu",  "exec_color": "#2196F3", "load_color": "#90CAF9"},
    "gpu-cpu":{"label": "gpu/cpu",  "exec_color": "#FF9800", "load_color": "#FFE0B2"},
    "gpu":    {"label": "cpu/gpu",  "exec_color": "#4CAF50", "load_color": "#C8E6C9"},
    "gpu-gpu":{"label": "gpu/gpu",  "exec_color": "#F44336", "load_color": "#FFCDD2"},
}

def plot_benchmark(df, benchmark, axes):
    """Plot CPU sweep (left) and GPU sweep (right) for one benchmark."""
    bdf = df[(df["benchmark"] == benchmark) & (df["is_coldstart"] == False)].copy()

    # Drop rows with no total_time
    bdf = bdf.dropna(subset=["avg_total_time"])

    # Aggregate duplicates (same mode + thread_percentage) by taking the mean
    group_cols = ["mode", "thread_percentage"]
    num_cols = ["avg_total_time", "avg_computation_time", "avg_transfer_time"]
    bdf = bdf.groupby(group_cols, as_index=False)[num_cols].mean()
    if bdf.empty:
        for ax in axes:
            ax.set_visible(False)
        return

    # Compute load time = total - computation (fallback to 0 if no computation time)
    bdf["exec_time"] = bdf["avg_computation_time"].fillna(bdf["avg_total_time"])
    bdf["load_time"] = (bdf["avg_total_time"] - bdf["exec_time"]).clip(lower=0)

    # --- CPU sweep (left subplot) ---
    cpu_modes = ["cpu", "gpu-cpu"]
    cpu_data = bdf[bdf["mode"].isin(cpu_modes)]
    ax = axes[0]

    if not cpu_data.empty:
        threads = sorted(cpu_data["thread_percentage"].unique())
        x = np.arange(len(threads))
        n_modes = len(cpu_modes)
        width = 0.8 / n_modes

        for idx, mode in enumerate(cpu_modes):
            mdata = cpu_data[cpu_data["mode"] == mode].set_index("thread_percentage")
            exec_vals = [mdata.loc[t, "exec_time"] if t in mdata.index else 0 for t in threads]
            load_vals = [mdata.loc[t, "load_time"] if t in mdata.index else 0 for t in threads]
            style = MODE_STYLE[mode]
            offset = x + (idx - n_modes / 2 + 0.5) * width
            ax.bar(offset, exec_vals, width, label=f"{style['label']} exec",
                   color=style["exec_color"], edgecolor="white", linewidth=0.5)
            ax.bar(offset, load_vals, width, bottom=exec_vals, label=f"{style['label']} load",
                   color=style["load_color"], edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in threads])
        ax.set_xlabel("CPU Threads")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"{benchmark} — CPU Thread Sweep")

    # --- GPU sweep (right subplot) ---
    gpu_modes = ["gpu", "gpu-gpu"]
    gpu_data = bdf[bdf["mode"].isin(gpu_modes)]
    ax = axes[1]

    if not gpu_data.empty:
        sm_pcts = sorted(gpu_data["thread_percentage"].unique())
        x = np.arange(len(sm_pcts))
        n_modes = len(gpu_modes)
        width = 0.8 / n_modes

        for idx, mode in enumerate(gpu_modes):
            mdata = gpu_data[gpu_data["mode"] == mode].set_index("thread_percentage")
            exec_vals = [mdata.loc[t, "exec_time"] if t in mdata.index else 0 for t in sm_pcts]
            load_vals = [mdata.loc[t, "load_time"] if t in mdata.index else 0 for t in sm_pcts]
            style = MODE_STYLE[mode]
            offset = x + (idx - n_modes / 2 + 0.5) * width
            ax.bar(offset, exec_vals, width, label=f"{style['label']} exec",
                   color=style["exec_color"], edgecolor="white", linewidth=0.5)
            ax.bar(offset, load_vals, width, bottom=exec_vals, label=f"{style['label']} load",
                   color=style["load_color"], edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{t}%" for t in sm_pcts])
        ax.set_xlabel("GPU SM Percentage")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"{benchmark} — GPU SM Sweep")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else CSV_PATH
    df = pd.read_csv(csv_path)

    benchmarks = [b for b in df["benchmark"].unique()
                  if not df[(df["benchmark"] == b) & df["avg_total_time"].notna()].empty]

    n = len(benchmarks)
    fig, all_axes = plt.subplots(n, 2, figsize=(14, 5 * n), squeeze=False)

    for row, bench in enumerate(benchmarks):
        plot_benchmark(df, bench, all_axes[row])
        # Add legends
        for ax in all_axes[row]:
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Rodinia Benchmarks — Execution + Load Time", fontsize=14, y=1.01)
    fig.tight_layout()

    out_path = os.path.join(os.path.dirname(csv_path), "benchmark_plots.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
