#!/usr/bin/env python3
"""
Plot Rodinia benchmark results as stacked bar charts with error bars.

For each benchmark, produces two subplots:
  - Left:  CPU thread sweep (modes: cpu/cpu, gpu/cpu)
  - Right: GPU SM% sweep (modes: cpu/gpu, gpu/gpu)

Each bar is stacked: execution time (bottom) + transfer overhead (top).
Error bars show min–max range from benchmark runs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.csv")

# Mode display names and colors
MODE_STYLE = {
    "cpu":    {"label": "cpu/cpu",  "exec_color": "#2196F3", "transfer_color": "#90CAF9"},
    "gpu-cpu":{"label": "gpu/cpu",  "exec_color": "#FF9800", "transfer_color": "#FFE0B2"},
    "gpu":    {"label": "cpu/gpu",  "exec_color": "#4CAF50", "transfer_color": "#C8E6C9"},
    "gpu-gpu":{"label": "gpu/gpu",  "exec_color": "#F44336", "transfer_color": "#FFCDD2"},
}


def safe_val(series, index, col, default=0):
    """Safely get a value from a DataFrame indexed by thread_percentage."""
    if index in series.index:
        v = series.loc[index, col]
        return v if pd.notna(v) else default
    return default


def plot_benchmark(df, benchmark, axes):
    """Plot CPU sweep (left) and GPU sweep (right) for one benchmark."""
    bdf = df[(df["benchmark"] == benchmark) & (df["is_coldstart"] == False)].copy()

    if bdf.empty:
        for ax in axes:
            ax.set_visible(False)
        return

    # For modes without total_time, use execution_time as total
    bdf["plot_total"] = bdf["avg_total_time"].fillna(bdf["avg_execution_time"])

    # Transfer overhead = h2d + d2h (where available)
    bdf["transfer_overhead"] = bdf["avg_transfer_time_h2d"].fillna(0) + bdf["avg_transfer_time_d2h"].fillna(0)

    # Execution error bars (min/max range around avg)
    bdf["exec_err_lo"] = (bdf["avg_execution_time"] - bdf["min_execution_time"]).fillna(0).clip(lower=0)
    bdf["exec_err_hi"] = (bdf["max_execution_time"] - bdf["avg_execution_time"]).fillna(0).clip(lower=0)

    # Aggregate duplicates (same mode + thread_percentage) by taking the mean
    group_cols = ["mode", "thread_percentage"]
    num_cols = [c for c in bdf.columns if c not in group_cols and bdf[c].dtype in ['float64', 'int64', 'float32']]
    bdf = bdf.groupby(group_cols, as_index=False)[num_cols].mean()

    # --- CPU sweep (left subplot) ---
    cpu_modes = ["cpu", "gpu-cpu"]
    cpu_data = bdf[bdf["mode"].isin(cpu_modes)]
    ax = axes[0]

    if not cpu_data.empty:
        threads = sorted(cpu_data["thread_percentage"].unique())
        x = np.arange(len(threads))
        n_modes = sum(1 for m in cpu_modes if not cpu_data[cpu_data["mode"] == m].empty)
        width = 0.8 / max(n_modes, 1)

        plot_idx = 0
        for mode in cpu_modes:
            mdata = cpu_data[cpu_data["mode"] == mode].set_index("thread_percentage")
            if mdata.empty:
                continue
            exec_vals = [safe_val(mdata, t, "avg_execution_time") for t in threads]
            transfer_vals = [safe_val(mdata, t, "transfer_overhead") for t in threads]
            err_lo = [safe_val(mdata, t, "exec_err_lo") for t in threads]
            err_hi = [safe_val(mdata, t, "exec_err_hi") for t in threads]
            style = MODE_STYLE[mode]
            offset = x + (plot_idx - n_modes / 2 + 0.5) * width

            ax.bar(offset, exec_vals, width, label=f"{style['label']} exec",
                   color=style["exec_color"], edgecolor="white", linewidth=0.5,
                   yerr=[err_lo, err_hi], error_kw={"capsize": 2, "capthick": 0.8, "elinewidth": 0.8, "ecolor": "black"})
            ax.bar(offset, transfer_vals, width, bottom=exec_vals,
                   label=f"{style['label']} transfer",
                   color=style["transfer_color"], edgecolor="white", linewidth=0.5)
            plot_idx += 1

        ax.set_xticks(x)
        ax.set_xticklabels([str(int(t)) for t in threads])
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
        n_modes = sum(1 for m in gpu_modes if not gpu_data[gpu_data["mode"] == m].empty)
        width = 0.8 / max(n_modes, 1)

        plot_idx = 0
        for mode in gpu_modes:
            mdata = gpu_data[gpu_data["mode"] == mode].set_index("thread_percentage")
            if mdata.empty:
                continue
            exec_vals = [safe_val(mdata, t, "avg_execution_time") for t in sm_pcts]
            transfer_vals = [safe_val(mdata, t, "transfer_overhead") for t in sm_pcts]
            err_lo = [safe_val(mdata, t, "exec_err_lo") for t in sm_pcts]
            err_hi = [safe_val(mdata, t, "exec_err_hi") for t in sm_pcts]
            style = MODE_STYLE[mode]
            offset = x + (plot_idx - n_modes / 2 + 0.5) * width

            ax.bar(offset, exec_vals, width, label=f"{style['label']} exec",
                   color=style["exec_color"], edgecolor="white", linewidth=0.5,
                   yerr=[err_lo, err_hi], error_kw={"capsize": 2, "capthick": 0.8, "elinewidth": 0.8, "ecolor": "black"})
            ax.bar(offset, transfer_vals, width, bottom=exec_vals,
                   label=f"{style['label']} transfer",
                   color=style["transfer_color"], edgecolor="white", linewidth=0.5)
            plot_idx += 1

        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(t)}%" for t in sm_pcts])
        ax.set_xlabel("GPU SM Percentage")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"{benchmark} — GPU SM Sweep")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else CSV_PATH
    df = pd.read_csv(csv_path)

    # Use benchmarks that have execution time data
    benchmarks = [b for b in df["benchmark"].unique()
                  if not df[(df["benchmark"] == b) & df["avg_execution_time"].notna()].empty]

    n = len(benchmarks)
    if n == 0:
        print("No benchmark data with execution times found.")
        return

    fig, all_axes = plt.subplots(n, 2, figsize=(14, 5 * n), squeeze=False)

    for row, bench in enumerate(benchmarks):
        plot_benchmark(df, bench, all_axes[row])
        for ax in all_axes[row]:
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Rodinia Benchmarks — Execution + Transfer Time", fontsize=14, y=1.01)
    fig.tight_layout()

    out_path = os.path.join(os.path.dirname(csv_path), "benchmark_plots.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
