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

# Mode display names and colors (execution=red, transfer=blue; darker shade for second mode)
MODE_STYLE = {
    "cpu":    {"label": "cpu/cpu",  "exec_color": "#e74c3c", "load_color": "#3498db"},
    "gpu-cpu":{"label": "gpu/cpu",  "exec_color": "#c0392b", "load_color": "#2471a3"},
    "gpu":    {"label": "cpu/gpu",  "exec_color": "#e74c3c", "load_color": "#3498db"},
    "gpu-gpu":{"label": "gpu/gpu",  "exec_color": "#c0392b", "load_color": "#2471a3"},
}


def _get_vals(mdata, index_vals, col):
    return np.array([mdata.loc[t, col] if t in mdata.index else 0 for t in index_vals],
                    dtype=float)


def plot_benchmark(df, benchmark, axes):
    """Plot CPU sweep (left) and GPU sweep (right) for one benchmark."""
    bdf = df[(df["benchmark"] == benchmark) & (df["is_coldstart"] == False)].copy()

    # Aggregate duplicates (same mode + thread_percentage) by taking the mean
    group_cols = ["mode", "thread_percentage"]
    num_cols = [
        "avg_execution_time", "min_execution_time", "max_execution_time",
        "avg_transfer_time_h2d", "avg_transfer_time_d2h",
        "avg_total_time", "min_total_time", "max_total_time",
    ]
    bdf = bdf.groupby(group_cols, as_index=False)[num_cols].mean()
    if bdf.empty:
        for ax in axes:
            ax.set_visible(False)
        return

    # For cpu/cpu: avg_total_time is NaN — fall back to execution time
    bdf["total_avg"] = bdf["avg_total_time"].fillna(bdf["avg_execution_time"])
    bdf["total_min"] = bdf["min_total_time"].fillna(bdf["min_execution_time"])
    bdf["total_max"] = bdf["max_total_time"].fillna(bdf["max_execution_time"])
    bdf["transfer_time"] = (bdf["avg_transfer_time_h2d"].fillna(0)
                            + bdf["avg_transfer_time_d2h"].fillna(0))

    def _draw_sweep(ax, modes, index_vals, xlabel, xticklabels):
        data = bdf[bdf["mode"].isin(modes)]
        if data.empty:
            return
        x = np.arange(len(index_vals))
        width = 0.8 / len(modes)
        for idx, mode in enumerate(modes):
            mdata = data[data["mode"] == mode].set_index("thread_percentage")
            exec_v  = _get_vals(mdata, index_vals, "avg_execution_time")
            trans_v = _get_vals(mdata, index_vals, "transfer_time")
            t_avg   = _get_vals(mdata, index_vals, "total_avg")
            t_min   = _get_vals(mdata, index_vals, "total_min")
            t_max   = _get_vals(mdata, index_vals, "total_max")
            err = np.clip(np.array([t_avg - t_min, t_max - t_avg]), 0, None)
            style = MODE_STYLE[mode]
            offset = x + (idx - len(modes) / 2 + 0.5) * width
            ax.bar(offset, exec_v, width, label=f"{style['label']} exec",
                   color=style["exec_color"], edgecolor="white", linewidth=0.5)
            ax.bar(offset, trans_v, width, bottom=exec_v,
                   label=f"{style['label']} transfer",
                   color=style["load_color"], edgecolor="white", linewidth=0.5,
                   yerr=err, capsize=3, error_kw={"elinewidth": 1})
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel(xlabel)

    # --- CPU sweep (left subplot) ---
    cpu_threads = sorted(bdf[bdf["mode"].isin(["cpu", "gpu-cpu"])]["thread_percentage"].unique())
    _draw_sweep(axes[0], ["cpu", "gpu-cpu"], cpu_threads,
                "CPU Threads", [str(int(t)) for t in cpu_threads])
    axes[0].set_ylabel("Time (s)")
    axes[0].set_title(f"{benchmark} — CPU Thread Sweep")

    # --- GPU sweep (right subplot) ---
    sm_pcts = sorted(bdf[bdf["mode"].isin(["gpu", "gpu-gpu"])]["thread_percentage"].unique())
    _draw_sweep(axes[1], ["gpu", "gpu-gpu"], sm_pcts,
                "GPU SM Percentage", [f"{int(t)}%" for t in sm_pcts])
    axes[1].set_ylabel("Time (s)")
    axes[1].set_title(f"{benchmark} — GPU SM Sweep")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_benchmarks.py <data_file.csv> [output_dir]")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) >= 3 else os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    csv_stem = os.path.basename(csv_path).replace(".csv", "")

    benchmarks = [b for b in df["benchmark"].unique()
                  if not df[(df["benchmark"] == b) & df["avg_execution_time"].notna()].empty]

    for bench in benchmarks:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        plot_benchmark(df, bench, axes)
        for ax in axes:
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle(f"Rodinia — {bench}", fontsize=14, fontweight="bold")
        fig.tight_layout()

        out_path = os.path.join(output_dir, f"{csv_stem}_{bench}.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved as '{out_path}'")
        plt.close(fig)


if __name__ == "__main__":
    main()
