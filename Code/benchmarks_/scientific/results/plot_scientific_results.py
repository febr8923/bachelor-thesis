#!/usr/bin/env python3
"""
Plot scientific benchmark results as stacked bar charts (computation + data transfer time).

Handles sweep CSVs (thread count or SM% sweep) and single-config CSVs.
Infers the sweep variable from whichever column varies in the data.

Usage:
    python plot_scientific_results.py <data_file.csv> [output_dir] [benchmark]

    data_file.csv  — results CSV produced by run-scientific.py
    output_dir     — directory for output PNGs (default: same dir as CSV)
    benchmark      — only plot this benchmark (default: all benchmarks in file)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os


def detect_sweep_var(df):
    """Return 'num_threads', 'sm_percentage', or None (config comparison)."""
    if df['num_threads'].nunique() > 1:
        return 'num_threads'
    if df['sm_percentage'].nunique() > 1:
        return 'sm_percentage'
    return None


def make_labels(sweep_var, values):
    if sweep_var == 'sm_percentage':
        return [f'{int(v)}%' for v in values]
    return [str(int(v)) for v in values]


def plot_benchmark(df_bench, benchmark_name, sweep_var, output_dir, csv_stem):
    if sweep_var:
        df_bench = df_bench.sort_values(sweep_var)
        labels = make_labels(sweep_var, df_bench[sweep_var].values)
        xlabel = 'GPU SM Utilization (%)' if sweep_var == 'sm_percentage' else 'Number of CPU Threads'
    else:
        df_bench = df_bench.copy()
        df_bench['_config'] = df_bench['data_loc'] + '-' + df_bench['exec_loc']
        df_bench['_config'] = pd.Categorical(
            df_bench['_config'],
            categories=['cpu-cpu', 'cpu-gpu', 'gpu-cpu'],
            ordered=True
        )
        df_bench = df_bench.sort_values('_config')
        labels = df_bench['_config'].values.tolist()
        xlabel = 'Configuration (data_loc-exec_loc)'

    compute = df_bench['avg_computation_ms'].values
    transfer = df_bench['avg_transfer_ms'].values
    total_avg = df_bench['avg_total_ms'].values

    x = np.arange(len(labels))
    width = 0.6

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x, transfer, width, label='Data Transfer', color='#3498db')

    # Error bars on total (placed at top of stacked bar)
    if 'min_total_ms' in df_bench.columns and 'max_total_ms' in df_bench.columns:
        total_min = df_bench['min_total_ms'].values
        total_max = df_bench['max_total_ms'].values
        err = np.clip(
            np.array([total_avg - total_min, total_max - total_avg]),
            0, None
        )
        ax.bar(x, compute, width, bottom=transfer, label='Computation', color='#e74c3c',
               yerr=err, capsize=3, error_kw={'elinewidth': 1})
    else:
        ax.bar(x, compute, width, bottom=transfer, label='Computation', color='#e74c3c')

    ax.set_title(f'{benchmark_name} — {csv_stem}', fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45 if sweep_var == 'sm_percentage' else 0, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'{csv_stem}_{benchmark_name}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{out_path}'")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if len(sys.argv) < 2:
    print("Usage: python plot_scientific_results.py <data_file.csv> [output_dir] [benchmark]")
    sys.exit(1)

DATA_FILE = sys.argv[1]
OUTPUT_DIR = sys.argv[2] if len(sys.argv) >= 3 else os.path.dirname(os.path.abspath(DATA_FILE))
FILTER_BENCH = sys.argv[3] if len(sys.argv) >= 4 else None

csv_stem = os.path.basename(DATA_FILE).replace('.csv', '')

df = pd.read_csv(DATA_FILE)
sweep_var = detect_sweep_var(df)

benchmarks = [FILTER_BENCH] if FILTER_BENCH else sorted(df['benchmark'].unique())

for bench in benchmarks:
    df_bench = df[df['benchmark'] == bench]
    if df_bench.empty:
        print(f"Warning: no data for benchmark '{bench}'")
        continue
    plot_benchmark(df_bench, bench, sweep_var, OUTPUT_DIR, csv_stem)
    print(f"Sweep: {sweep_var or 'config comparison'}, configs: {len(df_bench)}")
