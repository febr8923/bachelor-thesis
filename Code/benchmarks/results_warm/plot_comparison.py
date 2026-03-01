#!/usr/bin/env python3
"""
Plot DL benchmark results as stacked bar charts (load time + inference time).

Supports warm-start CSVs (one aggregated row per config) and cold-start CSVs
(NR_COLD_ITERATIONS raw rows per config that are averaged here).

The script infers CPU vs GPU and warm vs cold from the filename pattern
produced by run-dl-clean.py / benchmark-dl-pipeline.sh:
    <model_loc>-<exec_loc>-<mode>-<cold_start>.csv
    e.g.  cpu-cpu-1-False.csv   (warm, CPU exec)
          gpu-gpu-1-True.csv    (cold, GPU exec)

Usage:
    python plot_comparison.py <data_file.csv> [output_file.png] [title]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Must match NR_COLD_ITERATIONS in benchmark-dl-pipeline.sh
NR_COLD_ITERATIONS = 5

# Known x-axis configurations from benchmark-dl-pipeline.sh
CPU_THREAD_COUNTS = [1, 2, 4, 8, 16, 36, 72]
GPU_SM_PERCENTAGES = list(range(10, 101, 10))


def detect_exec_loc(filename):
    """Detect execution location from the filename pattern <model_loc>-<exec_loc>-..."""
    base = os.path.basename(filename).replace('.csv', '')
    parts = base.split('-')
    # parts[1] is exec_loc
    if len(parts) >= 2:
        return parts[1]
    # fallback: search for keywords
    fl = filename.lower()
    if 'gpu' in fl:
        return 'gpu'
    return 'cpu'


def detect_cold_start(filename):
    """Detect cold start from filename pattern ...-True/False.csv"""
    base = os.path.basename(filename).replace('.csv', '')
    return base.endswith('True')


def load_and_prepare(filepath):
    """Load a DL benchmark CSV and return per-config averages + min/max.

    Returns dict with keys:
        load_avg, load_min, load_max,
        inference_avg, inference_min, inference_max
    Each is a numpy array with one entry per config.
    """
    df = pd.read_csv(filepath)

    is_cold = detect_cold_start(filepath)

    if is_cold and len(df) > 1:
        # Cold files have NR_COLD_ITERATIONS rows per configuration.
        # Each row is a single cold run, so avg=min=max within that row.
        # We aggregate the 5 runs: avg across runs, true min, true max.
        n = NR_COLD_ITERATIONS
        num_configs = len(df) // n
        # Trim any trailing incomplete group
        df = df.iloc[:num_configs * n]
        df['config_group'] = np.repeat(np.arange(num_configs), n)
        grouped = df.groupby('config_group', sort=False)

        avg_cols = {
            'avg_load_time': 'mean',
            'avg_inference_time': 'mean',
            'avg_total_time': 'mean',
        }
        min_cols = {
            'min_load_time': 'min',
            'min_inference_time': 'min',
            'min_total_time': 'min',
        }
        max_cols = {
            'max_load_time': 'max',
            'max_inference_time': 'max',
            'max_total_time': 'max',
        }
        agg_dict = {**avg_cols, **min_cols, **max_cols}
        df = grouped.agg(agg_dict).reset_index(drop=True)

    return {
        'load_avg': df['avg_load_time'].values,
        'load_min': df['min_load_time'].values,
        'load_max': df['max_load_time'].values,
        'inference_avg': df['avg_inference_time'].values,
        'inference_min': df['min_inference_time'].values,
        'inference_max': df['max_inference_time'].values,
    }


def make_labels(exec_loc, num_configs):
    """Generate x-axis labels based on execution location and number of configs."""
    if exec_loc == 'gpu':
        all_labels = [f'{p}%' for p in GPU_SM_PERCENTAGES]
    else:
        all_labels = [str(t) for t in CPU_THREAD_COUNTS]

    # Use as many labels as we have data (some configs may have failed)
    if num_configs <= len(all_labels):
        return all_labels[:num_configs]
    # More rows than expected — just use indices
    return [str(i) for i in range(num_configs)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if len(sys.argv) < 2:
    print("Usage: python plot_comparison.py <data_file.csv> [output_file.png] [title]")
    sys.exit(1)

DATA_FILE = sys.argv[1]
OUTPUT_FILE = sys.argv[2] if len(sys.argv) >= 3 else DATA_FILE.replace('.csv', '_plot.png')
PLOT_TITLE = sys.argv[3] if len(sys.argv) >= 4 else os.path.basename(DATA_FILE).replace('.csv', '')

exec_loc = detect_exec_loc(DATA_FILE)
is_cold = detect_cold_start(DATA_FILE)
data = load_and_prepare(DATA_FILE)
load_avg = data['load_avg']
inference_avg = data['inference_avg']
num_configs = len(load_avg)
labels = make_labels(exec_loc, num_configs)

# Compute asymmetric error bars (distance from avg to min / avg to max)
load_err = np.array([load_avg - data['load_min'],
                     data['load_max'] - load_avg])
inference_err = np.array([inference_avg - data['inference_min'],
                          data['inference_max'] - inference_avg])
# Clamp negative errors to 0 (safety against floating-point quirks)
load_err = np.clip(load_err, 0, None)
inference_err = np.clip(inference_err, 0, None)

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.6
x = np.arange(num_configs)

ax.bar(x, load_avg, width, label='Load Time', color='#3498db',
       yerr=load_err, capsize=3, error_kw={'elinewidth': 1})
ax.bar(x, inference_avg, width, bottom=load_avg, label='Inference Time',
       color='#e74c3c', yerr=inference_err, capsize=3,
       error_kw={'elinewidth': 1})

xlabel = 'GPU SM Utilization' if exec_loc == 'gpu' else 'Number of CPU Threads'
cold_tag = ' (Cold Start)' if is_cold else ' (Warm)'
ax.set_xlabel(xlabel, fontsize=12)
ax.set_ylabel('Time (seconds)', fontsize=12)
ax.set_title(f'{exec_loc.upper()} Performance{cold_tag} — {PLOT_TITLE}',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45 if exec_loc == 'gpu' else 0, ha='right')

ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
print(f"Exec location: {exec_loc.upper()}, cold_start: {is_cold}, configs: {num_configs}")
print(f"Plot saved as '{OUTPUT_FILE}'")

plt.show()
