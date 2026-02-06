#!/usr/bin/env python3
"""
Plot scientific benchmark results for bfs, nn, and leukocyte.
Shows performance comparison across cpu-cpu, cpu-gpu, and gpu-cpu configurations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_data(csv_path):
    """Load CSV and return dataframe."""
    df = pd.read_csv(csv_path)
    # Create a config column combining data_loc and exec_loc
    df['config'] = df['data_loc'] + '-' + df['exec_loc']
    return df


def plot_benchmark_comparison(df, benchmark_name, output_dir):
    """Create bar plot comparing three configurations for a single benchmark."""

    bench_data = df[df['benchmark'] == benchmark_name].copy()

    if bench_data.empty:
        print(f"No data found for benchmark: {benchmark_name}")
        return

    # Sort by config for consistent ordering
    config_order = ['cpu-cpu', 'cpu-gpu', 'gpu-cpu']
    bench_data['config'] = pd.Categorical(bench_data['config'], categories=config_order, ordered=True)
    bench_data = bench_data.sort_values('config')

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f'{benchmark_name.upper()} Benchmark - Configuration Comparison', fontsize=14, fontweight='bold')

    configs = bench_data['config'].values
    total_times = bench_data['avg_total_ms'].values

    x_pos = np.arange(len(configs))
    colors = {'cpu-cpu': '#2E86AB', 'cpu-gpu': '#E94F37', 'gpu-cpu': '#8AC926'}
    bar_colors = [colors.get(c, '#888888') for c in configs]

    bars = ax.bar(x_pos, total_times, color=bar_colors, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Configuration (data_loc-exec_loc)', fontsize=12)
    ax.set_ylabel('Average Total Time (ms)', fontsize=12)
    ax.set_title('Total Execution Time by Configuration', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(configs)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, total_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    output_path = output_dir / f'{benchmark_name}_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_stacked_time_breakdown(df, benchmark_name, output_dir):
    """Create stacked bar plot showing compute vs transfer time breakdown."""

    bench_data = df[df['benchmark'] == benchmark_name].copy()

    if bench_data.empty:
        print(f"No data found for benchmark: {benchmark_name}")
        return

    # Sort by config for consistent ordering
    config_order = ['cpu-cpu', 'cpu-gpu', 'gpu-cpu']
    bench_data['config'] = pd.Categorical(bench_data['config'], categories=config_order, ordered=True)
    bench_data = bench_data.sort_values('config')

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f'{benchmark_name.upper()} Time Breakdown (Compute + Transfer)', fontsize=14, fontweight='bold')

    configs = bench_data['config'].values
    compute_times = bench_data['avg_computation_ms'].values.copy()
    transfer_times = bench_data['avg_transfer_ms'].values.copy()
    total_times = bench_data['avg_total_ms'].values

    # Fix missing breakdown: if compute+transfer is 0 but total exists, use total as compute
    for i in range(len(compute_times)):
        if compute_times[i] + transfer_times[i] < 0.001 and total_times[i] > 0:
            compute_times[i] = total_times[i]

    x_pos = np.arange(len(configs))
    width = 0.6

    # Stacked bars
    bars1 = ax.bar(x_pos, compute_times, width, label='Computation', color='#2E86AB', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x_pos, transfer_times, width, bottom=compute_times, label='Transfer', color='#A3CEF1', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Configuration (data_loc-exec_loc)', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Time Breakdown by Configuration', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(configs)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add total time labels on top
    for i, (c, t, total) in enumerate(zip(compute_times, transfer_times, total_times)):
        stacked_height = c + t
        ax.text(i, stacked_height + 0.5, f'{total:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    output_path = output_dir / f'{benchmark_name}_time_breakdown.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_all_benchmarks_overview(df, output_dir):
    """Create overview plot showing all benchmarks side by side."""

    benchmarks = df['benchmark'].unique()
    config_order = ['cpu-cpu', 'cpu-gpu', 'gpu-cpu']

    fig, axes = plt.subplots(1, len(benchmarks), figsize=(5 * len(benchmarks), 5))
    fig.suptitle('Scientific Benchmarks Overview', fontsize=14, fontweight='bold')

    if len(benchmarks) == 1:
        axes = [axes]

    colors = {'cpu-cpu': '#2E86AB', 'cpu-gpu': '#E94F37', 'gpu-cpu': '#8AC926'}

    for ax, benchmark in zip(axes, benchmarks):
        bench_data = df[df['benchmark'] == benchmark].copy()
        bench_data['config'] = pd.Categorical(bench_data['config'], categories=config_order, ordered=True)
        bench_data = bench_data.sort_values('config')

        configs = bench_data['config'].values
        total_times = bench_data['avg_total_ms'].values

        x_pos = np.arange(len(configs))
        bar_colors = [colors.get(c, '#888888') for c in configs]

        bars = ax.bar(x_pos, total_times, color=bar_colors, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Configuration', fontsize=10)
        ax.set_ylabel('Avg Total Time (ms)', fontsize=10)
        ax.set_title(f'{benchmark.upper()}', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, total_times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    output_path = output_dir / 'all_benchmarks_overview.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    # Find the results CSV
    script_dir = Path(__file__).parent
    csv_path = script_dir / 'scientific-benchmarks.csv'

    if not csv_path.exists():
        print(f"Error: Results file not found at {csv_path}")
        return

    # Create output directory for plots
    output_dir = script_dir / 'plots'
    output_dir.mkdir(exist_ok=True)

    # Load data
    print(f"Loading data from {csv_path}")
    df = load_data(csv_path)

    benchmarks = df['benchmark'].unique()
    print(f"Benchmarks found: {list(benchmarks)}")

    # Create individual plots for each benchmark
    for benchmark in benchmarks:
        plot_benchmark_comparison(df, benchmark, output_dir)
        plot_stacked_time_breakdown(df, benchmark, output_dir)

    # Create overview plot
    plot_all_benchmarks_overview(df, output_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == '__main__':
    main()
