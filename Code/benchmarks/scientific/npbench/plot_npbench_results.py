#!/usr/bin/env python3
"""
Plot npbench benchmark results for conv2d and nbody.
Shows performance vs CPU threads, GPU SM percentage,
config comparison bar charts, and stacked time breakdowns.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Colors matching scientific benchmark conventions
COLORS = {
    'cpu-cpu': '#2E86AB',
    'cpu-gpu': '#E94F37',
    'gpu-cpu': '#8AC926',
}


def load_and_process_data(csv_path):
    """Load CSV and compute mean/std per configuration."""
    df = pd.read_csv(csv_path)

    # Separate CPU and GPU data
    cpu_data = df[df['exec_loc'] == 'cpu'].copy()
    gpu_data = df[df['exec_loc'] == 'gpu'].copy()

    # Aggregate CPU data by benchmark and num_threads
    cpu_agg = cpu_data.groupby(['benchmark', 'num_threads']).agg({
        'total_time_ms': ['mean', 'std'],
        'computation_time_ms': ['mean', 'std'],
        'transfer_time_ms': ['mean', 'std']
    }).reset_index()
    cpu_agg.columns = ['benchmark', 'num_threads',
                       'total_mean', 'total_std',
                       'compute_mean', 'compute_std',
                       'transfer_mean', 'transfer_std']

    # Aggregate GPU data by benchmark and sm_percentage
    gpu_agg = gpu_data.groupby(['benchmark', 'sm_percentage']).agg({
        'total_time_ms': ['mean', 'std'],
        'computation_time_ms': ['mean', 'std'],
        'transfer_time_ms': ['mean', 'std']
    }).reset_index()
    gpu_agg.columns = ['benchmark', 'sm_percentage',
                       'total_mean', 'total_std',
                       'compute_mean', 'compute_std',
                       'transfer_mean', 'transfer_std']

    return df, cpu_agg, gpu_agg


def plot_benchmark(cpu_data, gpu_data, benchmark_name, output_dir):
    """Create a figure with CPU threads and GPU SM% plots for a single benchmark."""

    cpu_bench = cpu_data[cpu_data['benchmark'] == benchmark_name].sort_values('num_threads')
    gpu_bench = gpu_data[gpu_data['benchmark'] == benchmark_name].sort_values('sm_percentage')

    if cpu_bench.empty and gpu_bench.empty:
        print(f"No data found for benchmark: {benchmark_name}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{benchmark_name.upper()} Benchmark Performance', fontsize=14, fontweight='bold')

    # Plot 1: CPU threads
    ax1 = axes[0]
    if not cpu_bench.empty:
        threads = cpu_bench['num_threads'].values
        total_mean = cpu_bench['total_mean'].values
        total_std = cpu_bench['total_std'].values

        ax1.errorbar(threads, total_mean, yerr=total_std,
                     marker='o', capsize=4, linewidth=2, markersize=8,
                     color=COLORS['cpu-cpu'], label='Total Time')
        ax1.fill_between(threads, total_mean - total_std, total_mean + total_std,
                        alpha=0.2, color=COLORS['cpu-cpu'])

        ax1.set_xlabel('Number of CPU Threads', fontsize=12)
        ax1.set_ylabel('Time (ms)', fontsize=12)
        ax1.set_title('CPU Performance vs Thread Count', fontsize=12)
        ax1.set_xscale('log', base=2)
        ax1.set_xticks(threads)
        ax1.set_xticklabels([str(int(t)) for t in threads])
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'No CPU data available', ha='center', va='center', transform=ax1.transAxes)

    # Plot 2: GPU SM percentage
    ax2 = axes[1]
    if not gpu_bench.empty:
        sm_pct = gpu_bench['sm_percentage'].values
        total_mean = gpu_bench['total_mean'].values
        total_std = gpu_bench['total_std'].values

        ax2.errorbar(sm_pct, total_mean, yerr=total_std,
                     marker='s', capsize=4, linewidth=2, markersize=8,
                     color=COLORS['cpu-gpu'], label='Total Time')
        ax2.fill_between(sm_pct, total_mean - total_std, total_mean + total_std,
                        alpha=0.2, color=COLORS['cpu-gpu'])

        ax2.set_xlabel('GPU SM Percentage (%)', fontsize=12)
        ax2.set_ylabel('Time (ms)', fontsize=12)
        ax2.set_title('GPU Performance vs SM Percentage', fontsize=12)
        ax2.set_xticks(sm_pct)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No GPU data available', ha='center', va='center', transform=ax2.transAxes)

    plt.tight_layout()

    # Save figure
    output_path = output_dir / f'{benchmark_name}_performance.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_comparison(cpu_data, gpu_data, output_dir):
    """Create a comparison plot showing all benchmarks together."""

    benchmarks = cpu_data['benchmark'].unique()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('NPBench Performance Comparison', fontsize=14, fontweight='bold')

    colors = {'conv2d': COLORS['cpu-cpu'], 'nbody': COLORS['cpu-gpu']}

    # Top row: CPU performance
    ax_cpu = axes[0, 0]
    for bench in benchmarks:
        bench_data = cpu_data[cpu_data['benchmark'] == bench].sort_values('num_threads')
        if not bench_data.empty:
            ax_cpu.errorbar(bench_data['num_threads'], bench_data['total_mean'],
                           yerr=bench_data['total_std'], marker='o', capsize=3,
                           linewidth=2, label=bench, color=colors.get(bench, None))

    ax_cpu.set_xlabel('Number of CPU Threads', fontsize=11)
    ax_cpu.set_ylabel('Time (ms)', fontsize=11)
    ax_cpu.set_title('CPU: Total Time vs Threads', fontsize=12)
    ax_cpu.set_xscale('log', base=2)
    ax_cpu.legend()
    ax_cpu.grid(True, alpha=0.3)

    # Top right: CPU normalized
    ax_cpu_norm = axes[0, 1]
    for bench in benchmarks:
        bench_data = cpu_data[cpu_data['benchmark'] == bench].sort_values('num_threads')
        if not bench_data.empty:
            baseline = bench_data['total_mean'].iloc[0]
            normalized = bench_data['total_mean'] / baseline
            ax_cpu_norm.plot(bench_data['num_threads'], normalized,
                            marker='o', linewidth=2, label=bench, color=colors.get(bench, None))

    ax_cpu_norm.set_xlabel('Number of CPU Threads', fontsize=11)
    ax_cpu_norm.set_ylabel('Normalized Time (1 = 1 thread)', fontsize=11)
    ax_cpu_norm.set_title('CPU: Normalized Performance', fontsize=12)
    ax_cpu_norm.set_xscale('log', base=2)
    ax_cpu_norm.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax_cpu_norm.legend()
    ax_cpu_norm.grid(True, alpha=0.3)

    # Bottom left: GPU performance
    ax_gpu = axes[1, 0]
    for bench in benchmarks:
        bench_data = gpu_data[gpu_data['benchmark'] == bench].sort_values('sm_percentage')
        if not bench_data.empty:
            ax_gpu.errorbar(bench_data['sm_percentage'], bench_data['total_mean'],
                           yerr=bench_data['total_std'], marker='s', capsize=3,
                           linewidth=2, label=bench, color=colors.get(bench, None))

    ax_gpu.set_xlabel('GPU SM Percentage (%)', fontsize=11)
    ax_gpu.set_ylabel('Time (ms)', fontsize=11)
    ax_gpu.set_title('GPU: Total Time vs SM%', fontsize=12)
    ax_gpu.legend()
    ax_gpu.grid(True, alpha=0.3)

    # Bottom right: GPU normalized
    ax_gpu_norm = axes[1, 1]
    for bench in benchmarks:
        bench_data = gpu_data[gpu_data['benchmark'] == bench].sort_values('sm_percentage')
        if not bench_data.empty:
            baseline = bench_data[bench_data['sm_percentage'] == 100]['total_mean'].values
            if len(baseline) > 0:
                normalized = bench_data['total_mean'] / baseline[0]
                ax_gpu_norm.plot(bench_data['sm_percentage'], normalized,
                                marker='s', linewidth=2, label=bench, color=colors.get(bench, None))

    ax_gpu_norm.set_xlabel('GPU SM Percentage (%)', fontsize=11)
    ax_gpu_norm.set_ylabel('Normalized Time (1 = 100% SM)', fontsize=11)
    ax_gpu_norm.set_title('GPU: Normalized Performance', fontsize=12)
    ax_gpu_norm.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax_gpu_norm.legend()
    ax_gpu_norm.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / 'npbench_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_config_comparison(df, output_dir):
    """Bar chart comparing configurations (data_loc/exec_loc), like scientific benchmarks."""
    # Build config label
    df = df.copy()
    df['config'] = df['data_loc'] + '-' + df['exec_loc']

    benchmarks = sorted(df['benchmark'].unique())
    configs = sorted(df['config'].unique())

    # Aggregate: mean total_time_ms per benchmark+config
    agg = df.groupby(['benchmark', 'config']).agg(
        total_mean=('total_time_ms', 'mean'),
        total_std=('total_time_ms', 'std')
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('NPBench: Configuration Comparison', fontsize=14, fontweight='bold')

    x = np.arange(len(benchmarks))
    width = 0.8 / max(len(configs), 1)

    for i, config in enumerate(configs):
        config_data = agg[agg['config'] == config]
        means = []
        stds = []
        for bench in benchmarks:
            row = config_data[config_data['benchmark'] == bench]
            if not row.empty:
                means.append(row['total_mean'].values[0])
                stds.append(row['total_std'].values[0] if pd.notna(row['total_std'].values[0]) else 0)
            else:
                means.append(0)
                stds.append(0)

        color = COLORS.get(config, None)
        offset = (i - (len(configs) - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3,
               label=config, color=color, alpha=0.85)

    ax.set_xlabel('Benchmark', fontsize=12)
    ax.set_ylabel('Total Time (ms)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'npbench_config_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_stacked_time_breakdown(df, output_dir):
    """Stacked bar chart showing compute + transfer time breakdown per config."""
    df = df.copy()
    df['config'] = df['data_loc'] + '-' + df['exec_loc']

    benchmarks = sorted(df['benchmark'].unique())
    configs = sorted(df['config'].unique())

    agg = df.groupby(['benchmark', 'config']).agg(
        compute_mean=('computation_time_ms', 'mean'),
        transfer_mean=('transfer_time_ms', 'mean'),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('NPBench: Time Breakdown (Compute + Transfer)', fontsize=14, fontweight='bold')

    # One group per benchmark-config pair
    labels = []
    compute_vals = []
    transfer_vals = []
    bar_colors = []

    for bench in benchmarks:
        for config in configs:
            row = agg[(agg['benchmark'] == bench) & (agg['config'] == config)]
            if not row.empty:
                labels.append(f"{bench}\n{config}")
                compute_vals.append(row['compute_mean'].values[0])
                transfer_vals.append(row['transfer_mean'].values[0])
                bar_colors.append(COLORS.get(config, '#999999'))

    x = np.arange(len(labels))

    ax.bar(x, transfer_vals, label='Data Transfer', color='orange', alpha=0.85)
    ax.bar(x, compute_vals, bottom=transfer_vals, label='Computation', color='purple', alpha=0.85)

    ax.set_xlabel('Benchmark / Configuration', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'npbench_stacked_breakdown.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    # Find the results CSV
    script_dir = Path(__file__).parent
    csv_path = script_dir / 'results' / 'npbench_results.csv'

    if not csv_path.exists():
        print(f"Error: Results file not found at {csv_path}")
        print("Run the benchmark first: ./benchmark-npbench-pipeline.sh")
        return

    # Create output directory for plots
    output_dir = script_dir / 'plots'
    output_dir.mkdir(exist_ok=True)

    # Load and process data
    print(f"Loading data from {csv_path}")
    df, cpu_data, gpu_data = load_and_process_data(csv_path)

    print(f"CPU configurations: {len(cpu_data)} rows")
    print(f"GPU configurations: {len(gpu_data)} rows")

    # Get unique benchmarks
    benchmarks = set(cpu_data['benchmark'].unique()) | set(gpu_data['benchmark'].unique())
    print(f"Benchmarks found: {benchmarks}")

    # Create individual plots for each benchmark
    for benchmark in benchmarks:
        plot_benchmark(cpu_data, gpu_data, benchmark, output_dir)

    # Create comparison plot
    plot_comparison(cpu_data, gpu_data, output_dir)

    # Create config comparison bar chart (scientific style)
    plot_config_comparison(df, output_dir)

    # Create stacked time breakdown bar chart (scientific style)
    plot_stacked_time_breakdown(df, output_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == '__main__':
    main()
