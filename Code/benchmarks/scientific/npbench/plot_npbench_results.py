#!/usr/bin/env python3
"""
Plot npbench benchmark results for conv2d and nbody.
Shows performance vs CPU threads and GPU SM percentage.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_and_process_data(csv_path):
    """Load CSV and compute mean per configuration."""
    df = pd.read_csv(csv_path)

    # Separate CPU and GPU data
    cpu_data = df[df['exec_loc'] == 'cpu'].copy()
    gpu_data = df[df['exec_loc'] == 'gpu'].copy()

    # Aggregate CPU data by benchmark and num_threads
    cpu_agg = cpu_data.groupby(['benchmark', 'num_threads']).agg({
        'total_time_ms': 'mean',
        'computation_time_ms': 'mean',
        'transfer_time_ms': 'mean'
    }).reset_index()
    cpu_agg.columns = ['benchmark', 'num_threads', 'total_mean', 'compute_mean', 'transfer_mean']

    # Aggregate GPU data by benchmark and sm_percentage
    gpu_agg = gpu_data.groupby(['benchmark', 'sm_percentage']).agg({
        'total_time_ms': 'mean',
        'computation_time_ms': 'mean',
        'transfer_time_ms': 'mean'
    }).reset_index()
    gpu_agg.columns = ['benchmark', 'sm_percentage', 'total_mean', 'compute_mean', 'transfer_mean']

    return cpu_agg, gpu_agg


def plot_benchmark(cpu_data, gpu_data, benchmark_name, output_dir):
    """Create a figure with CPU threads and GPU SM% bar plots for a single benchmark."""

    cpu_bench = cpu_data[cpu_data['benchmark'] == benchmark_name].sort_values('num_threads')
    gpu_bench = gpu_data[gpu_data['benchmark'] == benchmark_name].sort_values('sm_percentage')

    if cpu_bench.empty and gpu_bench.empty:
        print(f"No data found for benchmark: {benchmark_name}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{benchmark_name.upper()} Benchmark Performance', fontsize=14, fontweight='bold')

    # Plot 1: CPU threads (bar plot)
    ax1 = axes[0]
    if not cpu_bench.empty:
        threads = cpu_bench['num_threads'].astype(int).astype(str).values
        total_mean = cpu_bench['total_mean'].values

        x_pos = np.arange(len(threads))
        bars = ax1.bar(x_pos, total_mean, color='#2E86AB', edgecolor='black', linewidth=0.5)

        ax1.set_xlabel('Number of CPU Threads', fontsize=12)
        ax1.set_ylabel('Time (ms)', fontsize=12)
        ax1.set_title('CPU Performance vs Thread Count', fontsize=12)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(threads)
        ax1.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, total_mean):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    else:
        ax1.text(0.5, 0.5, 'No CPU data available', ha='center', va='center', transform=ax1.transAxes)

    # Plot 2: GPU SM percentage (bar plot)
    ax2 = axes[1]
    if not gpu_bench.empty:
        sm_pct = gpu_bench['sm_percentage'].astype(int).astype(str).values
        total_mean = gpu_bench['total_mean'].values

        x_pos = np.arange(len(sm_pct))
        bars = ax2.bar(x_pos, total_mean, color='#E94F37', edgecolor='black', linewidth=0.5)

        ax2.set_xlabel('GPU SM Percentage (%)', fontsize=12)
        ax2.set_ylabel('Time (ms)', fontsize=12)
        ax2.set_title('GPU Performance vs SM Percentage', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(sm_pct)
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, total_mean):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'No GPU data available', ha='center', va='center', transform=ax2.transAxes)

    plt.tight_layout()

    # Save figure
    output_path = output_dir / f'{benchmark_name}_performance.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_fixed_config(cpu_data, gpu_data, output_dir, cpu_threads=4, gpu_sm=100):
    """Create bar plots for fixed configurations (4 threads CPU, 100% SM GPU)."""

    benchmarks = list(set(cpu_data['benchmark'].unique()) | set(gpu_data['benchmark'].unique()))
    colors = {'conv2d': '#2E86AB', 'nbody': '#E94F37'}

    # Filter data for specific configurations
    cpu_fixed = cpu_data[cpu_data['num_threads'] == cpu_threads]
    gpu_fixed = gpu_data[gpu_data['sm_percentage'] == gpu_sm]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Benchmark Performance at Fixed Configurations', fontsize=14, fontweight='bold')

    # Plot 1: CPU at fixed threads
    ax1 = axes[0]
    if not cpu_fixed.empty:
        bench_names = cpu_fixed['benchmark'].values
        total_times = cpu_fixed['total_mean'].values

        x_pos = np.arange(len(bench_names))
        bar_colors = [colors.get(b, '#888888') for b in bench_names]
        bars = ax1.bar(x_pos, total_times, color=bar_colors, edgecolor='black', linewidth=0.5)

        ax1.set_xlabel('Benchmark', fontsize=12)
        ax1.set_ylabel('Time (ms)', fontsize=12)
        ax1.set_title(f'CPU Performance ({cpu_threads} threads)', fontsize=12)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(bench_names)
        ax1.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, total_times):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    # Plot 2: GPU at fixed SM%
    ax2 = axes[1]
    if not gpu_fixed.empty:
        bench_names = gpu_fixed['benchmark'].values
        total_times = gpu_fixed['total_mean'].values

        x_pos = np.arange(len(bench_names))
        bar_colors = [colors.get(b, '#888888') for b in bench_names]
        bars = ax2.bar(x_pos, total_times, color=bar_colors, edgecolor='black', linewidth=0.5)

        ax2.set_xlabel('Benchmark', fontsize=12)
        ax2.set_ylabel('Time (ms)', fontsize=12)
        ax2.set_title(f'GPU Performance ({gpu_sm}% SM)', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(bench_names)
        ax2.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, total_times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    output_path = output_dir / f'fixed_config_cpu{cpu_threads}_gpu{gpu_sm}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_stacked_time_breakdown(cpu_data, gpu_data, benchmark_name, output_dir):
    """Create stacked bar plots showing compute vs transfer time breakdown."""

    cpu_bench = cpu_data[cpu_data['benchmark'] == benchmark_name].sort_values('num_threads')
    gpu_bench = gpu_data[gpu_data['benchmark'] == benchmark_name].sort_values('sm_percentage')

    if cpu_bench.empty and gpu_bench.empty:
        print(f"No data found for benchmark: {benchmark_name}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{benchmark_name.upper()} Time Breakdown (Compute + Transfer)', fontsize=14, fontweight='bold')

    # Plot 1: CPU threads (stacked bar)
    ax1 = axes[0]
    if not cpu_bench.empty:
        threads = cpu_bench['num_threads'].astype(int).astype(str).values
        compute = cpu_bench['compute_mean'].values
        transfer = cpu_bench['transfer_mean'].values

        x_pos = np.arange(len(threads))
        width = 0.6

        bars1 = ax1.bar(x_pos, compute, width, label='Computation', color='#2E86AB', edgecolor='black', linewidth=0.5)
        bars2 = ax1.bar(x_pos, transfer, width, bottom=compute, label='Transfer', color='#A3CEF1', edgecolor='black', linewidth=0.5)

        ax1.set_xlabel('Number of CPU Threads', fontsize=12)
        ax1.set_ylabel('Time (ms)', fontsize=12)
        ax1.set_title('CPU: Time Breakdown by Thread Count', fontsize=12)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(threads)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Add total time labels
        for i, (c, t) in enumerate(zip(compute, transfer)):
            total = c + t
            ax1.text(i, total + 0.5, f'{total:.1f}', ha='center', va='bottom', fontsize=8)
    else:
        ax1.text(0.5, 0.5, 'No CPU data available', ha='center', va='center', transform=ax1.transAxes)

    # Plot 2: GPU SM percentage (stacked bar)
    ax2 = axes[1]
    if not gpu_bench.empty:
        sm_pct = gpu_bench['sm_percentage'].astype(int).astype(str).values
        compute = gpu_bench['compute_mean'].values
        transfer = gpu_bench['transfer_mean'].values

        x_pos = np.arange(len(sm_pct))
        width = 0.6

        bars1 = ax2.bar(x_pos, compute, width, label='Computation', color='#E94F37', edgecolor='black', linewidth=0.5)
        bars2 = ax2.bar(x_pos, transfer, width, bottom=compute, label='Transfer', color='#F4A261', edgecolor='black', linewidth=0.5)

        ax2.set_xlabel('GPU SM Percentage (%)', fontsize=12)
        ax2.set_ylabel('Time (ms)', fontsize=12)
        ax2.set_title('GPU: Time Breakdown by SM%', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(sm_pct)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # Add total time labels
        for i, (c, t) in enumerate(zip(compute, transfer)):
            total = c + t
            ax2.text(i, total + 0.5, f'{total:.1f}', ha='center', va='bottom', fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'No GPU data available', ha='center', va='center', transform=ax2.transAxes)

    plt.tight_layout()

    output_path = output_dir / f'{benchmark_name}_time_breakdown.png'
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
    cpu_data, gpu_data = load_and_process_data(csv_path)

    print(f"CPU configurations: {len(cpu_data)} rows")
    print(f"GPU configurations: {len(gpu_data)} rows")

    # Get unique benchmarks
    benchmarks = set(cpu_data['benchmark'].unique()) | set(gpu_data['benchmark'].unique())
    print(f"Benchmarks found: {benchmarks}")

    # Create individual plots for each benchmark
    for benchmark in benchmarks:
        plot_benchmark(cpu_data, gpu_data, benchmark, output_dir)
        plot_stacked_time_breakdown(cpu_data, gpu_data, benchmark, output_dir)

    # Create fixed configuration plot
    plot_fixed_config(cpu_data, gpu_data, output_dir, cpu_threads=4, gpu_sm=100)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == '__main__':
    main()
