#!/usr/bin/env python3
"""
Plot scientific benchmark results from CSV files as bar plots.
Usage: python plot_benchmark.py <csv_file_path>
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path


def plot_csv(csv_path):
    """Plot benchmark results from CSV file."""
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    # Determine which variable is being swept
    if 'sm_percentage' in df.columns and df['sm_percentage'].nunique() > 1:
        x_var = 'sm_percentage'
        x_label = 'SM Percentage (%)'
        title_suffix = 'vs SM Percentage'
        x_labels = [f'{int(v)}' for v in df[x_var]]
    elif 'num_threads' in df.columns and df['num_threads'].nunique() > 1:
        x_var = 'num_threads'
        x_label = 'Number of Threads'
        title_suffix = 'vs Number of Threads'
        x_labels = [f'{int(v)}' for v in df[x_var]]
    elif 'data_loc' in df.columns and 'exec_loc' in df.columns:
        # Configuration comparison (data_loc/exec_loc combinations)
        x_var = 'config'
        df['config'] = df['data_loc'] + '/' + df['exec_loc']
        x_label = 'Configuration (Data/Execution)'
        title_suffix = 'by Configuration'
        x_labels = df['config'].tolist()
    else:
        print("Error: Could not determine sweep variable (sm_percentage, num_threads, or data_loc/exec_loc)")
        sys.exit(1)
    
    # Get benchmark name
    benchmark_name = df['benchmark'].iloc[0] if 'benchmark' in df.columns else 'Benchmark'
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{benchmark_name.upper()} Benchmark Results {title_suffix}', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Total Time
    ax1 = axes[0]
    x_values = df[x_var]
    x_pos = range(len(x_values))
    
    # Use avg_total_ms if available, otherwise total_time_ms
    if 'avg_total_ms' in df.columns:
        ax1.bar(x_pos, df['avg_total_ms'], alpha=0.8, color='steelblue')
    elif 'total_time_ms' in df.columns:
        ax1.bar(x_pos, df['total_time_ms'], alpha=0.8, color='steelblue')
    
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Total Execution Time')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Stacked Bar - Data Transfer & Computation
    ax2 = axes[1]
    if 'data_transfer_time_ms' in df.columns and 'computation_time_ms' in df.columns:
        ax2.bar(x_pos, df['data_transfer_time_ms'], 
                label='Data Transfer', alpha=0.8, color='orange')
        ax2.bar(x_pos, df['computation_time_ms'], 
                bottom=df['data_transfer_time_ms'],
                label='Computation', alpha=0.8, color='purple')
        ax2.set_xlabel(x_label)
        ax2.set_ylabel('Time (ms)')
        ax2.set_title('Data Transfer & Computation Time (Stacked)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(x_labels, rotation=45)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(csv_path).with_suffix('.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Show the plot
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Plot scientific benchmark results from CSV files')
    parser.add_argument('csv_file', type=str, 
                        help='Path to the CSV file to plot')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.csv_file).exists():
        print(f"Error: File not found: {args.csv_file}")
        sys.exit(1)
    
    plot_csv(args.csv_file)


if __name__ == '__main__':
    main()
