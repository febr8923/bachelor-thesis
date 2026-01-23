#!/usr/bin/env python3
"""
Script to plot CPU or GPU benchmark results with stacked bar plots.
Shows average load time and execution time for different configurations.

Usage: python plot_comparison.py <data_file.csv> [output_file.png] [title]
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os

def load_csv_data(filepath):
    """Load CSV data and return as numpy array."""
    data = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append([float(x) for x in row])
    return np.array(data)

def detect_data_type(data, filename):
    """Detect if data is from CPU or GPU based on number of rows and filename."""
    num_rows = data.shape[0]
    
    # Check filename first
    filename_lower = filename.lower()
    if 'gpu' in filename_lower and 'cpu' not in filename_lower:
        return 'gpu'
    elif 'cpu' in filename_lower and 'gpu' not in filename_lower:
        return 'cpu'
    
    # Fall back to row count heuristic
    if num_rows == 10:
        return 'gpu'  # 10% to 100% in 10% increments
    elif num_rows == 7:
        return 'cpu'  # 1, 2, 4, 8, 16, 36, 72 threads
    else:
        print(f"Warning: Unexpected number of rows ({num_rows}). Assuming GPU data.")
        return 'gpu'

# Parse command-line arguments
if len(sys.argv) < 2:
    print("Usage: python plot_comparison.py <data_file.csv> [output_file.png] [title]")
    sys.exit(1)

DATA_FILE = sys.argv[1]
OUTPUT_FILE = sys.argv[2] if len(sys.argv) >= 3 else DATA_FILE.replace('.csv', '_plot.png')
PLOT_TITLE = sys.argv[3] if len(sys.argv) >= 4 else os.path.basename(DATA_FILE).replace('.csv', '')

# Load data
data = load_csv_data(DATA_FILE)
data_type = detect_data_type(data, DATA_FILE)

# Extract average times (columns 1, 4, 7 are avg load, avg execution, avg total)
# Column indices: 0-2 (load max/avg/min), 3-5 (exec max/avg/min), 6-8 (total max/avg/min)
load_avg = data[:, 1]  # avg load time
exec_avg = data[:, 4]  # avg execution time

# Create figure with single plot
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.6

if data_type == 'gpu':
    # GPU plot
    labels = [f'{i}%' for i in range(10, 101, 10)]
    x = np.arange(len(labels))
    
    ax.bar(x, load_avg, width, label='Load Time', color='#3498db')
    ax.bar(x, exec_avg, width, bottom=load_avg, label='Execution Time', color='#e74c3c')
    
    ax.set_xlabel('GPU SM Utilization', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title(f'GPU Performance - {PLOT_TITLE}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
else:
    # CPU plot
    labels = ['1', '2', '4', '8', '16', '36', '72']
    x = np.arange(len(labels))
    
    ax.bar(x, load_avg, width, label='Load Time', color='#3498db')
    ax.bar(x, exec_avg, width, bottom=load_avg, label='Execution Time', color='#e74c3c')
    
    ax.set_xlabel('Number of CPU Threads', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title(f'CPU Performance - {PLOT_TITLE}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

ax.legend()
ax.grid(axis='y', alpha=0.3)

# Adjust layout and save
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
print(f"Detected {data_type.upper()} data with {data.shape[0]} rows")
print(f"Plot saved as '{OUTPUT_FILE}'")

plt.show()
