import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse


path = argparse.ArgumentParser()
path.add_argument('--csv', type=str, required=False, default='results/alexnet/cpu-1-memory.csv',
                  help='Path to CSV file with CPU and memory usage data')
args = path.parse_args()
file = args.csv
# Read the new CSV format without header
df = pd.read_csv(file, header=None, names=['timestamp', 'cpu_util_pct', 'mem_total_bytes','mem_used_bytes','mem_available_bytes'])
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Calculate relative time and derived metrics
start_time = df['timestamp'].min()
df['relative_time_s'] = (df['timestamp'] - start_time).dt.total_seconds()
df['mem_util_pct'] = (df['mem_used_bytes'] / df['mem_total_bytes']) * 100
df['mem_total_mb'] = df['mem_total_bytes'] / (1024**2)
df['mem_used_mb'] = df['mem_used_bytes'] / (1024**2)
df['mem_available_mb'] = df['mem_available_bytes'] / (1024**2)

# Group by relative time and aggregate with max
df_grouped = df.groupby('relative_time_s').agg({
    'cpu_util_pct': 'max',
    'mem_util_pct': 'max',
    'mem_total_mb': 'max',
    'mem_used_mb': 'max',
    'mem_available_mb': 'max'
}).reset_index()

# PLOT 1: CPU/Memory utilization %
df_util = df_grouped[['relative_time_s', 'cpu_util_pct', 'mem_util_pct']].melt(
    id_vars='relative_time_s', 
    value_vars=['cpu_util_pct', 'mem_util_pct'],
    var_name='metric', value_name='value'
)

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_util, x='relative_time_s', y='value', hue='metric', 
             palette='Set1', linewidth=2.5)
plt.title('CPU/Memory utilization aggregated with max in %', fontsize=16, fontweight='bold')
plt.xlabel('Time (seconds from start)', fontsize=12)
plt.ylabel('Utilization %', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(title='Metrics', title_fontsize=11, fontsize=10)
plt.tight_layout()
plt.savefig('cpu_memory_relative.png', dpi=500, bbox_inches='tight', facecolor='white')

# PLOT 2: Memory values (MB scale) - MAX values
df_mem = df_grouped[['relative_time_s', 'mem_total_mb', 'mem_used_mb', 'mem_available_mb']].melt(
    id_vars='relative_time_s', 
    value_vars=['mem_total_mb', 'mem_used_mb', 'mem_available_mb'],
    var_name='metric', value_name='value'
)

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_mem, x='relative_time_s', y='value', hue='metric', 
             palette='Set2', linewidth=2.5)
plt.title('System Memory aggregated with max (MB)', fontsize=16, fontweight='bold')
plt.xlabel('Time (seconds from start)', fontsize=12)
plt.ylabel('Memory (MB)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(title='Metrics', title_fontsize=11, fontsize=10)
plt.tight_layout()
plt.savefig('cpu_memory_absolute.png', dpi=500, bbox_inches='tight', facecolor='white')
