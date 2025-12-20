import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('gpu-memory-gpu-gpu-False-75.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

start_time = df['timestamp'].min()
df['relative_time_s'] = (df['timestamp'] - start_time).dt.total_seconds()

df_grouped = df.groupby('relative_time_s').agg({
    'gpu_util%': 'max',
    'mem_util%': 'max',
    'mem_total_mb': 'max',
    'mem_free_mb': 'max', 
    'mem_used_mb': 'max',
    'mem_reserved_mb': 'max'
}).reset_index()

df_util = df_grouped[['relative_time_s', 'gpu_util%', 'mem_util%']].melt(
    id_vars='relative_time_s', 
    value_vars=['gpu_util%', 'mem_util%'],
    var_name='metric', value_name='value'
)

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_util, x='relative_time_s', y='value', hue='metric', 
             palette='Set1', linewidth=2.5, marker='o')
plt.title('GPU Memory aggregated with max in %', fontsize=16, fontweight='bold')
plt.xlabel('Time (seconds from start)', fontsize=12)
plt.ylabel('Utilization %', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(title='Metrics', title_fontsize=11, fontsize=10)
plt.tight_layout()

plt.savefig('gpu_relative.png', dpi=300, bbox_inches='tight', facecolor='white')

# PLOT 2: Memory values (MB scale) - MAX values
df_mem = df_grouped[['relative_time_s', 'mem_total_mb', 'mem_free_mb', 
                     'mem_used_mb', 'mem_reserved_mb']].melt(
    id_vars='relative_time_s', 
    value_vars=['mem_total_mb', 'mem_free_mb', 'mem_used_mb', 'mem_reserved_mb'],
    var_name='metric', value_name='value'
)

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_mem, x='relative_time_s', y='value', hue='metric', 
             palette='Set2', linewidth=2.5, marker='o')
plt.title('GPU Memory aggregated with max', fontsize=16, fontweight='bold')
plt.xlabel('Time (seconds from start)', fontsize=12)
plt.ylabel('Memory (MB)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(title='Metrics', title_fontsize=11, fontsize=10)
plt.tight_layout()

# Save memory plot
plt.savefig('gpu_absolute.png', dpi=300, bbox_inches='tight', facecolor='white')

