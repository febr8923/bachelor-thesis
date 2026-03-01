import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np


def plot_results(csv_file, output_dir="plots", y_param="thread_percentage"):

    df = pd.read_csv(csv_file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})

    constant_params = [col for col in df.columns.tolist()[:7] if col != y_param]
    params_text = "\n".join(f"{col}: {df.iloc[0][col]}" for col in constant_params)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Benchmark Results: {y_param}', fontsize=16, fontweight='bold')

    x = np.arange(len(df))
    x_labels = [str(v) for v in df[y_param].tolist()]
    bar_kwargs = dict(color="steelblue", width=0.6, capsize=4,
                      error_kw=dict(ecolor='black', elinewidth=1.2, alpha=0.8))

    # throughput
    throughput_yerr = [
        (df['avg_throughput'] - df['min_throughput']).tolist(),
        (df['max_throughput'] - df['avg_throughput']).tolist()
    ]
    axes[0].bar(x, df['avg_throughput'], yerr=throughput_yerr, **bar_kwargs)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(x_labels, rotation=45, ha='right')
    axes[0].set_xlabel(y_param.replace('_', ' ').title(), fontsize=12)
    axes[0].set_ylabel('Average Throughput (tokens/sec)', fontsize=12)
    axes[0].set_title(y_param.replace('_', ' ').title() + ' vs Throughput', fontsize=13)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].text(0.02, 0.98, params_text, transform=axes[0].transAxes,
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # total time — stack load time on top for cold start files
    is_cold = ('avg_load_time' in df.columns and df['avg_load_time'].notna().any()
               and df['avg_load_time'].sum() > 0)

    if is_cold:
        load_avg = df['avg_load_time'].fillna(0)
        load_min = df['min_load_time'].fillna(0)
        load_max = df['max_load_time'].fillna(0)
        total_with_load_avg = df['avg_total'] + load_avg
        total_with_load_min = df['min_total'] + load_min
        total_with_load_max = df['max_total'] + load_max
        total_yerr = np.clip([
            (total_with_load_avg - total_with_load_min).tolist(),
            (total_with_load_max - total_with_load_avg).tolist()
        ], 0, None)
        axes[1].bar(x, df['avg_total'], width=0.6, label='Inference', color='#e74c3c')
        axes[1].bar(x, load_avg, width=0.6, bottom=df['avg_total'], label='Load',
                    color='#3498db', yerr=total_yerr, capsize=4,
                    error_kw=dict(ecolor='black', elinewidth=1.2, alpha=0.8))
        axes[1].legend()
    else:
        total_yerr = [
            (df['avg_total'] - df['min_total']).tolist(),
            (df['max_total'] - df['avg_total']).tolist()
        ]
        axes[1].bar(x, df['avg_total'], yerr=total_yerr, **bar_kwargs)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x_labels, rotation=45, ha='right')
    axes[1].set_xlabel(y_param.replace('_', ' ').title(), fontsize=12)
    axes[1].set_ylabel('Average Total Time (sec)', fontsize=12)
    axes[1].set_title(y_param.replace('_', ' ').title() + ' vs Total Time', fontsize=13)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    csv_stem = os.path.splitext(os.path.basename(csv_file))[0]
    output_path = os.path.join(output_dir, f'{csv_stem}_{y_param}_plot.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file with benchmark results')
    parser.add_argument('--output_dir', type=str, default='plots', help='Directory to save plots (default: plots)')
    parser.add_argument('--y_param', type=str, required=True,
                        help='Column used as x-axis sweep variable (thread_percentage, nr_batches, nr_input_tokens, memory_rate)')

    args = parser.parse_args()

    plot_results(args.csv, args.output_dir, args.y_param)
