import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

def plot_results(csv_file, output_dir="plots"):

    df = pd.read_csv(csv_file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sns.set_style("whitegrid")
    #sns.set_palette("husl")

    #y_params = ['nr_input_tokens', 'nr_batches', 'thread_percentage', 'memory_rate']
    y_params = ['memory_rate']
    x_params = ['avg_throughput', 'avg_total']


    for y_param in y_params:

        constant_params = [col for col in df.columns.tolist()[:7] if col != y_param]
        params_text = "\n".join(f"{col}: {df.iloc[0][col]}" for col in constant_params)


        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Benchmark Results: {y_param}', fontsize=16, fontweight='bold')

        # avg_throughput
        sns.barplot(data=df, x=y_param, y='avg_throughput', ax=axes[0], color="steelblue")
        axes[0].set_xlabel(y_param.replace('_', ' ').title(), fontsize=12)
        axes[0].set_ylabel('Average Throughput (tokens/sec)', fontsize=12)
        axes[0].set_title(y_param.replace('_', ' ').title() + ' vs Throughput', fontsize=13)
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].text(0.02, 0.98, params_text, transform=axes[0].transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # avg_total
        sns.barplot(data=df, x=y_param, y='avg_total', ax=axes[1], color="steelblue")
        axes[1].set_xlabel(y_param.replace('_', ' ').title(), fontsize=12)
        axes[1].set_ylabel('Average Total Time (sec)', fontsize=12)
        axes[1].set_title(y_param.replace('_', ' ').title() + ' vs Total Time', fontsize=13)
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        output_path = os.path.join(output_dir, f'{y_param}_plot.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file with benchmark results')
    parser.add_argument('--output_dir', type=str, default='plots', help='Directory to save plots (default: plots)')

    args = parser.parse_args()

    plot_results(args.csv, args.output_dir)
