"""
Unified plotting utilities for all benchmark types
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from typing import List, Optional, Tuple


class BenchmarkPlotter:
    """Unified plotter for benchmark results"""

    def __init__(self, output_dir: str = "plots"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        sns.set_style("whitegrid")

    def plot_vllm_results(self, csv_file: str, model_name: str):
        """Plot vLLM benchmark results"""
        df = pd.read_csv(csv_file)

        # Determine varying parameter
        varying_params = []
        for col in ['thread_percentage', 'nr_batches', 'nr_input_tokens', 'memory_rate']:
            if col in df.columns and df[col].nunique() > 1:
                varying_params.append(col)

        if not varying_params:
            print(f"No varying parameters found in {csv_file}")
            return

        for param in varying_params:
            self._plot_vllm_param(df, param, model_name)

    def _plot_vllm_param(self, df: pd.DataFrame, param: str, model_name: str):
        """Plot vLLM results for a specific varying parameter"""
        # Get constant parameters for annotation
        constant_params = [col for col in ['nr_input_tokens', 'nr_batches',
                                           'thread_percentage', 'memory_rate']
                          if col in df.columns and col != param]
        params_text = "\n".join(f"{col}: {df.iloc[0][col]}" for col in constant_params)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{model_name}: {param.replace("_", " ").title()}',
                    fontsize=16, fontweight='bold')

        # Throughput plot
        sns.barplot(data=df, x=param, y='avg_throughput', ax=axes[0], color="steelblue")
        axes[0].set_xlabel(param.replace('_', ' ').title(), fontsize=12)
        axes[0].set_ylabel('Average Throughput (tokens/sec)', fontsize=12)
        axes[0].set_title('Throughput', fontsize=13)
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].text(0.02, 0.98, params_text, transform=axes[0].transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Total time plot
        sns.barplot(data=df, x=param, y='avg_total', ax=axes[1], color="coral")
        axes[1].set_xlabel(param.replace('_', ' ').title(), fontsize=12)
        axes[1].set_ylabel('Average Total Time (sec)', fontsize=12)
        axes[1].set_title('Total Time', fontsize=13)
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, f'{model_name}_{param}_vllm.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {output_path}")
        plt.close(fig)

    def plot_image_results(self, csv_file: str, model_name: str):
        """Plot image model benchmark results"""
        df = pd.read_csv(csv_file)

        # Determine varying parameter
        varying_params = []
        for col in ['thread_percentage', 'batch_size']:
            if col in df.columns and df[col].nunique() > 1:
                varying_params.append(col)

        if not varying_params:
            print(f"No varying parameters found in {csv_file}")
            return

        for param in varying_params:
            self._plot_image_param(df, param, model_name)

    def _plot_image_param(self, df: pd.DataFrame, param: str, model_name: str):
        """Plot image model results for a specific varying parameter"""
        # Get constant parameters for annotation
        constant_params = [col for col in ['batch_size', 'thread_percentage',
                                           'model_loc', 'exec_loc']
                          if col in df.columns and col != param]
        params_text = "\n".join(f"{col}: {df.iloc[0][col]}" for col in constant_params)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'{model_name}: {param.replace("_", " ").title()}',
                    fontsize=16, fontweight='bold')

        # Load time plot
        sns.barplot(data=df, x=param, y='avg_load_time', ax=axes[0], color="steelblue")
        axes[0].set_xlabel(param.replace('_', ' ').title(), fontsize=12)
        axes[0].set_ylabel('Average Load Time (sec)', fontsize=12)
        axes[0].set_title('Load Time', fontsize=13)
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].text(0.02, 0.98, params_text, transform=axes[0].transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Inference time plot
        sns.barplot(data=df, x=param, y='avg_inference_time', ax=axes[1], color="coral")
        axes[1].set_xlabel(param.replace('_', ' ').title(), fontsize=12)
        axes[1].set_ylabel('Average Inference Time (sec)', fontsize=12)
        axes[1].set_title('Inference Time', fontsize=13)
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].tick_params(axis='x', rotation=45)

        # Total time plot
        sns.barplot(data=df, x=param, y='avg_total_time', ax=axes[2], color="mediumseagreen")
        axes[2].set_xlabel(param.replace('_', ' ').title(), fontsize=12)
        axes[2].set_ylabel('Average Total Time (sec)', fontsize=12)
        axes[2].set_title('Total Time', fontsize=13)
        axes[2].grid(True, alpha=0.3, axis='y')
        axes[2].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, f'{model_name}_{param}_image.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {output_path}")
        plt.close(fig)

    def plot_memory_usage(self, csv_file: str, model_name: str, device: str = "gpu"):
        """Plot memory usage over time"""
        df = pd.read_csv(csv_file)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{model_name}: {device.upper()} Memory Usage',
                    fontsize=16, fontweight='bold')

        if device == "gpu":
            # GPU utilization
            axes[0].plot(df.index, df['gpu_util%'], color='steelblue', linewidth=2)
            axes[0].set_xlabel('Sample', fontsize=12)
            axes[0].set_ylabel('GPU Utilization (%)', fontsize=12)
            axes[0].set_title('GPU Utilization Over Time', fontsize=13)
            axes[0].grid(True, alpha=0.3)

            # Memory utilization
            axes[1].plot(df.index, df['mem_used_mb'], color='coral', linewidth=2, label='Used')
            axes[1].plot(df.index, df['mem_total_mb'], color='gray', linewidth=2,
                        linestyle='--', label='Total')
            axes[1].set_xlabel('Sample', fontsize=12)
            axes[1].set_ylabel('Memory (MB)', fontsize=12)
            axes[1].set_title('GPU Memory Usage Over Time', fontsize=13)
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
        else:
            # CPU utilization
            axes[0].plot(df.index, df['cpu_util_pct'], color='steelblue', linewidth=2)
            axes[0].set_xlabel('Sample', fontsize=12)
            axes[0].set_ylabel('CPU Utilization (%)', fontsize=12)
            axes[0].set_title('CPU Utilization Over Time', fontsize=13)
            axes[0].grid(True, alpha=0.3)

            # Memory usage
            df['mem_used_gb'] = df['mem_used_bytes'] / 1e9
            df['mem_total_gb'] = df['mem_total_bytes'] / 1e9
            axes[1].plot(df.index, df['mem_used_gb'], color='coral', linewidth=2, label='Used')
            axes[1].plot(df.index, df['mem_total_gb'], color='gray', linewidth=2,
                        linestyle='--', label='Total')
            axes[1].set_xlabel('Sample', fontsize=12)
            axes[1].set_ylabel('Memory (GB)', fontsize=12)
            axes[1].set_title('System Memory Usage Over Time', fontsize=13)
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, f'{model_name}_memory_{device}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {output_path}")
        plt.close(fig)

    def plot_all_results(self, results_dir: str, model_name: str, benchmark_type: str = "vllm"):
        """Plot all CSV results in a directory"""
        csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]

        for csv_file in csv_files:
            csv_path = os.path.join(results_dir, csv_file)

            if 'memory' in csv_file:
                device = "gpu" if "gpu" in csv_file else "cpu"
                self.plot_memory_usage(csv_path, model_name, device)
            elif benchmark_type == "vllm":
                self.plot_vllm_results(csv_path, model_name)
            elif benchmark_type == "image":
                self.plot_image_results(csv_path, model_name)
