"""
Unified result management for all benchmark types
"""
import pandas as pd
import os
from typing import Dict, List, Optional


class BenchmarkResult:
    """Manages benchmark results with consistent schema"""

    def __init__(self, benchmark_type: str = "generic"):
        """
        Initialize benchmark result container

        Args:
            benchmark_type: Type of benchmark ("vllm", "image", "llm")
        """
        self.benchmark_type = benchmark_type
        self.data = self._init_dataframe()

    def _init_dataframe(self) -> pd.DataFrame:
        """Initialize appropriate dataframe based on benchmark type"""
        base_columns = {
            "model_name": [],
            "model_loc": [],
            "exec_loc": [],
            "cold_start": [],
        }

        if self.benchmark_type == "vllm":
            return pd.DataFrame({
                **base_columns,
                "nr_input_tokens": [],
                "nr_batches": [],
                "thread_percentage": [],
                "memory_rate": [],
                "avg_ttft": [],
                "max_ttft": [],
                "min_ttft": [],
                "avg_throughput": [],
                "max_throughput": [],
                "min_throughput": [],
                "avg_total": [],
                "max_total": [],
                "min_total": [],
            })
        elif self.benchmark_type == "image":
            return pd.DataFrame({
                **base_columns,
                "batch_size": [],
                "thread_percentage": [],
                "avg_load_time": [],
                "max_load_time": [],
                "min_load_time": [],
                "avg_inference_time": [],
                "max_inference_time": [],
                "min_inference_time": [],
                "avg_total_time": [],
                "max_total_time": [],
                "min_total_time": [],
            })
        else:
            # Generic fallback
            return pd.DataFrame(base_columns)

    def add_datapoint(self, datapoint: pd.DataFrame):
        """Add a single datapoint to results"""
        if isinstance(datapoint, pd.DataFrame) and list(datapoint.columns) == list(self.data.columns):
            self.data = pd.concat([self.data, datapoint], ignore_index=True)
        else:
            raise ValueError(f"Invalid datapoint schema. Expected: {list(self.data.columns)}")

    def add_raw_result_vllm(self, raw_result: Dict, model_name: str, nr_input_tokens: int,
                            nr_batches: int, thread_percentage: int, memory_rate: float,
                            cold_start: bool, model_loc: str, exec_loc: str):
        """Add vLLM benchmark results"""
        n = len(raw_result["ttfts"])
        datapoint = pd.DataFrame({
            "model_name": [model_name],
            "nr_input_tokens": [nr_input_tokens],
            "nr_batches": [nr_batches],
            "thread_percentage": [thread_percentage],
            "memory_rate": [memory_rate],
            "cold_start": [cold_start],
            "model_loc": [model_loc],
            "exec_loc": [exec_loc],
            "avg_ttft": [sum(raw_result["ttfts"]) / n],
            "max_ttft": [max(raw_result["ttfts"])],
            "min_ttft": [min(raw_result["ttfts"])],
            "avg_throughput": [sum(raw_result["throughputs"]) / n],
            "max_throughput": [max(raw_result["throughputs"])],
            "min_throughput": [min(raw_result["throughputs"])],
            "avg_total": [sum(raw_result["totals"]) / n],
            "max_total": [max(raw_result["totals"])],
            "min_total": [min(raw_result["totals"])],
        })
        self.add_datapoint(datapoint)

    def add_raw_result_image(self, raw_result: Dict, model_name: str, batch_size: int,
                            thread_percentage: int, cold_start: bool,
                            model_loc: str, exec_loc: str):
        """Add image model benchmark results"""
        n = len(raw_result["load"])
        datapoint = pd.DataFrame({
            "model_name": [model_name],
            "batch_size": [batch_size],
            "thread_percentage": [thread_percentage],
            "cold_start": [cold_start],
            "model_loc": [model_loc],
            "exec_loc": [exec_loc],
            "avg_load_time": [sum(raw_result["load"]) / n],
            "max_load_time": [max(raw_result["load"])],
            "min_load_time": [min(raw_result["load"])],
            "avg_inference_time": [sum(raw_result["execute"]) / n],
            "max_inference_time": [max(raw_result["execute"])],
            "min_inference_time": [min(raw_result["execute"])],
            "avg_total_time": [sum(raw_result["totals"]) / n],
            "max_total_time": [max(raw_result["totals"])],
            "min_total_time": [min(raw_result["totals"])],
        })
        self.add_datapoint(datapoint)

    def save_to_csv(self, dir_path: str, filename: str):
        """Save results to CSV file"""
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        file_path = os.path.join(dir_path, f"{filename}.csv")
        self.data.to_csv(file_path, mode='w', header=True, index=False)
        print(f"Results saved to: {file_path}")

    def get_data(self) -> pd.DataFrame:
        """Get the results dataframe"""
        return self.data
