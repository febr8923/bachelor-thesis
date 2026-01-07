"""Benchmark runners"""
from .vllm_benchmark import VLLMBenchmark
from .image_benchmark import ImageBenchmark

__all__ = ['VLLMBenchmark', 'ImageBenchmark']
