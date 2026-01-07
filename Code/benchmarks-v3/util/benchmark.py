import time
from typing import Dict, List, Any, Optional
from pathlib import Path
from vllm import LLM, SamplingParams
from util.result import Result
from util.config import (
    DEFAULT_IS_COLD_START,
    DEFAULT_NUM_INPUT_TOKENS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_SM_PERCENTAGE,
    DEFAULT_THREAD_NR,
    DEFAULT_MEMORY_UTIL,
    NUM_INPUT_TOKENS_GPU,
    NUM_INPUT_TOKENS_CPU,
    BATCH_SIZES_CPU,
    BATCH_SIZES_GPU,
    THREAD_NRS,
    SM_PERCENTAGES,
    MEMORY_UTILS,
    NR_ITERATIONS,
    NR_WARUMUP_ITERATIONS,
    BENCHMARK_MODES,
    RESULTS_EXECUTION,
    PARAMETERS_VLLM_GPU,
    PARAMETERS_VLLM_CPU,
    RESULTS_DIR,
)
from util.watcher import start_watcher, stop_watcher


class VLLMBenchmark:
    """
    Benchmark class for executing vLLM inference with various configurations.
    Supports GPU and CPU execution with different benchmark modes.
    """

    def __init__(
        self,
        model_name: str,
        execution_location: str = "gpu",
        results_dir: str = RESULTS_DIR,
    ):
        """
        Initialize the vLLM benchmark.

        Args:
            model_name: Name/path of the model (e.g., "meta-llama/Llama-2-7b-hf")
            execution_location: "gpu" or "cpu"
            results_dir: Directory to save results
        """
        self.model_name = model_name
        self.execution_location = execution_location.lower()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.watcher = None  # Will be set by run_benchmark_mode

        # Initialize LLM instance as None (lazy loading)
        self.llm: Optional[LLM] = None

        # Select appropriate parameter schema
        if self.execution_location == "gpu":
            self.parameter_schema = PARAMETERS_VLLM_GPU
            self.num_input_tokens_options = NUM_INPUT_TOKENS_GPU
            self.batch_size_options = BATCH_SIZES_GPU
        else:
            self.parameter_schema = PARAMETERS_VLLM_CPU
            self.num_input_tokens_options = NUM_INPUT_TOKENS_CPU
            self.batch_size_options = BATCH_SIZES_CPU

    def _load_model(
        self,
        gpu_memory_utilization: float = DEFAULT_MEMORY_UTIL,
        tensor_parallel_size: int = 1,
    ):
        """Load the vLLM model with specified configuration."""
        if self.execution_location == "gpu":
            self.llm = LLM(
                model=self.model_name,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,
            )
        else:
            # CPU execution
            self.llm = LLM(
                model=self.model_name,
                device="cpu",
            )

    def _unload_model(self):
        """Unload the model from memory."""
        if self.llm is not None:
            del self.llm
            self.llm = None
            # Force garbage collection
            import gc
            gc.collect()

    def _generate_prompt(self, num_tokens: int) -> str:
        """Generate a prompt with approximately num_tokens tokens."""
        # Approximate: 1 token ≈ 4 characters
        base_text = "This is a sample text for benchmarking. "
        repetitions = (num_tokens * 4) // len(base_text)
        return base_text * repetitions

    def _run_inference(
        self,
        num_input_tokens: int,
        max_output_tokens: int,
        batch_size: int,
    ) -> Dict[str, Any]:
        """
        Run a single inference and collect metrics.

        Returns:
            Dictionary with execution metrics (TTFT, ITL, TPS, etc.)
        """
        # Generate prompts
        prompt = self._generate_prompt(num_input_tokens)
        prompts = [prompt] * batch_size

        # Configure sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_output_tokens,
            temperature=0.0,  # Deterministic for benchmarking
        )

        # Measure load duration (if model not loaded)
        load_start = time.perf_counter()
        if self.llm is None:
            raise RuntimeError("Model not loaded. Call _load_model first.")
        load_duration = time.perf_counter() - load_start

        # Run inference and measure time
        start_time = time.perf_counter()
        outputs = self.llm.generate(prompts, sampling_params)
        end_time = time.perf_counter()

        # Calculate metrics
        e2e_latency = end_time - start_time

        # Extract token-level metrics from outputs
        total_tokens = 0
        first_token_times = []

        for output in outputs:
            if hasattr(output, 'metrics'):
                metrics = output.metrics
                if hasattr(metrics, 'first_token_time'):
                    first_token_times.append(metrics.first_token_time)

            # Count output tokens
            if hasattr(output, 'outputs') and output.outputs:
                total_tokens += len(output.outputs[0].token_ids)

        # Calculate aggregate metrics
        ttft = sum(first_token_times) / len(first_token_times) if first_token_times else None
        throughput = (total_tokens / e2e_latency) if e2e_latency > 0 else 0

        # Inter-token latency (ITL) approximation
        if total_tokens > batch_size:
            itl = (e2e_latency - (ttft or 0)) / (total_tokens - batch_size)
        else:
            itl = None

        tps = total_tokens / e2e_latency if e2e_latency > 0 else 0

        return {
            "TTFT": ttft,
            "ITL": itl,
            "TPS": tps,
            "e2e_inference_latency": e2e_latency,
            "throughput": throughput,
            "load_duration": load_duration,
        }

    def _run_benchmark_iteration(
        self,
        is_cold_start: bool,
        num_input_tokens: int,
        max_output_tokens: int,
        batch_size: int,
        thread_percentage: Optional[int] = None,
        thread_nr: Optional[int] = None,
        memory_percentage: float = DEFAULT_MEMORY_UTIL,
    ) -> Dict[str, Any]:
        """Run a single benchmark iteration with specified parameters."""

        # Handle cold start
        if is_cold_start:
            self._unload_model()

        # Add POI tag 1: Model loading
        if self.watcher:
            self.watcher.add_poi("1")

        if self.llm is None:
            self._load_model(gpu_memory_utilization=memory_percentage)

        # Add POI tag 2: Inference start
        if self.watcher:
            self.watcher.add_poi("2")

        # Run inference
        metrics = self._run_inference(
            num_input_tokens=num_input_tokens,
            max_output_tokens=max_output_tokens,
            batch_size=batch_size,
        )

        # Add parameters to metrics
        params = {
            "model_name": self.model_name,
            "model_location": self.execution_location,
            "execution_location": self.execution_location,
            "is_cold_start": is_cold_start,
            "num_input_tokens": num_input_tokens,
            "max_output_tokens": max_output_tokens,
            "batch_size": batch_size,
            "memory_percentage": memory_percentage,
        }

        if self.execution_location == "gpu":
            params["thread_percentage"] = thread_percentage
        else:
            params["thread_nr"] = thread_nr

        return {**params, **metrics}

    def run_resource_variation(
        self,
        is_cold_start: bool = DEFAULT_IS_COLD_START,
        num_input_tokens: int = DEFAULT_NUM_INPUT_TOKENS,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        memory_percentage: float = DEFAULT_MEMORY_UTIL,
    ) -> Result:
        """
        Benchmark Mode 1: Resource variation (SM% for GPU or threads for CPU).
        """
        result = Result(self.parameter_schema + RESULTS_EXECUTION)

        if self.execution_location == "gpu":
            # Vary SM percentages
            variations = SM_PERCENTAGES
            param_name = "thread_percentage"
        else:
            # Vary thread numbers
            variations = THREAD_NRS
            param_name = "thread_nr"

        for variation in variations:
            # Warmup iterations
            for _ in range(NR_WARUMUP_ITERATIONS):
                kwargs = {
                    "is_cold_start": False,
                    "num_input_tokens": num_input_tokens,
                    "max_output_tokens": max_output_tokens,
                    "batch_size": batch_size,
                    "memory_percentage": memory_percentage,
                    param_name: variation,
                }
                self._run_benchmark_iteration(**kwargs)

            # Actual iterations
            for _ in range(NR_ITERATIONS):
                kwargs = {
                    "is_cold_start": is_cold_start,
                    "num_input_tokens": num_input_tokens,
                    "max_output_tokens": max_output_tokens,
                    "batch_size": batch_size,
                    "memory_percentage": memory_percentage,
                    param_name: variation,
                }
                metrics = self._run_benchmark_iteration(**kwargs)
                result.add_row(**metrics)

        return result

    def run_batch_size_variation(
        self,
        is_cold_start: bool = DEFAULT_IS_COLD_START,
        num_input_tokens: int = DEFAULT_NUM_INPUT_TOKENS,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        thread_percentage: int = DEFAULT_SM_PERCENTAGE,
        thread_nr: int = DEFAULT_THREAD_NR,
        memory_percentage: float = DEFAULT_MEMORY_UTIL,
    ) -> Result:
        """
        Benchmark Mode 2: Batch size variation.
        """
        result = Result(self.parameter_schema + RESULTS_EXECUTION)

        for batch_size in self.batch_size_options:
            # Warmup iterations
            for _ in range(NR_WARUMUP_ITERATIONS):
                kwargs = {
                    "is_cold_start": False,
                    "num_input_tokens": num_input_tokens,
                    "max_output_tokens": max_output_tokens,
                    "batch_size": batch_size,
                    "memory_percentage": memory_percentage,
                }
                if self.execution_location == "gpu":
                    kwargs["thread_percentage"] = thread_percentage
                else:
                    kwargs["thread_nr"] = thread_nr

                self._run_benchmark_iteration(**kwargs)

            # Actual iterations
            for _ in range(NR_ITERATIONS):
                kwargs = {
                    "is_cold_start": is_cold_start,
                    "num_input_tokens": num_input_tokens,
                    "max_output_tokens": max_output_tokens,
                    "batch_size": batch_size,
                    "memory_percentage": memory_percentage,
                }
                if self.execution_location == "gpu":
                    kwargs["thread_percentage"] = thread_percentage
                else:
                    kwargs["thread_nr"] = thread_nr

                metrics = self._run_benchmark_iteration(**kwargs)
                result.add_row(**metrics)

        return result

    def run_input_length_variation(
        self,
        is_cold_start: bool = DEFAULT_IS_COLD_START,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        thread_percentage: int = DEFAULT_SM_PERCENTAGE,
        thread_nr: int = DEFAULT_THREAD_NR,
        memory_percentage: float = DEFAULT_MEMORY_UTIL,
    ) -> Result:
        """
        Benchmark Mode 3: Input length variation.
        """
        result = Result(self.parameter_schema + RESULTS_EXECUTION)

        for num_input_tokens in self.num_input_tokens_options:
            # Warmup iterations
            for _ in range(NR_WARUMUP_ITERATIONS):
                kwargs = {
                    "is_cold_start": False,
                    "num_input_tokens": num_input_tokens,
                    "max_output_tokens": max_output_tokens,
                    "batch_size": batch_size,
                    "memory_percentage": memory_percentage,
                }
                if self.execution_location == "gpu":
                    kwargs["thread_percentage"] = thread_percentage
                else:
                    kwargs["thread_nr"] = thread_nr

                self._run_benchmark_iteration(**kwargs)

            # Actual iterations
            for _ in range(NR_ITERATIONS):
                kwargs = {
                    "is_cold_start": is_cold_start,
                    "num_input_tokens": num_input_tokens,
                    "max_output_tokens": max_output_tokens,
                    "batch_size": batch_size,
                    "memory_percentage": memory_percentage,
                }
                if self.execution_location == "gpu":
                    kwargs["thread_percentage"] = thread_percentage
                else:
                    kwargs["thread_nr"] = thread_nr

                metrics = self._run_benchmark_iteration(**kwargs)
                result.add_row(**metrics)

        return result

    def run_memory_util_variation(
        self,
        is_cold_start: bool = DEFAULT_IS_COLD_START,
        num_input_tokens: int = DEFAULT_NUM_INPUT_TOKENS,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        thread_percentage: int = DEFAULT_SM_PERCENTAGE,
    ) -> Result:
        """
        Benchmark Mode 4: GPU memory utilization variation (GPU only).
        """
        if self.execution_location != "gpu":
            raise ValueError("Memory utilization benchmarking is only supported on GPU")

        result = Result(self.parameter_schema + RESULTS_EXECUTION)

        for memory_util in MEMORY_UTILS:
            # Warmup iterations
            for _ in range(NR_WARUMUP_ITERATIONS):
                kwargs = {
                    "is_cold_start": False,
                    "num_input_tokens": num_input_tokens,
                    "max_output_tokens": max_output_tokens,
                    "batch_size": batch_size,
                    "thread_percentage": thread_percentage,
                    "memory_percentage": memory_util,
                }
                self._run_benchmark_iteration(**kwargs)

            # Actual iterations
            for _ in range(NR_ITERATIONS):
                kwargs = {
                    "is_cold_start": is_cold_start,
                    "num_input_tokens": num_input_tokens,
                    "max_output_tokens": max_output_tokens,
                    "batch_size": batch_size,
                    "thread_percentage": thread_percentage,
                    "memory_percentage": memory_util,
                }
                metrics = self._run_benchmark_iteration(**kwargs)
                result.add_row(**metrics)

        return result

    def run_all_benchmarks(
        self,
        is_cold_start: bool = DEFAULT_IS_COLD_START,
    ) -> Dict[str, Result]:
        """
        Benchmark Mode 5: Run all applicable benchmarks.

        Returns:
            Dictionary mapping benchmark mode name to Result object
        """
        results = {}

        # Mode 1: Resource variation
        results["resource_variation"] = self.run_resource_variation(
            is_cold_start=is_cold_start
        )

        # Mode 2: Batch size variation
        results["batch_size"] = self.run_batch_size_variation(
            is_cold_start=is_cold_start
        )

        # Mode 3: Input length variation
        results["input_length"] = self.run_input_length_variation(
            is_cold_start=is_cold_start
        )

        # Mode 4: Memory utilization (GPU only)
        if self.execution_location == "gpu":
            results["memory_util"] = self.run_memory_util_variation(
                is_cold_start=is_cold_start
            )

        return results

    def run_benchmark_mode(
        self,
        mode: int,
        is_cold_start: bool = DEFAULT_IS_COLD_START,
        save_results: bool = True,
        monitor_resources: bool = True,
        **kwargs
    ):
        """
        Run a specific benchmark mode.

        Args:
            mode: Benchmark mode (1-5)
            is_cold_start: Whether to do cold start for each iteration
            save_results: Whether to save results to CSV
            monitor_resources: Whether to monitor GPU/CPU resources
            **kwargs: Additional parameters for specific benchmark modes

        Returns:
            Result object or dictionary of Result objects (for mode 5)
        """
        if mode not in BENCHMARK_MODES:
            raise ValueError(f"Invalid benchmark mode: {mode}. Must be 1-5.")

        mode_name = BENCHMARK_MODES[mode]

        # Start resource monitoring if enabled
        if monitor_resources:
            memory_file = self.results_dir / f"{mode_name}_memory.csv"
            self.watcher = start_watcher(
                execution_location=self.execution_location,
                save_loc=str(memory_file),
                id=0
            )

        # Load model initially (if not cold start every time)
        if not is_cold_start:
            self._load_model()

        try:
            # Run the appropriate benchmark
            if mode == 1:
                result = self.run_resource_variation(is_cold_start=is_cold_start, **kwargs)
            elif mode == 2:
                result = self.run_batch_size_variation(is_cold_start=is_cold_start, **kwargs)
            elif mode == 3:
                result = self.run_input_length_variation(is_cold_start=is_cold_start, **kwargs)
            elif mode == 4:
                result = self.run_memory_util_variation(is_cold_start=is_cold_start, **kwargs)
            elif mode == 5:
                result = self.run_all_benchmarks(is_cold_start=is_cold_start)

            # Save results if requested
            if save_results:
                if mode == 5:
                    # Save each benchmark result separately
                    for bench_name, bench_result in result.items():
                        result_file = self.results_dir / f"{bench_name}_results.csv"
                        bench_result.save_csv(str(result_file))
                else:
                    result_file = self.results_dir / f"{mode_name}_results.csv"
                    result.save_csv(str(result_file))

            return result

        finally:
            # Stop resource monitoring
            if self.watcher:
                stop_watcher(self.watcher)
                self.watcher = None

            # Cleanup
            self._unload_model()
