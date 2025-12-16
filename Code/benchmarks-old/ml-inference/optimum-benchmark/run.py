from optimum_benchmark import Benchmark, BenchmarkConfig, TorchrunConfig, InferenceConfig, PyTorchConfig
from optimum_benchmark.logging_utils import setup_logging


import rmm
from rmm.allocators.torch import rmm_torch_allocator

rmm.reinitialize(pool_allocator=True, managed_memory=True)
import torch
torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

setup_logging(level="INFO")

if __name__ == "__main__":
    launcher_config = TorchrunConfig(nproc_per_node=2)
    scenario_config = InferenceConfig(latency=True, memory=True, input_shapes={"batch_size": 1, "sequence_length": 16})
    backend_config = PyTorchConfig(model="gpt2", device="cuda", device_ids="0,1", no_weights=True)
    benchmark_config = BenchmarkConfig(
        name="pytorch_gpt2",
        scenario=scenario_config,
        launcher=launcher_config,
        backend=backend_config,
    )
    benchmark_report = Benchmark.launch(benchmark_config)
    benchmark_config.save_json("benchmark_config.json")
    benchmark_report.save_json("benchmark_report.json")

    benchmark = Benchmark(config=benchmark_config, report=benchmark_report)
    benchmark.save_json("benchmark.json")
