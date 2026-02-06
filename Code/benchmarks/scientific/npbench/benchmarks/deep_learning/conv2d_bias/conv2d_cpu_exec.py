import numpy as np
import time
import os
import csv
from pathlib import Path

# CSV output configuration
RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "results"
CSV_FILE = RESULTS_DIR / "npbench_results.csv"
CSV_COLUMNS = ["benchmark", "data_loc", "exec_loc", "num_threads", "sm_percentage",
               "iteration", "transfer_time_ms", "computation_time_ms", "total_time_ms"]

def ensure_csv_exists():
    """Create results directory and CSV file with headers if they don't exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_FILE.exists():
        with open(CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)

def append_result(benchmark, data_loc, exec_loc, num_threads, sm_percentage,
                  iteration, transfer_time_ms, computation_time_ms, total_time_ms):
    """Append a single result row to the CSV file."""
    ensure_csv_exists()
    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([benchmark, data_loc, exec_loc, num_threads, sm_percentage,
                        iteration, transfer_time_ms, computation_time_ms, total_time_ms])


# Deep learning convolutional operator (stride = 1)
def conv2d(input, weights):
    K = weights.shape[0]  # Assuming square kernel
    N = input.shape[0]
    H_out = input.shape[1] - K + 1
    W_out = input.shape[2] - K + 1
    C_out = weights.shape[3]
    output = np.empty((N, H_out, W_out, C_out), dtype=np.float32)

    # Loop structure adapted from https://github.com/SkalskiP/ILearnDeepLearning.py/blob/ba0b5ba589d4e656141995e8d1a06d44db6ce58d/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net/src/layers/convolutional.py#L88
    for i in range(H_out):
        for j in range(W_out):
            output[:, i, j, :] = np.sum(
                input[:, i:i + K, j:j + K, :, np.newaxis] *
                weights[np.newaxis, :, :, :],
                axis=(1, 2, 3),
            )

    return output


def conv2d_bias(input, weights, bias):
    return conv2d(input, weights) + bias


def run_benchmark(num_iterations=5, data_loc="cpu"):
    """
    Run conv2d benchmark on CPU.

    Args:
        num_iterations: Number of benchmark iterations
        data_loc: "cpu" for CPU-only, "gpu" for GPU->CPU transfer scenario
    """
    # Convolution parameters
    N, H, W, C_in = 32, 64, 64, 3
    K, C_out = 3, 16

    num_threads = int(os.environ.get("OMP_NUM_THREADS", 1))

    # Warmup runs
    for _ in range(2):
        input_cpu = np.random.randn(N, H, W, C_in).astype(np.float32)
        weights_cpu = np.random.randn(K, K, C_in, C_out).astype(np.float32)
        bias_cpu = np.random.randn(C_out).astype(np.float32)
        _ = conv2d_bias(input_cpu, weights_cpu, bias_cpu)

    for iteration in range(num_iterations):
        if data_loc == "gpu":
            import cupy as cp

            input_gpu = cp.random.randn(N, H, W, C_in).astype(cp.float32)
            weights_gpu = cp.random.randn(K, K, C_in, C_out).astype(cp.float32)
            bias_gpu = cp.random.randn(C_out).astype(cp.float32)

            start_time_transfer = cp.cuda.Event()
            end_time_transfer = cp.cuda.Event()
            start_time_transfer.record()

            input_cpu = cp.asnumpy(input_gpu)
            weights_cpu = cp.asnumpy(weights_gpu)
            bias_cpu = cp.asnumpy(bias_gpu)

            end_time_transfer.record()
            end_time_transfer.synchronize()
            transfer_time = cp.cuda.get_elapsed_time(start_time_transfer, end_time_transfer)
        else:
            input_cpu = np.random.randn(N, H, W, C_in).astype(np.float32)
            weights_cpu = np.random.randn(K, K, C_in, C_out).astype(np.float32)
            bias_cpu = np.random.randn(C_out).astype(np.float32)
            transfer_time = 0.0

        start_time_compute = time.perf_counter()
        result = conv2d_bias(input_cpu, weights_cpu, bias_cpu)
        end_time_compute = time.perf_counter()

        compute_time = (end_time_compute - start_time_compute) * 1000
        total_time = transfer_time + compute_time

        append_result(
            benchmark="conv2d",
            data_loc=data_loc,
            exec_loc="cpu",
            num_threads=num_threads,
            sm_percentage="",
            iteration=iteration,
            transfer_time_ms=round(transfer_time, 3),
            computation_time_ms=round(compute_time, 3),
            total_time_ms=round(total_time, 3)
        )

        print(f"[conv2d cpu] threads={num_threads} iter={iteration} "
              f"transfer={transfer_time:.3f}ms compute={compute_time:.3f}ms total={total_time:.3f}ms")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Conv2D CPU benchmark")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations")
    parser.add_argument("--data-loc", choices=["cpu", "gpu"], default="cpu",
                        help="Data location: cpu (no transfer) or gpu (GPU->CPU transfer)")
    args = parser.parse_args()

    run_benchmark(num_iterations=args.iterations, data_loc=args.data_loc)
