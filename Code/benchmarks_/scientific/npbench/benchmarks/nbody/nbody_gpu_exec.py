# Adapted from https://github.com/pmocz/nbody-python/blob/master/nbody.py
# TODO: Add GPL-3.0 License

import cupy as np
import time
import os
import csv
from pathlib import Path

"""
Create Your Own N-body Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz
Simulate orbits of stars interacting due to gravity
Code calculates pairwise forces according to Newton's Law of Gravity
"""

# CSV output configuration
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
CSV_FILE = RESULTS_DIR / "npbench_results.csv"
CSV_COLUMNS = ["benchmark", "data_loc", "exec_loc", "num_threads", "sm_percentage",
               "cold_start", "iteration", "transfer_time_ms", "computation_time_ms",
               "total_time_ms", "wall_time_ms"]

def ensure_csv_exists():
    """Create results directory and CSV file with headers if they don't exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_FILE.exists():
        with open(CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)

def append_result(benchmark, data_loc, exec_loc, num_threads, sm_percentage,
                  cold_start, iteration, transfer_time_ms, computation_time_ms,
                  total_time_ms, wall_time_ms):
    """Append a single result row to the CSV file."""
    ensure_csv_exists()
    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([benchmark, data_loc, exec_loc, num_threads, sm_percentage,
                        cold_start, iteration, transfer_time_ms, computation_time_ms,
                        total_time_ms, wall_time_ms])


def initialize(N, tEnd, dt):
    import numpy as cpu_np
    from numpy.random import default_rng
    rng = default_rng(42)
    mass = 20.0 * cpu_np.ones((N, 1)) / N  # total mass of particles is 20
    pos = rng.random((N, 3))  # randomly selected positions and velocities
    vel = rng.random((N, 3))
    Nt = int(cpu_np.ceil(tEnd / dt))
    return mass, pos, vel, Nt


def getAcc(pos, mass, G, softening):
    """
    Calculate the acceleration on each particle due to Newton's Law
    pos  is an N x 3 matrix of positions
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    softening is the softening length
    a is N x 3 matrix of accelerations
    """
    # positions r = [x,y,z] for all particles
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # matrix that stores 1/r^3 for all particle pairwise particle separations
    inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2)
    inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0]**(-1.5)

    ax = G * (dx * inv_r3) @ mass
    ay = G * (dy * inv_r3) @ mass
    az = G * (dz * inv_r3) @ mass

    # pack together the acceleration components
    a = np.hstack((ax, ay, az))

    return a


def getEnergy(pos, vel, mass, G):
    """
    Get kinetic energy (KE) and potential energy (PE) of simulation
    pos is N x 3 matrix of positions
    vel is N x 3 matrix of velocities
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    KE is the kinetic energy of the system
    PE is the potential energy of the system
    """
    # Kinetic Energy:
    # KE = 0.5 * np.sum(np.sum( mass * vel**2 ))
    KE = 0.5 * np.sum(mass * vel**2)

    # Potential Energy:

    # positions r = [x,y,z] for all particles
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # matrix that stores 1/r for all particle pairwise particle separations
    inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
    inv_r[inv_r > 0] = 1.0 / inv_r[inv_r > 0]

    # sum over upper triangle, to count each interaction only once
    # PE = G * np.sum(np.sum(np.triu(-(mass*mass.T)*inv_r,1)))
    PE = G * np.sum(np.triu(-(mass * mass.T) * inv_r, 1))

    return KE, PE


def nbody(mass, pos, vel, N, Nt, dt, G, softening):

    # Convert to Center-of-Mass frame
    vel -= np.mean(mass * vel, axis=0) / np.mean(mass)

    # calculate initial gravitational accelerations
    acc = getAcc(pos, mass, G, softening)

    # calculate initial energy of system
    KE = np.ndarray(Nt + 1, dtype=np.float64)
    PE = np.ndarray(Nt + 1, dtype=np.float64)
    KE[0], PE[0] = getEnergy(pos, vel, mass, G)

    t = 0.0

    # Simulation Main Loop
    for i in range(Nt):
        # (1/2) kick
        vel += acc * dt / 2.0

        # drift
        pos += vel * dt

        # update accelerations
        acc = getAcc(pos, mass, G, softening)

        # (1/2) kick
        vel += acc * dt / 2.0

        # update time
        t += dt

        # get energy of system
        KE[i + 1], PE[i + 1] = getEnergy(pos, vel, mass, G)

    return KE, PE



def run_single(N=10000, num_iterations=50, num_warmup=2, data_loc="cpu", cold_start=False):
    """
    Run nbody benchmark on GPU with fixed N.

    Args:
        N: Number of particles
        num_iterations: Number of benchmark iterations
        num_warmup: Number of warmup iterations
        data_loc: "cpu" for CPU->GPU transfer, "gpu" for GPU-only (no transfer)
        cold_start: If True, skip warmup and run single iteration
    """
    if cold_start:
        num_warmup = 0
        num_iterations = 1

    tEnd = 1.0
    dt = 0.01
    G = 1.0
    softening = 0.1

    sm_percentage = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "100")

    # Warmup runs
    for _ in range(num_warmup):
        mass_cpu, pos_cpu, vel_cpu, Nt = initialize(N, tEnd, dt)
        mass = np.asarray(mass_cpu)
        pos = np.asarray(pos_cpu)
        vel = np.asarray(vel_cpu)
        KE, PE = nbody(mass, pos, vel, N, Nt, dt, G, softening)

    for iteration in range(num_iterations):
        mass_cpu, pos_cpu, vel_cpu, Nt = initialize(N, tEnd, dt)

        wall_start = time.perf_counter()

        if data_loc == "cpu":
            # Data starts on CPU, measure transfer time to GPU
            start_time_transfer = np.cuda.Event()
            end_time_transfer = np.cuda.Event()
            start_time_transfer.record()

            mass = np.asarray(mass_cpu)
            pos = np.asarray(pos_cpu)
            vel = np.asarray(vel_cpu)

            end_time_transfer.record()
            end_time_transfer.synchronize()
            transfer_time = np.cuda.get_elapsed_time(start_time_transfer, end_time_transfer)
        else:
            mass = np.asarray(mass_cpu)
            pos = np.asarray(pos_cpu)
            vel = np.asarray(vel_cpu)
            transfer_time = 0.0

        start_time_compute = np.cuda.Event()
        end_time_compute = np.cuda.Event()
        start_time_compute.record()

        KE, PE = nbody(mass, pos, vel, N, Nt, dt, G, softening)

        end_time_compute.record()
        end_time_compute.synchronize()

        compute_time = np.cuda.get_elapsed_time(start_time_compute, end_time_compute)
        total_time = transfer_time + compute_time
        wall_time = (time.perf_counter() - wall_start) * 1000

        append_result(
            benchmark="nbody",
            data_loc=data_loc,
            exec_loc="gpu",
            num_threads="",
            sm_percentage=sm_percentage,
            cold_start=cold_start,
            iteration=iteration,
            transfer_time_ms=round(transfer_time, 3),
            computation_time_ms=round(compute_time, 3),
            total_time_ms=round(total_time, 3),
            wall_time_ms=round(wall_time, 3)
        )

        print(f"transfer={transfer_time:.3f} compute={compute_time:.3f} total={total_time:.3f}")


def sweep_n(n_range=range(100, 1100, 100), num_iterations=5, num_warmup=2, data_loc="cpu", cold_start=False):
    """
    Run nbody benchmark on GPU, sweeping over different N values.

    Args:
        n_range: Range of N values to test
        num_iterations: Number of benchmark iterations per N
        num_warmup: Number of warmup iterations
        data_loc: "cpu" for CPU->GPU transfer, "gpu" for GPU-only (no transfer)
        cold_start: If True, skip warmup and run single iteration
    """
    if cold_start:
        num_warmup = 0
        num_iterations = 1

    tEnd = 1.0
    dt = 0.01
    G = 1.0
    softening = 0.1

    sm_percentage = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "100")

    # Warmup runs
    for _ in range(num_warmup):
        mass_cpu, pos_cpu, vel_cpu, Nt = initialize(1000, tEnd, dt)
        mass = np.asarray(mass_cpu)
        pos = np.asarray(pos_cpu)
        vel = np.asarray(vel_cpu)
        KE, PE = nbody(mass, pos, vel, 1000, Nt, dt, G, softening)

    for N in n_range:
        for iteration in range(num_iterations):
            mass_cpu, pos_cpu, vel_cpu, Nt = initialize(N, tEnd, dt)

            wall_start = time.perf_counter()

            if data_loc == "cpu":
                start_time_transfer = np.cuda.Event()
                end_time_transfer = np.cuda.Event()
                start_time_transfer.record()

                mass = np.asarray(mass_cpu)
                pos = np.asarray(pos_cpu)
                vel = np.asarray(vel_cpu)

                end_time_transfer.record()
                end_time_transfer.synchronize()
                transfer_time = np.cuda.get_elapsed_time(start_time_transfer, end_time_transfer)
            else:
                mass = np.asarray(mass_cpu)
                pos = np.asarray(pos_cpu)
                vel = np.asarray(vel_cpu)
                transfer_time = 0.0

            start_time_compute = np.cuda.Event()
            end_time_compute = np.cuda.Event()
            start_time_compute.record()

            KE, PE = nbody(mass, pos, vel, N, Nt, dt, G, softening)

            end_time_compute.record()
            end_time_compute.synchronize()

            compute_time = np.cuda.get_elapsed_time(start_time_compute, end_time_compute)
            total_time = transfer_time + compute_time
            wall_time = (time.perf_counter() - wall_start) * 1000

            append_result(
                benchmark=f"nbody_N{N}",
                data_loc=data_loc,
                exec_loc="gpu",
                num_threads="",
                sm_percentage=sm_percentage,
                cold_start=cold_start,
                iteration=iteration,
                transfer_time_ms=round(transfer_time, 3),
                computation_time_ms=round(compute_time, 3),
                total_time_ms=round(total_time, 3),
                wall_time_ms=round(wall_time, 3)
            )

            print(f"transfer={transfer_time:.3f} compute={compute_time:.3f} total={total_time:.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="N-body GPU benchmark")
    parser.add_argument("--mode", choices=["single", "sweep"], default="single",
                        help="Run mode: single (fixed N) or sweep (vary N)")
    parser.add_argument("--N", type=int, default=1000, help="Number of particles (for single mode)")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup iterations")
    parser.add_argument("--data-loc", choices=["cpu", "gpu"], default="cpu",
                        help="Data location: cpu (CPU->GPU transfer) or gpu (no transfer)")
    parser.add_argument("--cold_start", action="store_true",
                        help="Cold start mode: no warmup, single iteration")
    args = parser.parse_args()

    if args.mode == "single":
        run_single(N=args.N, num_iterations=args.iterations, num_warmup=args.warmup,
                   data_loc=args.data_loc, cold_start=args.cold_start)
    else:
        sweep_n(num_iterations=args.iterations, num_warmup=args.warmup,
                data_loc=args.data_loc, cold_start=args.cold_start)
