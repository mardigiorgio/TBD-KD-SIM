"""
Scalability and Work-Precision Benchmarks
Global dt vs Per-Thread dt Adaptive Double Pendulum

Collects timing, step counts, and energy drift data.
Results saved as .npz for separate plotting (plot_benchmarks.py).

Usage:
    python benchmark_scalability.py --scalability
    python benchmark_scalability.py --work-precision
    python benchmark_scalability.py --all
"""

import argparse
import numpy as np
import time
import sys

from adaptive_double_pendulum_warp import AdaptiveDoublePendulumWarp
from adaptive_double_pendulum_warp_pt import AdaptiveDoublePendulumWarpPT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_chaotic_initial_conditions(N, seed=42):
    """
    Generate N sets of ICs in the chaotic regime near [pi, 0].

    Base: theta1=pi, theta2=0, omega1=0, omega2=0
    Perturbation: uniform +/-0.05 rad on angles.
    """
    rng = np.random.default_rng(seed)
    theta1 = np.float32(np.pi) + rng.uniform(-0.05, 0.05, N).astype(np.float32)
    theta2 = rng.uniform(-0.05, 0.05, N).astype(np.float32)
    omega1 = np.zeros(N, dtype=np.float32)
    omega2 = np.zeros(N, dtype=np.float32)
    return theta1, omega1, theta2, omega2


def compute_hamiltonian(theta1, omega1, theta2, omega2,
                        m1=1.0, m2=1.0, L1=1.0, L2=1.0, g=9.81):
    """
    Double pendulum Hamiltonian H = T + V.

    All inputs are arrays of shape [N] (or scalars).
    """
    delta = theta1 - theta2
    T = (0.5 * m1 * L1**2 * omega1**2
         + 0.5 * m2 * (L1**2 * omega1**2 + L2**2 * omega2**2
                        + 2 * L1 * L2 * omega1 * omega2 * np.cos(delta)))
    V = -(m1 + m2) * g * L1 * np.cos(theta1) - m2 * g * L2 * np.cos(theta2)
    return T + V


def compute_energy_drift(t1_i, w1_i, t2_i, w2_i,
                         t1_f, w1_f, t2_f, w2_f):
    """Max |H_final - H_initial| across all pendulums."""
    H0 = compute_hamiltonian(t1_i.astype(np.float64), w1_i.astype(np.float64),
                             t2_i.astype(np.float64), w2_i.astype(np.float64))
    Hf = compute_hamiltonian(t1_f.astype(np.float64), w1_f.astype(np.float64),
                             t2_f.astype(np.float64), w2_f.astype(np.float64))
    return np.max(np.abs(Hf - H0))


def warmup_gpu():
    """Run a tiny simulation to JIT-compile Warp kernels before benchmarking."""
    t1, w1, t2, w2 = generate_chaotic_initial_conditions(2, seed=0)
    sim = AdaptiveDoublePendulumWarp(
        num_pendulums=2,
        initial_theta1=t1, initial_omega1=w1,
        initial_theta2=t2, initial_omega2=w2,
        epsilon_acc=1e-2, end_time=0.1, initial_dt=0.01,
        quiet=True, store_history=False,
    )
    sim.run(verbose=False)
    sim2 = AdaptiveDoublePendulumWarpPT(
        num_pendulums=2,
        initial_theta1=t1, initial_omega1=w1,
        initial_theta2=t2, initial_omega2=w2,
        epsilon_acc=1e-2, end_time=0.1, initial_dt=0.01,
        quiet=True, store_history=False,
    )
    sim2.run(verbose=False)


# ---------------------------------------------------------------------------
# Scalability benchmark (wall time vs N)
# ---------------------------------------------------------------------------

def run_scalability_benchmark(N_values=None, epsilon_acc=1e-3,
                              end_time=10.0, num_repeats=3):
    if N_values is None:
        N_values = [1, 2, 5, 10, 50, 100, 500, 1000, 2000, 5000]

    n_pts = len(N_values)
    results = dict(
        N_values=np.array(N_values),
        epsilon_acc=epsilon_acc,
        end_time=end_time,
        num_repeats=num_repeats,
        global_times=np.zeros((n_pts, num_repeats)),
        pt_times=np.zeros((n_pts, num_repeats)),
        global_accepted=np.zeros((n_pts, num_repeats)),
        global_rejected=np.zeros((n_pts, num_repeats)),
        pt_accepted_total=np.zeros((n_pts, num_repeats)),
        pt_rejected_total=np.zeros((n_pts, num_repeats)),
        pt_accepted_mean=np.zeros((n_pts, num_repeats)),
        pt_rejected_mean=np.zeros((n_pts, num_repeats)),
    )

    for i, N in enumerate(N_values):
        print(f"Scalability: N = {N}")
        for rep in range(num_repeats):
            t1, w1, t2, w2 = generate_chaotic_initial_conditions(N, seed=42 + rep)

            # --- Global dt ---
            sim_g = AdaptiveDoublePendulumWarp(
                num_pendulums=N,
                initial_theta1=t1.copy(), initial_omega1=w1.copy(),
                initial_theta2=t2.copy(), initial_omega2=w2.copy(),
                epsilon_acc=epsilon_acc, end_time=end_time,
                initial_dt=0.01,
                quiet=True, store_history=False,
            )
            sim_g.run(verbose=False)

            results['global_times'][i, rep] = sim_g.wall_clock_time
            results['global_accepted'][i, rep] = sim_g.accepted_steps
            results['global_rejected'][i, rep] = sim_g.rejected_steps

            # --- Per-thread dt ---
            sim_pt = AdaptiveDoublePendulumWarpPT(
                num_pendulums=N,
                initial_theta1=t1.copy(), initial_omega1=w1.copy(),
                initial_theta2=t2.copy(), initial_omega2=w2.copy(),
                epsilon_acc=epsilon_acc, end_time=end_time,
                initial_dt=0.01,
                quiet=True, store_history=False,
            )
            sim_pt.run(verbose=False)

            results['pt_times'][i, rep] = sim_pt.wall_clock_time
            results['pt_accepted_total'][i, rep] = np.sum(sim_pt.accepted_steps)
            results['pt_rejected_total'][i, rep] = np.sum(sim_pt.rejected_steps)
            results['pt_accepted_mean'][i, rep] = np.mean(sim_pt.accepted_steps)
            results['pt_rejected_mean'][i, rep] = np.mean(sim_pt.rejected_steps)

            print(f"  Rep {rep+1}/{num_repeats}: "
                  f"Global={sim_g.wall_clock_time:.3f}s "
                  f"({sim_g.accepted_steps} acc / {sim_g.rejected_steps} rej), "
                  f"PT={sim_pt.wall_clock_time:.3f}s "
                  f"({np.sum(sim_pt.accepted_steps)} acc / {np.sum(sim_pt.rejected_steps)} rej)")

    np.savez('benchmark_results_scalability.npz', **results)
    return results


# ---------------------------------------------------------------------------
# Work-precision benchmark (wall time vs epsilon)
# ---------------------------------------------------------------------------

def run_work_precision_benchmark(epsilon_values=None, N=100,
                                 end_time=10.0, num_repeats=3):
    if epsilon_values is None:
        epsilon_values = [1e-1, 1e-2, 1e-3, 1e-4]

    n_pts = len(epsilon_values)
    results = dict(
        epsilon_values=np.array(epsilon_values),
        N=N,
        end_time=end_time,
        num_repeats=num_repeats,
        global_times=np.zeros((n_pts, num_repeats)),
        pt_times=np.zeros((n_pts, num_repeats)),
        global_accepted=np.zeros((n_pts, num_repeats)),
        global_rejected=np.zeros((n_pts, num_repeats)),
        pt_accepted_total=np.zeros((n_pts, num_repeats)),
        pt_rejected_total=np.zeros((n_pts, num_repeats)),
        pt_accepted_mean=np.zeros((n_pts, num_repeats)),
        pt_rejected_mean=np.zeros((n_pts, num_repeats)),
        global_energy_drift=np.zeros((n_pts, num_repeats)),
        pt_energy_drift=np.zeros((n_pts, num_repeats)),
    )

    for i, eps in enumerate(epsilon_values):
        print(f"Work-precision: epsilon = {eps:.1e}, N = {N}")
        for rep in range(num_repeats):
            t1, w1, t2, w2 = generate_chaotic_initial_conditions(N, seed=42 + rep)

            # --- Global dt ---
            sim_g = AdaptiveDoublePendulumWarp(
                num_pendulums=N,
                initial_theta1=t1.copy(), initial_omega1=w1.copy(),
                initial_theta2=t2.copy(), initial_omega2=w2.copy(),
                epsilon_acc=eps, end_time=end_time,
                initial_dt=0.01,
                quiet=True, store_history=False,
            )
            sim_g.run(verbose=False)

            results['global_times'][i, rep] = sim_g.wall_clock_time
            results['global_accepted'][i, rep] = sim_g.accepted_steps
            results['global_rejected'][i, rep] = sim_g.rejected_steps
            results['global_energy_drift'][i, rep] = compute_energy_drift(
                t1, w1, t2, w2,
                sim_g.final_theta1, sim_g.final_omega1,
                sim_g.final_theta2, sim_g.final_omega2)

            # --- Per-thread dt ---
            sim_pt = AdaptiveDoublePendulumWarpPT(
                num_pendulums=N,
                initial_theta1=t1.copy(), initial_omega1=w1.copy(),
                initial_theta2=t2.copy(), initial_omega2=w2.copy(),
                epsilon_acc=eps, end_time=end_time,
                initial_dt=0.01,
                quiet=True, store_history=False,
            )
            sim_pt.run(verbose=False)

            results['pt_times'][i, rep] = sim_pt.wall_clock_time
            results['pt_accepted_total'][i, rep] = np.sum(sim_pt.accepted_steps)
            results['pt_rejected_total'][i, rep] = np.sum(sim_pt.rejected_steps)
            results['pt_accepted_mean'][i, rep] = np.mean(sim_pt.accepted_steps)
            results['pt_rejected_mean'][i, rep] = np.mean(sim_pt.rejected_steps)
            results['pt_energy_drift'][i, rep] = compute_energy_drift(
                t1, w1, t2, w2,
                sim_pt.final_theta1, sim_pt.final_omega1,
                sim_pt.final_theta2, sim_pt.final_omega2)

            print(f"  Rep {rep+1}/{num_repeats}: "
                  f"Global={sim_g.wall_clock_time:.3f}s "
                  f"(E_drift={results['global_energy_drift'][i,rep]:.2e}), "
                  f"PT={sim_pt.wall_clock_time:.3f}s "
                  f"(E_drift={results['pt_energy_drift'][i,rep]:.2e})")

    np.savez('benchmark_results_work_precision.npz', **results)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Scalability and work-precision benchmarks for adaptive double pendulum')
    parser.add_argument('--scalability', action='store_true',
                        help='Run scalability benchmark (wall time vs N)')
    parser.add_argument('--work-precision', action='store_true',
                        help='Run work-precision benchmark (wall time vs epsilon)')
    parser.add_argument('--all', action='store_true',
                        help='Run all benchmarks')
    parser.add_argument('--end-time', type=float, default=10.0,
                        help='Simulation end time (default: 10.0)')
    parser.add_argument('--repeats', type=int, default=3,
                        help='Repetitions per configuration (default: 3)')
    args = parser.parse_args()

    if args.all:
        args.scalability = True
        args.work_precision = True

    if not (args.scalability or args.work_precision):
        args.scalability = True

    warmup_gpu()

    if args.scalability:
        run_scalability_benchmark(end_time=args.end_time, num_repeats=args.repeats)

    if args.work_precision:
        run_work_precision_benchmark(end_time=args.end_time, num_repeats=args.repeats)


if __name__ == '__main__':
    main()
