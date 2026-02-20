"""
Benchmark Visualization: Scalability and Work-Precision Plots

Loads .npz files produced by benchmark_scalability.py and generates:
  1. Scalability plot (wall time vs N)
  2. Work-precision plot (wall time vs accuracy)
  3. Total steps plot (work proxy)

Usage:
    python plot_benchmarks.py
    python plot_benchmarks.py --scalability-only
    python plot_benchmarks.py --work-precision-only
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


COLOR_GLOBAL = '#1f77b4'
COLOR_PT = '#ff7f0e'


# ---------------------------------------------------------------------------
# Plot 1: Scalability (wall time vs N) — MOST IMPORTANT
# ---------------------------------------------------------------------------

def plot_scalability(data):
    N = data['N_values']
    eps = float(data['epsilon_acc'])
    end_t = float(data['end_time'])

    global_med = np.median(data['global_times'], axis=1)
    global_min = np.min(data['global_times'], axis=1)
    global_max = np.max(data['global_times'], axis=1)

    pt_med = np.median(data['pt_times'], axis=1)
    pt_min = np.min(data['pt_times'], axis=1)
    pt_max = np.max(data['pt_times'], axis=1)

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.loglog(N, global_med, 'o-', color=COLOR_GLOBAL, linewidth=2.5,
              markersize=8, markerfacecolor='white', markeredgewidth=2.5,
              label='Global dt', zorder=5)
    ax.fill_between(N, global_min, global_max, alpha=0.15, color=COLOR_GLOBAL)

    ax.loglog(N, pt_med, 's-', color=COLOR_PT, linewidth=2.5,
              markersize=8, markerfacecolor='white', markeredgewidth=2.5,
              label='Per-Thread dt', zorder=5)
    ax.fill_between(N, pt_min, pt_max, alpha=0.15, color=COLOR_PT)

    # O(N) reference line
    mid = len(N) // 2
    ref = N.astype(float)
    scale = global_med[mid] / ref[mid]
    ax.loglog(N, scale * ref, '--', color='gray', alpha=0.5,
              linewidth=1.5, label='O(N) reference')

    ax.set_xlabel('Number of Parallel Simulations (N)', fontsize=14)
    ax.set_ylabel('Wall Time (s)', fontsize=14)
    ax.set_title(f'Scalability: Wall Time vs N\n'
                 f'(eps={eps:.0e}, T={end_t:.0f}s, chaotic ICs near [pi, 0])',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig('fig_scalability.png', dpi=150, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# Plot 2: Work-Precision (wall time vs accuracy)
# ---------------------------------------------------------------------------

def plot_work_precision(data):
    eps = data['epsilon_values']
    N = int(data['N'])
    end_t = float(data['end_time'])

    global_time_med = np.median(data['global_times'], axis=1)
    pt_time_med = np.median(data['pt_times'], axis=1)

    global_drift_med = np.median(data['global_energy_drift'], axis=1)
    pt_drift_med = np.median(data['pt_energy_drift'], axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    # --- Top: wall time vs epsilon_acc ---
    ax = axes[0]
    ax.loglog(eps, global_time_med, 'o-', color=COLOR_GLOBAL, linewidth=2.5,
              markersize=8, markerfacecolor='white', markeredgewidth=2.5,
              label='Global dt')
    ax.loglog(eps, pt_time_med, 's-', color=COLOR_PT, linewidth=2.5,
              markersize=8, markerfacecolor='white', markeredgewidth=2.5,
              label='Per-Thread dt')
    ax.set_xlabel('Accuracy Target (epsilon_acc)', fontsize=14)
    ax.set_ylabel('Wall Time (s)', fontsize=14)
    ax.set_title(f'Work-Precision: Wall Time vs Accuracy Target\n'
                 f'(N={N}, T={end_t:.0f}s)',
                 fontsize=15, fontweight='bold')
    ax.invert_xaxis()
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.tick_params(labelsize=12)

    # --- Bottom: wall time vs energy drift ---
    ax = axes[1]
    ax.loglog(global_drift_med, global_time_med, 'o-', color=COLOR_GLOBAL,
              linewidth=2.5, markersize=8, markerfacecolor='white',
              markeredgewidth=2.5, label='Global dt')
    ax.loglog(pt_drift_med, pt_time_med, 's-', color=COLOR_PT,
              linewidth=2.5, markersize=8, markerfacecolor='white',
              markeredgewidth=2.5, label='Per-Thread dt')
    ax.set_xlabel('Energy Drift |H(T) - H(0)|  (achieved accuracy)', fontsize=14)
    ax.set_ylabel('Wall Time (s)', fontsize=14)
    ax.set_title('Work-Precision: Wall Time vs Energy Conservation Error',
                 fontsize=15, fontweight='bold')
    ax.invert_xaxis()
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.tick_params(labelsize=12)

    fig.tight_layout()
    fig.savefig('fig_work_precision.png', dpi=150, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# Plot 3: Total Steps (accepted + rejected)
# ---------------------------------------------------------------------------

def plot_total_steps(data_wp=None, data_sc=None):
    has_wp = data_wp is not None
    has_sc = data_sc is not None
    ncols = int(has_wp) + int(has_sc)
    if ncols == 0:
        return

    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 7))
    if ncols == 1:
        axes = [axes]
    col = 0

    # --- Left: steps vs epsilon ---
    if has_wp:
        ax = axes[col]
        col += 1
        eps = data_wp['epsilon_values']

        global_total = np.median(
            data_wp['global_accepted'] + data_wp['global_rejected'], axis=1)
        pt_mean_steps = np.median(
            data_wp['pt_accepted_mean'] + data_wp['pt_rejected_mean'], axis=1)
        pt_sum_steps = np.median(
            data_wp['pt_accepted_total'] + data_wp['pt_rejected_total'], axis=1)

        ax.loglog(eps, global_total, 'o-', color=COLOR_GLOBAL, linewidth=2.5,
                  markersize=8, markerfacecolor='white', markeredgewidth=2.5,
                  label='Global dt (shared step count)')
        ax.loglog(eps, pt_mean_steps, 's-', color=COLOR_PT, linewidth=2.5,
                  markersize=8, markerfacecolor='white', markeredgewidth=2.5,
                  label='Per-Thread dt (mean per pendulum)')
        ax.loglog(eps, pt_sum_steps, 'd--', color=COLOR_PT, linewidth=1.5,
                  markersize=6, alpha=0.6,
                  label='Per-Thread dt (sum across all)')

        ax.set_xlabel('Accuracy Target (epsilon_acc)', fontsize=14)
        ax.set_ylabel('Total Steps (Accepted + Rejected)', fontsize=14)
        ax.set_title(f'Total Steps vs Accuracy\n(N={int(data_wp["N"])})',
                     fontsize=15, fontweight='bold')
        ax.invert_xaxis()
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which='both', linestyle='--')
        ax.tick_params(labelsize=12)

    # --- Right: steps vs N ---
    if has_sc:
        ax = axes[col]
        N = data_sc['N_values']

        global_total_N = np.median(
            data_sc['global_accepted'] + data_sc['global_rejected'], axis=1)
        pt_mean_N = np.median(
            data_sc['pt_accepted_mean'] + data_sc['pt_rejected_mean'], axis=1)

        ax.loglog(N, global_total_N, 'o-', color=COLOR_GLOBAL, linewidth=2.5,
                  markersize=8, markerfacecolor='white', markeredgewidth=2.5,
                  label='Global dt')
        ax.loglog(N, pt_mean_N, 's-', color=COLOR_PT, linewidth=2.5,
                  markersize=8, markerfacecolor='white', markeredgewidth=2.5,
                  label='Per-Thread dt (mean per pendulum)')

        ax.set_xlabel('Number of Parallel Simulations (N)', fontsize=14)
        ax.set_ylabel('Total Steps (Accepted + Rejected)', fontsize=14)
        ax.set_title(f'Total Steps vs N\n(eps={float(data_sc["epsilon_acc"]):.0e})',
                     fontsize=15, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which='both', linestyle='--')
        ax.tick_params(labelsize=12)

    fig.tight_layout()
    fig.savefig('fig_total_steps.png', dpi=150, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Plot benchmark results')
    parser.add_argument('--scalability-only', action='store_true',
                        help='Only plot scalability results')
    parser.add_argument('--work-precision-only', action='store_true',
                        help='Only plot work-precision results')
    args = parser.parse_args()

    data_sc = None
    data_wp = None

    sc_file = 'benchmark_results_scalability.npz'
    wp_file = 'benchmark_results_work_precision.npz'

    if not args.work_precision_only and os.path.exists(sc_file):
        data_sc = dict(np.load(sc_file, allow_pickle=True))

    if not args.scalability_only and os.path.exists(wp_file):
        data_wp = dict(np.load(wp_file, allow_pickle=True))

    if data_sc is not None:
        plot_scalability(data_sc)

    if data_wp is not None:
        plot_work_precision(data_wp)

    if data_sc is not None or data_wp is not None:
        plot_total_steps(data_wp, data_sc)

    if data_sc is None and data_wp is None:
        print("No benchmark data found. Run benchmark_scalability.py first.")
        sys.exit(1)


if __name__ == '__main__':
    main()
