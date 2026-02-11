"""Generate work-precision diagram comparing computational cost vs accuracy"""

import numpy as np
import matplotlib.pyplot as plt
from adaptive_pendulum import AdaptivePendulum


accuracy_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
newton_iterations = []
wall_clock_times = []

for eps in accuracy_values:
    integrator = AdaptivePendulum(
        initial_theta=np.pi / 4,
        initial_omega=0.0,
        epsilon_acc=eps,
        end_time=10.0,
        initial_dt=0.1
    )

    integrator.run(verbose=False)

    newton_iterations.append(integrator.total_newton_iterations)
    wall_clock_times.append(integrator.wall_clock_time)

    print(f"{eps:.1e}: {integrator.total_newton_iterations:5d} Newton iterations, "
          f"{integrator.wall_clock_time:.4f}s wall clock time")

# Plot 1: Newton iterations vs accuracy
plt.figure(figsize=(10, 7))
plt.loglog(accuracy_values, newton_iterations, 'o-', linewidth=2.5,
           markersize=8, color='#1f77b4', markerfacecolor='white',
           markeredgewidth=2.5, label='Implicit Euler')
plt.xlabel('Accuracy', fontsize=14)
plt.ylabel('Total Newton iterations', fontsize=14)
plt.title('Newton Iterations vs Accuracy', fontsize=15, fontweight='bold')
plt.gca().invert_xaxis()
plt.grid(True, alpha=0.3, which='both', linestyle='--')
plt.legend(fontsize=12, loc='best')
plt.tight_layout()
plt.show()

# Plot 2: Wall clock time vs accuracy
plt.figure(figsize=(10, 7))
plt.loglog(accuracy_values, wall_clock_times, 'd-', linewidth=2.5,
           markersize=8, color='#2ca02c', markerfacecolor='white',
           markeredgewidth=2.5)
plt.xlabel('Accuracy', fontsize=14)
plt.ylabel('Wall Clock Time (s)', fontsize=14)
plt.title('Wall Clock Time vs Accuracy', fontsize=15, fontweight='bold')
plt.gca().invert_xaxis()
plt.grid(True, alpha=0.3, which='both', linestyle='--')
plt.tight_layout()
plt.show()
