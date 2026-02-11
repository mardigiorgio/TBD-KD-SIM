"""Plot Newton-Raphson solver diagnostics for implicit Euler method"""

import numpy as np
import matplotlib.pyplot as plt
from implicit_euler_pendulum import ImplicitEulerPendulum

integrator = ImplicitEulerPendulum(
    initial_theta=np.pi / 4,
    initial_omega=0.0,
    dt=0.01,
    end_time=10.0
)

integrator.run(verbose=False)

# Plot 1: Newton iterations vs time
plt.figure(figsize=(10, 7))
plt.plot(integrator.time[:-1], integrator.newton_iterations_per_step, linewidth=2.5, color='#1f77b4')
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Newton iterations per step', fontsize=14)
plt.title('Newton Iterations vs Time', fontsize=15, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()

# Plot 2: Convergence - Newton iterations vs residual for first time step
plt.figure(figsize=(10, 7))
first_step_residuals = integrator.residuals_per_step[0]
iterations_range = np.arange(1, len(first_step_residuals) + 1)
plt.semilogy(iterations_range, first_step_residuals, 'o-', linewidth=2.5,
             markersize=8, color='#d62728', markerfacecolor='white',
             markeredgewidth=2.5)
plt.axhline(y=integrator.newton_tol, color='#2ca02c', linestyle='--', linewidth=2.5,
            label=f'Tolerance = {integrator.newton_tol:.0e}')
plt.xlabel('Newton iteration', fontsize=14)
plt.ylabel('Residual norm', fontsize=14)
plt.title('Newton-Raphson Convergence (First Time Step)', fontsize=15, fontweight='bold')
plt.legend(fontsize=12, loc='best')
plt.grid(True, alpha=0.3, which='both', linestyle='--')
plt.tight_layout()
plt.show()
