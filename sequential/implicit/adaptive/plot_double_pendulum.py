"""Plot double pendulum trajectory and adaptive time stepping diagnostics"""

import numpy as np
import matplotlib.pyplot as plt
from adaptive_double_pendulum import AdaptiveDoublePendulum

integrator = AdaptiveDoublePendulum(
    initial_theta1=np.pi / 4,
    initial_omega1=0.0,
    initial_theta2=np.pi / 2,
    initial_omega2=0.0,
    epsilon_acc=1e-5,
    end_time=10.0
)

integrator.run(verbose=True)

states = integrator.get_states_array()

# Plot 1: Both angles vs time
plt.figure(figsize=(10, 7))
plt.plot(integrator.time, states[:, 0], linewidth=2.5, color='#1f77b4', label='θ₁')
plt.plot(integrator.time, states[:, 2], linewidth=2.5, color='#ff7f0e', label='θ₂')
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Angle (rad)', fontsize=14)
plt.title('Double Pendulum Angles Over Time', fontsize=15, fontweight='bold')
plt.legend(fontsize=12, loc='best')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()

# Plot 2: Adaptive step size
plt.figure(figsize=(10, 7))
plt.semilogy(integrator.time, integrator.dt_history, linewidth=2.5, color='#ff7f0e')
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Step size dt (s)', fontsize=14)
plt.title('Adaptive Step Size Over Time', fontsize=15, fontweight='bold')
plt.grid(True, alpha=0.3, which='both', linestyle='--')
plt.tight_layout()
plt.show()

# Plot 3: Local truncation error
plt.figure(figsize=(10, 7))
plt.semilogy(integrator.time, integrator.errors, linewidth=2.5, color='#d62728',
             label='Error estimate')
plt.axhline(y=integrator.epsilon_acc, color='#2ca02c', linestyle='--', linewidth=2.5,
            label=f'Target accuracy = {integrator.epsilon_acc:.0e}')
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Error estimate', fontsize=14)
plt.title('Local Truncation Error', fontsize=15, fontweight='bold')
plt.legend(fontsize=12, loc='best')
plt.grid(True, alpha=0.3, which='both', linestyle='--')
plt.tight_layout()
plt.show()

# Plot 4: Phase space (θ₁ vs θ₂)
plt.figure(figsize=(10, 7))
plt.plot(states[:, 0], states[:, 2], linewidth=1.5, color='#9467bd')
plt.xlabel('θ₁ (rad)', fontsize=14)
plt.ylabel('θ₂ (rad)', fontsize=14)
plt.title('Double Pendulum Phase Space', fontsize=15, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()
