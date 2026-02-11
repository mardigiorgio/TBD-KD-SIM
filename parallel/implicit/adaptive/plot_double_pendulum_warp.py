"""Plot double pendulum WARP trajectories - 5 pendulums per window"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Interactive backend
import matplotlib.pyplot as plt
from adaptive_double_pendulum_warp import AdaptiveDoublePendulumWarp

# 5 pendulums with slightly different initial conditions
num = 5
theta1_init = np.array([np.pi/4 + i * 0.05 for i in range(num)], dtype=np.float32)
theta2_init = np.array([np.pi/2 + i * 0.05 for i in range(num)], dtype=np.float32)

integrator = AdaptiveDoublePendulumWarp(
    num_pendulums=num,
    initial_theta1=theta1_init,
    initial_omega1=0.0,
    initial_theta2=theta2_init,
    initial_omega2=0.0,
    epsilon_acc=1e-4,
    end_time=5.0,
    initial_dt=0.1
)

integrator.run(verbose=True)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Plot 1: θ₁ for all 5 pendulums
plt.figure(figsize=(10, 7))
for i in range(num):
    states = integrator.get_states_array(i)
    plt.plot(integrator.time, states[:, 0], linewidth=2.0, color=colors[i],
             label=f'Pendulum {i+1}')
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('θ₁ (rad)', fontsize=14)
plt.title('Double Pendulum θ₁ Over Time (5 Pendulums)', fontsize=15, fontweight='bold')
plt.legend(fontsize=11, loc='best')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()

# Plot 2: θ₂ for all 5 pendulums
plt.figure(figsize=(10, 7))
for i in range(num):
    states = integrator.get_states_array(i)
    plt.plot(integrator.time, states[:, 2], linewidth=2.0, color=colors[i],
             label=f'Pendulum {i+1}')
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('θ₂ (rad)', fontsize=14)
plt.title('Double Pendulum θ₂ Over Time (5 Pendulums)', fontsize=15, fontweight='bold')
plt.legend(fontsize=11, loc='best')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()

# Plot 3: Adaptive step size
plt.figure(figsize=(10, 7))
plt.semilogy(integrator.time, integrator.dt_history, linewidth=2.5, color='#ff7f0e')
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Step size dt (s)', fontsize=14)
plt.title('Adaptive Step Size Over Time', fontsize=15, fontweight='bold')
plt.grid(True, alpha=0.3, which='both', linestyle='--')
plt.tight_layout()
plt.show()

# Plot 4: Local truncation error
plt.figure(figsize=(10, 7))
plt.semilogy(integrator.time, integrator.errors, linewidth=2.5, color='#d62728',
             label='Max error across pendulums')
plt.axhline(y=integrator.epsilon_acc, color='#2ca02c', linestyle='--', linewidth=2.5,
            label=f'Target accuracy = {integrator.epsilon_acc:.0e}')
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Error estimate', fontsize=14)
plt.title('Local Truncation Error', fontsize=15, fontweight='bold')
plt.legend(fontsize=12, loc='best')
plt.grid(True, alpha=0.3, which='both', linestyle='--')
plt.tight_layout()
plt.show()

# Plot 5: Phase space θ₁ vs θ₂ for all 5 pendulums
plt.figure(figsize=(10, 7))
for i in range(num):
    states = integrator.get_states_array(i)
    plt.plot(states[:, 0], states[:, 2], linewidth=1.5, color=colors[i],
             label=f'Pendulum {i+1}')
plt.xlabel('θ₁ (rad)', fontsize=14)
plt.ylabel('θ₂ (rad)', fontsize=14)
plt.title('Double Pendulum Phase Space (5 Pendulums)', fontsize=15, fontweight='bold')
plt.legend(fontsize=11, loc='best')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()
