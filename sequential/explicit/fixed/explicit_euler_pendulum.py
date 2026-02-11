"""
Simple Pendulum - Explicit Euler Method
"""

import numpy as np
import matplotlib.pyplot as plt

# Physical constants
GRAVITY = 9.81
PENDULUM_LENGTH = 1.0

# Simulation parameters
TIME_STEP = 0.01
END_TIME = 10.0


def pendulum_dynamics(state):
    """Compute state derivative: θ̇ = ω, ω̇ = -(g/L)sin(θ)"""
    theta, omega = state
    theta_dot = omega
    omega_dot = -(GRAVITY / PENDULUM_LENGTH) * np.sin(theta)
    return np.array([theta_dot, omega_dot])


# Initial conditions: 45 degrees, at rest
initial_theta = np.pi / 4
initial_omega = 0.0
state = np.array([initial_theta, initial_omega])

current_time = 0.0
state_history = [state.copy()]

while current_time < END_TIME:
    state = state + TIME_STEP * pendulum_dynamics(state)
    current_time += TIME_STEP
    state_history.append(state.copy())

state_history = np.array(state_history)
time_array = np.arange(len(state_history)) * TIME_STEP

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(time_array, state_history[:, 0], label='Angle (theta)', linewidth=2)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Angle (rad)', fontsize=12)
plt.title('Simple Pendulum - Explicit Euler Method', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()