"""
Simple Pendulum - Implicit Euler Method with Newton-Raphson solver
"""

import numpy as np
import matplotlib.pyplot as plt

# Physical constants
GRAVITY = 9.81
PENDULUM_LENGTH = 1.0

# Simulation parameters
TIME_STEP = 0.01
END_TIME = 10.0
NEWTON_MAX_ITERATIONS = 10
NEWTON_TOLERANCE = 1e-10


def pendulum_dynamics(state):
    """Compute state derivative: [theta_dot, omega_dot]"""
    theta, omega = state
    theta_dot = omega
    omega_dot = -(GRAVITY / PENDULUM_LENGTH) * np.sin(theta)
    return np.array([theta_dot, omega_dot])


def pendulum_jacobian(state):
    """Compute Jacobian matrix of pendulum dynamics"""
    theta, omega = state
    return np.array([
        [0, 1],
        [-(GRAVITY / PENDULUM_LENGTH) * np.cos(theta), 0]
    ])


def implicit_euler_step(state_current):
    """Perform one implicit Euler step using Newton's method"""
    state_next_guess = state_current.copy()

    for iteration in range(NEWTON_MAX_ITERATIONS):
        residual = state_next_guess - state_current - TIME_STEP * pendulum_dynamics(state_next_guess)
        residual_jacobian = np.eye(2) - TIME_STEP * pendulum_jacobian(state_next_guess)
        delta = np.linalg.solve(residual_jacobian, -residual)
        state_next_guess = state_next_guess + delta

        if np.linalg.norm(residual) < NEWTON_TOLERANCE:
            break

    return state_next_guess


# Initial conditions
initial_theta = np.pi / 4
initial_omega = 0.0
state = np.array([initial_theta, initial_omega])

# Run simulation
current_time = 0.0
state_history = [state.copy()]

while current_time < END_TIME:
    state = implicit_euler_step(state)
    current_time += TIME_STEP
    state_history.append(state.copy())

state_history = np.array(state_history)
time_array = np.arange(len(state_history)) * TIME_STEP

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(time_array, state_history[:, 0], label='Angle (theta)', linewidth=2)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Angle (rad)', fontsize=12)
plt.title('Simple Pendulum - Implicit Euler Method', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()