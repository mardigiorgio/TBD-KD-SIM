"""
Simple Pendulum - Sequential CPU Version
Simulates multiple pendulums using sequential processing (baseline for comparison)
"""

import time
import numpy as np

# Simulation configuration
NUM_PENDULUMS = 10000
GRAVITY = 9.81
PENDULUM_LENGTH = 1.0
TIME_STEP = 0.001
NUM_STEPS = 1000


def initialize_angles(num_pendulums: int) -> np.ndarray:
    """Initialize pendulum angles with small variations around base_angle"""
    base_angle = 1.0
    angles = np.array(
        [base_angle + i * 0.01 for i in range(num_pendulums)],
        dtype=np.float32
    )
    return angles


def explicit_euler_step_sequential(
    theta: np.ndarray,
    omega: np.ndarray,
    length: float,
    gravity: float,
    dt: float
) -> None:
    """Perform explicit Euler step for all pendulums in sequence"""
    for i in range(len(theta)):
        angular_acceleration = -(gravity / length) * np.sin(theta[i])
        omega[i] = omega[i] + angular_acceleration * dt
        theta[i] = theta[i] + omega[i] * dt


def run_simulation() -> None:
    """Run sequential simulation and benchmark performance"""
    initial_angles = initialize_angles(NUM_PENDULUMS)
    initial_velocities = np.zeros(NUM_PENDULUMS, dtype=np.float32)

    theta = initial_angles.copy()
    omega = initial_velocities.copy()

    print(f"Simulating {NUM_PENDULUMS} pendulums sequentially")
    print(f"Time step: {TIME_STEP}s, Steps: {NUM_STEPS}")
    print("-" * 50)

    start_time = time.perf_counter()

    for step in range(NUM_STEPS):
        explicit_euler_step_sequential(theta, omega, PENDULUM_LENGTH, GRAVITY, TIME_STEP)

    elapsed_time = time.perf_counter() - start_time

    print(f"Final angles (rad): {theta}")
    print(f"Final velocities (rad/s): {omega}")
    print("-" * 50)
    print(f"Elapsed time: {elapsed_time:.4f}s")


if __name__ == "__main__":
    run_simulation()
