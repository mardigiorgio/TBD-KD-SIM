"""
Simple Pendulum - GPU Parallel Version using NVIDIA Warp
Simulates multiple pendulums using GPU acceleration (requires NVIDIA GPU)
"""

import time
import warp as wp
import numpy as np

wp.init()

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
    print(f"Initial angles (rad): {angles}")
    return angles


@wp.kernel
def explicit_euler_step_parallel(
    theta: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    omega: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    length: wp.float32,
    gravity: wp.float32,
    dt: wp.float32,
):
    """GPU kernel: explicit Euler step with one thread per pendulum"""
    thread_id = wp.tid()
    current_theta = theta[thread_id]
    current_omega = omega[thread_id]

    angular_acceleration = -(gravity / length) * wp.sin(current_theta)

    omega[thread_id] = current_omega + angular_acceleration * dt
    theta[thread_id] = current_theta + current_omega * dt


def run_simulation() -> None:
    """Run GPU-accelerated simulation and benchmark performance"""
    initial_angles = initialize_angles(NUM_PENDULUMS)
    initial_velocities = np.zeros(NUM_PENDULUMS, dtype=np.float32)

    theta = wp.array(initial_angles, dtype=float)
    omega = wp.array(initial_velocities, dtype=float)

    print(f"Simulating {NUM_PENDULUMS} pendulums in parallel (GPU)")
    print(f"Time step: {TIME_STEP}s, Steps: {NUM_STEPS}")
    print("-" * 50)

    start_time = time.perf_counter()

    for step in range(NUM_STEPS):
        wp.launch(
            kernel=explicit_euler_step_parallel,
            dim=NUM_PENDULUMS,
            inputs=[theta, omega, PENDULUM_LENGTH, GRAVITY, TIME_STEP],
        )

    wp.synchronize()
    elapsed_time = time.perf_counter() - start_time

    print(f"Final angles (rad): {theta.numpy()}")
    print(f"Final velocities (rad/s): {omega.numpy()}")
    print("-" * 50)
    print(f"Elapsed time: {elapsed_time:.4f}s")


if __name__ == "__main__":
    run_simulation()
