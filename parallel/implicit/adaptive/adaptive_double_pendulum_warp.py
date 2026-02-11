"""
Adaptive Time Stepping Double Pendulum - NVIDIA Warp GPU Parallel Version

Step-doubling error control with multiple pendulums simulated in parallel.
All pendulums use synchronized adaptive stepping (same δt based on max error).

State per pendulum: [θ₁, ω₁, θ₂, ω₂] (4 values)
"""

import numpy as np
import warp as wp
import time

wp.init()


@wp.kernel
def double_pendulum_dynamics(
    theta1: wp.array(dtype=wp.float32),
    omega1: wp.array(dtype=wp.float32),
    theta2: wp.array(dtype=wp.float32),
    omega2: wp.array(dtype=wp.float32),
    alpha1_out: wp.array(dtype=wp.float32),
    alpha2_out: wp.array(dtype=wp.float32),
    L1: wp.float32,
    L2: wp.float32,
    m1: wp.float32,
    m2: wp.float32,
    g: wp.float32,
):
    """Compute angular accelerations for double pendulum"""
    tid = wp.tid()

    t1 = theta1[tid]
    w1 = omega1[tid]
    t2 = theta2[tid]
    w2 = omega2[tid]

    delta = t1 - t2
    sin_delta = wp.sin(delta)
    cos_delta = wp.cos(delta)
    cos_2delta = wp.cos(2.0 * delta)

    denom1 = L1 * (2.0 * m1 + m2 - m2 * cos_2delta)
    denom2 = L2 * (2.0 * m1 + m2 - m2 * cos_2delta)

    num1 = (-g * (2.0 * m1 + m2) * wp.sin(t1)
            - m2 * g * wp.sin(t1 - 2.0 * t2)
            - 2.0 * sin_delta * m2 * (w2 * w2 * L2 + w1 * w1 * L1 * cos_delta))

    num2 = (2.0 * sin_delta * (w1 * w1 * L1 * (m1 + m2)
            + g * (m1 + m2) * wp.cos(t1)
            + w2 * w2 * L2 * m2 * cos_delta))

    alpha1_out[tid] = num1 / denom1
    alpha2_out[tid] = num2 / denom2


@wp.kernel
def implicit_euler_residual(
    theta1: wp.array(dtype=wp.float32),
    omega1: wp.array(dtype=wp.float32),
    theta2: wp.array(dtype=wp.float32),
    omega2: wp.array(dtype=wp.float32),
    theta1_prev: wp.array(dtype=wp.float32),
    omega1_prev: wp.array(dtype=wp.float32),
    theta2_prev: wp.array(dtype=wp.float32),
    omega2_prev: wp.array(dtype=wp.float32),
    alpha1: wp.array(dtype=wp.float32),
    alpha2: wp.array(dtype=wp.float32),
    residual: wp.array(dtype=wp.float32),
    dt: wp.float32,
):
    """Compute residual and its norm for implicit Euler"""
    tid = wp.tid()

    r0 = theta1[tid] - theta1_prev[tid] - dt * omega1[tid]
    r1 = omega1[tid] - omega1_prev[tid] - dt * alpha1[tid]
    r2 = theta2[tid] - theta2_prev[tid] - dt * omega2[tid]
    r3 = omega2[tid] - omega2_prev[tid] - dt * alpha2[tid]

    residual[tid] = wp.sqrt(r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3)


@wp.kernel
def compute_error(
    theta1_full: wp.array(dtype=wp.float32),
    omega1_full: wp.array(dtype=wp.float32),
    theta2_full: wp.array(dtype=wp.float32),
    omega2_full: wp.array(dtype=wp.float32),
    theta1_half: wp.array(dtype=wp.float32),
    omega1_half: wp.array(dtype=wp.float32),
    theta2_half: wp.array(dtype=wp.float32),
    omega2_half: wp.array(dtype=wp.float32),
    error_out: wp.array(dtype=wp.float32),
):
    """Compute error between full step and two half steps"""
    tid = wp.tid()

    d0 = theta1_full[tid] - theta1_half[tid]
    d1 = omega1_full[tid] - omega1_half[tid]
    d2 = theta2_full[tid] - theta2_half[tid]
    d3 = omega2_full[tid] - omega2_half[tid]

    error_out[tid] = wp.sqrt(d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3)


@wp.kernel
def copy_state(
    src_t1: wp.array(dtype=wp.float32),
    src_w1: wp.array(dtype=wp.float32),
    src_t2: wp.array(dtype=wp.float32),
    src_w2: wp.array(dtype=wp.float32),
    dst_t1: wp.array(dtype=wp.float32),
    dst_w1: wp.array(dtype=wp.float32),
    dst_t2: wp.array(dtype=wp.float32),
    dst_w2: wp.array(dtype=wp.float32),
):
    """Copy state arrays"""
    tid = wp.tid()
    dst_t1[tid] = src_t1[tid]
    dst_w1[tid] = src_w1[tid]
    dst_t2[tid] = src_t2[tid]
    dst_w2[tid] = src_w2[tid]


@wp.kernel
def explicit_euler_step(
    theta1: wp.array(dtype=wp.float32),
    omega1: wp.array(dtype=wp.float32),
    theta2: wp.array(dtype=wp.float32),
    omega2: wp.array(dtype=wp.float32),
    alpha1: wp.array(dtype=wp.float32),
    alpha2: wp.array(dtype=wp.float32),
    dt: wp.float32,
):
    """Explicit Euler update (used as initial guess for implicit)"""
    tid = wp.tid()
    theta1[tid] = theta1[tid] + dt * omega1[tid]
    omega1[tid] = omega1[tid] + dt * alpha1[tid]
    theta2[tid] = theta2[tid] + dt * omega2[tid]
    omega2[tid] = omega2[tid] + dt * alpha2[tid]


class AdaptiveDoublePendulumWarp:
    def __init__(self, num_pendulums, initial_theta1, initial_omega1,
                 initial_theta2, initial_omega2, epsilon_acc,
                 end_time=10.0, initial_dt=0.1,
                 gravity=9.81, length1=1.0, length2=1.0,
                 mass1=1.0, mass2=1.0):
        """
        Initialize adaptive double pendulum simulator with WARP GPU acceleration

        Parameters:
            num_pendulums: Number of pendulums to simulate in parallel
            initial_theta1: Initial angles of first pendulum (array or scalar)
            initial_omega1: Initial angular velocities of first pendulum
            initial_theta2: Initial angles of second pendulum
            initial_omega2: Initial angular velocities of second pendulum
            epsilon_acc: Target accuracy ε_acc for error control
            end_time: Simulation end time (s)
            initial_dt: Initial time step size δt₀ (s)
            gravity: Gravitational acceleration g (m/s²)
            length1: First pendulum length L₁ (m)
            length2: Second pendulum length L₂ (m)
            mass1: First pendulum mass m₁ (kg)
            mass2: Second pendulum mass m₂ (kg)
        """
        self.num_pendulums = num_pendulums
        self.epsilon_acc = epsilon_acc
        self.end_time = end_time
        self.initial_dt = initial_dt
        self.gravity = np.float32(gravity)
        self.length1 = np.float32(length1)
        self.length2 = np.float32(length2)
        self.mass1 = np.float32(mass1)
        self.mass2 = np.float32(mass2)

        self.error_order = 1
        self.newton_tol = 1e-8
        self.newton_max_iter = 15

        # Convert initial conditions to arrays
        if np.isscalar(initial_theta1):
            initial_theta1 = np.full(num_pendulums, initial_theta1, dtype=np.float32)
        if np.isscalar(initial_omega1):
            initial_omega1 = np.full(num_pendulums, initial_omega1, dtype=np.float32)
        if np.isscalar(initial_theta2):
            initial_theta2 = np.full(num_pendulums, initial_theta2, dtype=np.float32)
        if np.isscalar(initial_omega2):
            initial_omega2 = np.full(num_pendulums, initial_omega2, dtype=np.float32)

        self.initial_theta1 = np.asarray(initial_theta1, dtype=np.float32)
        self.initial_omega1 = np.asarray(initial_omega1, dtype=np.float32)
        self.initial_theta2 = np.asarray(initial_theta2, dtype=np.float32)
        self.initial_omega2 = np.asarray(initial_omega2, dtype=np.float32)

        # Simulation results (per pendulum history)
        self.time = []
        self.states_theta1 = []
        self.states_omega1 = []
        self.states_theta2 = []
        self.states_omega2 = []
        self.dt_history = []
        self.errors = []
        self.accepted_steps = 0
        self.rejected_steps = 0
        self.wall_clock_time = 0.0

    def _implicit_euler_step_gpu(self, theta1, omega1, theta2, omega2, dt):
        """Implicit Euler step using fixed-point iteration on GPU"""
        print("\rStep Size: " + str(dt) + "    ", end="", flush=True)
        n = self.num_pendulums

        # Store previous state
        theta1_prev = wp.array(theta1.numpy(), dtype=wp.float32)
        omega1_prev = wp.array(omega1.numpy(), dtype=wp.float32)
        theta2_prev = wp.array(theta2.numpy(), dtype=wp.float32)
        omega2_prev = wp.array(omega2.numpy(), dtype=wp.float32)

        # Work arrays
        alpha1 = wp.zeros(n, dtype=wp.float32)
        alpha2 = wp.zeros(n, dtype=wp.float32)
        residual = wp.zeros(n, dtype=wp.float32)

        # Initial guess: explicit Euler
        wp.launch(double_pendulum_dynamics, dim=n,
                  inputs=[theta1, omega1, theta2, omega2, alpha1, alpha2,
                          self.length1, self.length2, self.mass1, self.mass2, self.gravity])
        wp.launch(explicit_euler_step, dim=n,
                  inputs=[theta1, omega1, theta2, omega2, alpha1, alpha2, np.float32(dt)])

        # Fixed-point iteration
        for _ in range(self.newton_max_iter):
            wp.launch(double_pendulum_dynamics, dim=n,
                      inputs=[theta1, omega1, theta2, omega2, alpha1, alpha2,
                              self.length1, self.length2, self.mass1, self.mass2, self.gravity])

            wp.launch(implicit_euler_residual, dim=n,
                      inputs=[theta1, omega1, theta2, omega2,
                              theta1_prev, omega1_prev, theta2_prev, omega2_prev,
                              alpha1, alpha2, residual, np.float32(dt)])

            wp.synchronize()
            max_residual = np.max(residual.numpy())

            if max_residual < self.newton_tol:
                break

            # Update using simple fixed-point
            theta1_np = theta1_prev.numpy() + dt * omega1.numpy()
            omega1_np = omega1_prev.numpy() + dt * alpha1.numpy()
            theta2_np = theta2_prev.numpy() + dt * omega2.numpy()
            omega2_np = omega2_prev.numpy() + dt * alpha2.numpy()

            theta1 = wp.array(theta1_np.astype(np.float32), dtype=wp.float32)
            omega1 = wp.array(omega1_np.astype(np.float32), dtype=wp.float32)
            theta2 = wp.array(theta2_np.astype(np.float32), dtype=wp.float32)
            omega2 = wp.array(omega2_np.astype(np.float32), dtype=wp.float32)

        return theta1, omega1, theta2, omega2

    def _calc_adjusted_step_size(self, err, dt):
        """Adjust step size with safety factor and hysteresis (cf. Drake CalcAdjustedStepSize)"""
        safety = 0.9
        min_shrink = 0.1
        max_grow = 5.0
        hysteresis_low = 0.9
        hysteresis_high = 1.2

        if np.isnan(err) or np.isinf(err):
            return dt * min_shrink

        if err == 0:
            dt_new = dt * max_grow
        else:
            dt_new = safety * dt * (self.epsilon_acc / err) ** (1.0 / self.error_order)

        if dt_new > dt:
            if dt_new < hysteresis_high * dt:
                dt_new = dt
        elif dt_new < dt:
            if err <= self.epsilon_acc:
                dt_new = dt
            else:
                dt_new = min(dt_new, hysteresis_low * dt)

        dt_new = max(dt_new, dt * min_shrink)
        dt_new = min(dt_new, dt * max_grow)
        dt_new = max(dt_new, 1e-10)
        return dt_new

    def _step_doubling(self, theta1, omega1, theta2, omega2, dt):
        """Step-doubling error estimation"""
        n = self.num_pendulums

        # Full step
        t1_full = wp.array(theta1.numpy(), dtype=wp.float32)
        w1_full = wp.array(omega1.numpy(), dtype=wp.float32)
        t2_full = wp.array(theta2.numpy(), dtype=wp.float32)
        w2_full = wp.array(omega2.numpy(), dtype=wp.float32)
        t1_full, w1_full, t2_full, w2_full = self._implicit_euler_step_gpu(
            t1_full, w1_full, t2_full, w2_full, dt)

        # Two half steps
        t1_half = wp.array(theta1.numpy(), dtype=wp.float32)
        w1_half = wp.array(omega1.numpy(), dtype=wp.float32)
        t2_half = wp.array(theta2.numpy(), dtype=wp.float32)
        w2_half = wp.array(omega2.numpy(), dtype=wp.float32)
        t1_half, w1_half, t2_half, w2_half = self._implicit_euler_step_gpu(
            t1_half, w1_half, t2_half, w2_half, dt / 2)
        t1_half, w1_half, t2_half, w2_half = self._implicit_euler_step_gpu(
            t1_half, w1_half, t2_half, w2_half, dt / 2)

        # Compute error
        error_arr = wp.zeros(n, dtype=wp.float32)
        wp.launch(compute_error, dim=n,
                  inputs=[t1_full, w1_full, t2_full, w2_full,
                          t1_half, w1_half, t2_half, w2_half, error_arr])
        wp.synchronize()

        max_error = np.max(error_arr.numpy())
        dt_new = self._calc_adjusted_step_size(max_error, dt)
        return t1_half, w1_half, t2_half, w2_half, dt_new, max_error

    def run(self, verbose=False):
        """Run adaptive simulation"""
        start_time = time.perf_counter()

        theta1 = wp.array(self.initial_theta1, dtype=wp.float32)
        omega1 = wp.array(self.initial_omega1, dtype=wp.float32)
        theta2 = wp.array(self.initial_theta2, dtype=wp.float32)
        omega2 = wp.array(self.initial_omega2, dtype=wp.float32)

        current_time = 0.0
        dt = self.initial_dt

        self.time = [current_time]
        self.states_theta1 = [theta1.numpy().copy()]
        self.states_omega1 = [omega1.numpy().copy()]
        self.states_theta2 = [theta2.numpy().copy()]
        self.states_omega2 = [omega2.numpy().copy()]
        self.dt_history = [dt]
        self.errors = [0.0]
        self.accepted_steps = 0
        self.rejected_steps = 0

        while current_time < self.end_time:
            if current_time + dt > self.end_time:
                dt = self.end_time - current_time

            t1_new, w1_new, t2_new, w2_new, dt_new, error = self._step_doubling(
                theta1, omega1, theta2, omega2, dt)

            if error <= self.epsilon_acc:
                theta1 = t1_new
                omega1 = w1_new
                theta2 = t2_new
                omega2 = w2_new
                current_time += dt

                self.time.append(current_time)
                self.states_theta1.append(theta1.numpy().copy())
                self.states_omega1.append(omega1.numpy().copy())
                self.states_theta2.append(theta2.numpy().copy())
                self.states_omega2.append(omega2.numpy().copy())
                self.dt_history.append(dt)
                self.errors.append(error)

                self.accepted_steps += 1
            else:
                self.rejected_steps += 1

            dt = dt_new

        self.wall_clock_time = time.perf_counter() - start_time

        if verbose:
            self.print_summary()

        return self

    def print_summary(self):
        """Print simulation statistics"""
        total = self.accepted_steps + self.rejected_steps
        rejection_rate = 100 * self.rejected_steps / total if total > 0 else 0

        print(f"\nDouble Pendulum WARP Simulation ({self.num_pendulums} pendulums):")
        print(f"  Accepted steps: {self.accepted_steps}")
        print(f"  Rejected steps: {self.rejected_steps}")
        print(f"  Rejection rate: {rejection_rate:.1f}%")
        print(f"  Wall clock time: {self.wall_clock_time:.4f}s")
        print(f"  Final time: {self.time[-1]:.6f}s")
        print(f"  Min step size: {min(self.dt_history):.2e}s")
        print(f"  Max step size: {max(self.dt_history):.2e}s")
        print(f"  Mean step size: {np.mean(self.dt_history):.2e}s")
        print(f"  Max error: {max(self.errors):.2e}")

    def get_states_array(self, pendulum_idx=0):
        """Return states for one pendulum as [N×4] array"""
        return np.column_stack([
            [s[pendulum_idx] for s in self.states_theta1],
            [s[pendulum_idx] for s in self.states_omega1],
            [s[pendulum_idx] for s in self.states_theta2],
            [s[pendulum_idx] for s in self.states_omega2],
        ])
