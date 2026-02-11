"""
Adaptive Time Stepping Double Pendulum - Per-Thread dt (NVIDIA Warp GPU)

Each pendulum independently controls its own adaptive time step.
Unlike the global-dt variant where max(error) across all pendulums
sets a single shared dt, here each pendulum advances at its own pace.

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
def explicit_euler_step_pt(
    theta1: wp.array(dtype=wp.float32),
    omega1: wp.array(dtype=wp.float32),
    theta2: wp.array(dtype=wp.float32),
    omega2: wp.array(dtype=wp.float32),
    alpha1: wp.array(dtype=wp.float32),
    alpha2: wp.array(dtype=wp.float32),
    dt: wp.array(dtype=wp.float32),
):
    """Explicit Euler update with per-thread dt"""
    tid = wp.tid()
    theta1[tid] = theta1[tid] + dt[tid] * omega1[tid]
    omega1[tid] = omega1[tid] + dt[tid] * alpha1[tid]
    theta2[tid] = theta2[tid] + dt[tid] * omega2[tid]
    omega2[tid] = omega2[tid] + dt[tid] * alpha2[tid]


@wp.kernel
def implicit_euler_residual_pt(
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
    dt: wp.array(dtype=wp.float32),
):
    """Compute residual with per-thread dt"""
    tid = wp.tid()

    r0 = theta1[tid] - theta1_prev[tid] - dt[tid] * omega1[tid]
    r1 = omega1[tid] - omega1_prev[tid] - dt[tid] * alpha1[tid]
    r2 = theta2[tid] - theta2_prev[tid] - dt[tid] * omega2[tid]
    r3 = omega2[tid] - omega2_prev[tid] - dt[tid] * alpha2[tid]

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


class AdaptiveDoublePendulumWarpPT:
    def __init__(self, num_pendulums, initial_theta1, initial_omega1,
                 initial_theta2, initial_omega2, epsilon_acc,
                 end_time=10.0, initial_dt=0.1,
                 gravity=9.81, length1=1.0, length2=1.0,
                 mass1=1.0, mass2=1.0):
        """
        Per-thread adaptive double pendulum simulator with WARP GPU acceleration.

        Each pendulum independently controls its own time step based on its
        own local truncation error, rather than being constrained by the
        worst-case pendulum across the batch.
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

        # Per-pendulum ragged storage (each list[i] grows independently)
        n = num_pendulums
        self.times = [[] for _ in range(n)]
        self.states_theta1 = [[] for _ in range(n)]
        self.states_omega1 = [[] for _ in range(n)]
        self.states_theta2 = [[] for _ in range(n)]
        self.states_omega2 = [[] for _ in range(n)]
        self.dt_histories = [[] for _ in range(n)]
        self.error_histories = [[] for _ in range(n)]
        self.accepted_steps = np.zeros(n, dtype=int)
        self.rejected_steps = np.zeros(n, dtype=int)
        self.wall_clock_time = 0.0

    def _implicit_euler_step_gpu_pt(self, theta1, omega1, theta2, omega2, dt_arr):
        """Implicit Euler step with per-thread dt array.

        dt_arr: numpy float32 array [n] — each pendulum's time step.
        """
        print("\rStep Size: " + str(dt_arr) + "    ", end="", flush=True)
        n = self.num_pendulums

        theta1_prev = wp.array(theta1.numpy(), dtype=wp.float32)
        omega1_prev = wp.array(omega1.numpy(), dtype=wp.float32)
        theta2_prev = wp.array(theta2.numpy(), dtype=wp.float32)
        omega2_prev = wp.array(omega2.numpy(), dtype=wp.float32)

        alpha1 = wp.zeros(n, dtype=wp.float32)
        alpha2 = wp.zeros(n, dtype=wp.float32)
        residual = wp.zeros(n, dtype=wp.float32)

        dt_wp = wp.array(dt_arr.astype(np.float32), dtype=wp.float32)

        # Initial guess: explicit Euler with per-thread dt
        wp.launch(double_pendulum_dynamics, dim=n,
                  inputs=[theta1, omega1, theta2, omega2, alpha1, alpha2,
                          self.length1, self.length2, self.mass1, self.mass2, self.gravity])
        wp.launch(explicit_euler_step_pt, dim=n,
                  inputs=[theta1, omega1, theta2, omega2, alpha1, alpha2, dt_wp])

        # Fixed-point iteration
        for _ in range(self.newton_max_iter):
            wp.launch(double_pendulum_dynamics, dim=n,
                      inputs=[theta1, omega1, theta2, omega2, alpha1, alpha2,
                              self.length1, self.length2, self.mass1, self.mass2, self.gravity])

            wp.launch(implicit_euler_residual_pt, dim=n,
                      inputs=[theta1, omega1, theta2, omega2,
                              theta1_prev, omega1_prev, theta2_prev, omega2_prev,
                              alpha1, alpha2, residual, dt_wp])

            wp.synchronize()
            max_residual = np.max(residual.numpy())

            if max_residual < self.newton_tol:
                break

            # Fixed-point update with per-thread dt (numpy broadcast)
            dt_col = dt_arr  # shape [n]
            theta1_np = theta1_prev.numpy() + dt_col * omega1.numpy()
            omega1_np = omega1_prev.numpy() + dt_col * alpha1.numpy()
            theta2_np = theta2_prev.numpy() + dt_col * omega2.numpy()
            omega2_np = omega2_prev.numpy() + dt_col * alpha2.numpy()

            theta1 = wp.array(theta1_np.astype(np.float32), dtype=wp.float32)
            omega1 = wp.array(omega1_np.astype(np.float32), dtype=wp.float32)
            theta2 = wp.array(theta2_np.astype(np.float32), dtype=wp.float32)
            omega2 = wp.array(omega2_np.astype(np.float32), dtype=wp.float32)

        return theta1, omega1, theta2, omega2

    def _calc_adjusted_step_size(self, err, dt):
        """Adjust step size with safety factor and hysteresis"""
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
            dt_new = safety * dt * (self.epsilon_acc / err) ** (1.0 / (self.error_order + 1))

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

    def _step_doubling_pt(self, theta1, omega1, theta2, omega2, dt_arr):
        """Step-doubling error estimation with per-thread dt.

        Returns: (t1_half, w1_half, t2_half, w2_half, errors_per_pendulum, dt_new_arr)
        """
        n = self.num_pendulums

        # Full step
        t1_full = wp.array(theta1.numpy(), dtype=wp.float32)
        w1_full = wp.array(omega1.numpy(), dtype=wp.float32)
        t2_full = wp.array(theta2.numpy(), dtype=wp.float32)
        w2_full = wp.array(omega2.numpy(), dtype=wp.float32)
        t1_full, w1_full, t2_full, w2_full = self._implicit_euler_step_gpu_pt(
            t1_full, w1_full, t2_full, w2_full, dt_arr)

        # Two half steps
        half_dt = dt_arr / 2.0
        t1_half = wp.array(theta1.numpy(), dtype=wp.float32)
        w1_half = wp.array(omega1.numpy(), dtype=wp.float32)
        t2_half = wp.array(theta2.numpy(), dtype=wp.float32)
        w2_half = wp.array(omega2.numpy(), dtype=wp.float32)
        t1_half, w1_half, t2_half, w2_half = self._implicit_euler_step_gpu_pt(
            t1_half, w1_half, t2_half, w2_half, half_dt)
        t1_half, w1_half, t2_half, w2_half = self._implicit_euler_step_gpu_pt(
            t1_half, w1_half, t2_half, w2_half, half_dt)

        # Compute per-pendulum error
        error_arr = wp.zeros(n, dtype=wp.float32)
        wp.launch(compute_error, dim=n,
                  inputs=[t1_full, w1_full, t2_full, w2_full,
                          t1_half, w1_half, t2_half, w2_half, error_arr])
        wp.synchronize()

        errors = error_arr.numpy().copy()

        # Per-pendulum step size adjustment
        dt_new = np.empty(n, dtype=np.float64)
        for i in range(n):
            dt_new[i] = self._calc_adjusted_step_size(float(errors[i]), float(dt_arr[i]))

        return t1_half, w1_half, t2_half, w2_half, errors, dt_new

    def run(self, verbose=False):
        """Run per-thread adaptive simulation"""
        start_time = time.perf_counter()
        n = self.num_pendulums

        # Current state as numpy (we'll wrap to warp per step)
        theta1_np = self.initial_theta1.copy()
        omega1_np = self.initial_omega1.copy()
        theta2_np = self.initial_theta2.copy()
        omega2_np = self.initial_omega2.copy()

        current_time = np.zeros(n, dtype=np.float64)
        dt = np.full(n, self.initial_dt, dtype=np.float64)
        active = np.ones(n, dtype=bool)

        # Record initial state
        for i in range(n):
            self.times[i].append(0.0)
            self.states_theta1[i].append(float(theta1_np[i]))
            self.states_omega1[i].append(float(omega1_np[i]))
            self.states_theta2[i].append(float(theta2_np[i]))
            self.states_omega2[i].append(float(omega2_np[i]))
            self.dt_histories[i].append(float(dt[i]))
            self.error_histories[i].append(0.0)

        while np.any(active):
            # Clamp dt so we don't overshoot end_time
            remaining = self.end_time - current_time
            dt_clamped = np.where(active, np.minimum(dt, remaining), dt)

            # Build warp arrays from current numpy state
            theta1 = wp.array(theta1_np.astype(np.float32), dtype=wp.float32)
            omega1 = wp.array(omega1_np.astype(np.float32), dtype=wp.float32)
            theta2 = wp.array(theta2_np.astype(np.float32), dtype=wp.float32)
            omega2 = wp.array(omega2_np.astype(np.float32), dtype=wp.float32)

            t1_new, w1_new, t2_new, w2_new, errors, dt_new = self._step_doubling_pt(
                theta1, omega1, theta2, omega2, dt_clamped.astype(np.float32))

            # Accept/reject per pendulum
            accepted = active & (errors <= self.epsilon_acc)
            rejected = active & (errors > self.epsilon_acc)

            # Extract new state from warp
            t1_np_new = t1_new.numpy()
            w1_np_new = w1_new.numpy()
            t2_np_new = t2_new.numpy()
            w2_np_new = w2_new.numpy()

            # Update accepted pendulums
            for i in np.where(accepted)[0]:
                theta1_np[i] = t1_np_new[i]
                omega1_np[i] = w1_np_new[i]
                theta2_np[i] = t2_np_new[i]
                omega2_np[i] = w2_np_new[i]
                current_time[i] += dt_clamped[i]

                self.times[i].append(float(current_time[i]))
                self.states_theta1[i].append(float(theta1_np[i]))
                self.states_omega1[i].append(float(omega1_np[i]))
                self.states_theta2[i].append(float(theta2_np[i]))
                self.states_omega2[i].append(float(omega2_np[i]))
                self.dt_histories[i].append(float(dt_clamped[i]))
                self.error_histories[i].append(float(errors[i]))
                self.accepted_steps[i] += 1

            self.rejected_steps[rejected] += 1

            # Mark done
            active[current_time >= self.end_time] = False

            # Update dt for next iteration
            dt = dt_new

        self.wall_clock_time = time.perf_counter() - start_time

        if verbose:
            self.print_summary()

        return self

    def print_summary(self):
        """Print per-pendulum and aggregate statistics"""
        n = self.num_pendulums
        print(f"\nPer-Thread dt Double Pendulum WARP Simulation ({n} pendulums):")
        print(f"  Wall clock time: {self.wall_clock_time:.4f}s")
        print(f"  {'Pend':>4s}  {'Accepted':>8s}  {'Rejected':>8s}  {'Rej%':>6s}  {'dt_min':>10s}  {'dt_max':>10s}  {'dt_mean':>10s}  {'Final t':>9s}")
        for i in range(n):
            total = self.accepted_steps[i] + self.rejected_steps[i]
            rej_rate = 100 * self.rejected_steps[i] / total if total > 0 else 0
            dts = self.dt_histories[i]
            print(f"  {i:>4d}  {self.accepted_steps[i]:>8d}  {self.rejected_steps[i]:>8d}  {rej_rate:>5.1f}%  {min(dts):>10.2e}  {max(dts):>10.2e}  {np.mean(dts):>10.2e}  {self.times[i][-1]:>9.4f}")
        print(f"  Total accepted: {np.sum(self.accepted_steps)}")
        print(f"  Total rejected: {np.sum(self.rejected_steps)}")

    def get_time(self, pendulum_idx):
        """Return time array for one pendulum"""
        return np.array(self.times[pendulum_idx])

    def get_states_array(self, pendulum_idx):
        """Return states for one pendulum as [N×4] array"""
        return np.column_stack([
            self.states_theta1[pendulum_idx],
            self.states_omega1[pendulum_idx],
            self.states_theta2[pendulum_idx],
            self.states_omega2[pendulum_idx],
        ])

    def get_dt_history(self, pendulum_idx):
        """Return dt history for one pendulum"""
        return np.array(self.dt_histories[pendulum_idx])

    def get_errors(self, pendulum_idx):
        """Return error history for one pendulum"""
        return np.array(self.error_histories[pendulum_idx])
