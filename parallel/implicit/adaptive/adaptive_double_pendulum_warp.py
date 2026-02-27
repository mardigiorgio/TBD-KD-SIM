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


@wp.func
def pendulum_dynamics(
    t1: wp.float32, w1: wp.float32,
    t2: wp.float32, w2: wp.float32,
    L1: wp.float32, L2: wp.float32,
    m1: wp.float32, m2: wp.float32,
    g:  wp.float32,
) -> wp.vec2:
    """Compute (alpha1, alpha2) for double pendulum — callable from within a kernel."""
    delta  = t1 - t2
    sin_d  = wp.sin(delta)
    cos_d  = wp.cos(delta)
    cos_2d = wp.cos(2.0 * delta)
    denom  = 2.0 * m1 + m2 - m2 * cos_2d
    num1 = (-g * (2.0 * m1 + m2) * wp.sin(t1)
            - m2 * g * wp.sin(t1 - 2.0 * t2)
            - 2.0 * sin_d * m2 * (w2*w2*L2 + w1*w1*L1*cos_d))
    num2 = (2.0 * sin_d * (w1*w1*L1*(m1+m2)
            + g*(m1+m2)*wp.cos(t1)
            + w2*w2*L2*m2*cos_d))
    return wp.vec2(num1 / (L1*denom), num2 / (L2*denom))


@wp.kernel
def implicit_euler_solve(
    theta1: wp.array(dtype=wp.float32), omega1: wp.array(dtype=wp.float32),
    theta2: wp.array(dtype=wp.float32), omega2: wp.array(dtype=wp.float32),
    dt: wp.float32,
    L1: wp.float32, L2: wp.float32, m1: wp.float32, m2: wp.float32, g: wp.float32,
    newton_tol: wp.float32, newton_max_iter: wp.int32,
):
    """Run the full fixed-point loop on-device — no CPU round-trips."""
    tid = wp.tid()
    t1_prev = theta1[tid];  w1_prev = omega1[tid]
    t2_prev = theta2[tid];  w2_prev = omega2[tid]

    # Initial guess: explicit Euler
    acc = pendulum_dynamics(t1_prev, w1_prev, t2_prev, w2_prev, L1, L2, m1, m2, g)
    t1 = t1_prev + dt * w1_prev;  w1 = w1_prev + dt * acc[0]
    t2 = t2_prev + dt * w2_prev;  w2 = w2_prev + dt * acc[1]

    # Fixed-point loop — per-thread convergence, no CPU round-trip
    for _iter in range(newton_max_iter):
        acc = pendulum_dynamics(t1, w1, t2, w2, L1, L2, m1, m2, g)
        r0 = t1 - t1_prev - dt * w1;  r1 = w1 - w1_prev - dt * acc[0]
        r2 = t2 - t2_prev - dt * w2;  r3 = w2 - w2_prev - dt * acc[1]
        if wp.sqrt(r0*r0 + r1*r1 + r2*r2 + r3*r3) < newton_tol:
            break
        t1 = t1_prev + dt * w1;  w1 = w1_prev + dt * acc[0]
        t2 = t2_prev + dt * w2;  w2 = w2_prev + dt * acc[1]

    theta1[tid] = t1;  omega1[tid] = w1
    theta2[tid] = t2;  omega2[tid] = w2


class AdaptiveDoublePendulumWarp:
    def __init__(self, num_pendulums, initial_theta1, initial_omega1,
                 initial_theta2, initial_omega2, epsilon_acc,
                 end_time=10.0, initial_dt=0.1,
                 gravity=9.81, length1=1.0, length2=1.0,
                 mass1=1.0, mass2=1.0,
                 minimum_step_size=1e-10,
                 throw_on_minimum_step_size_violation=False,
                 quiet=False, store_history=True):
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
            minimum_step_size: Smallest permitted step size (cf. Drake
                working_minimum_step_size). Default 1e-10 s — below float32
                significance, acts as a last-resort Zeno guard only.
            throw_on_minimum_step_size_violation: If True, raise RuntimeError
                when dt < minimum_step_size; if False (default), silently
                force-accept the step (cf. Drake ValidateSmallerStepSize).
            quiet: Suppress per-step print output
            store_history: Store full trajectory history (disable for benchmarking)
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

        self.quiet = quiet
        self.store_history = store_history

        self.error_order = 2
        # Fix 1: couple newton_tol to epsilon_acc so solver noise << epsilon_acc.
        # Measured float32 residual floor ~1.077e-7; clamp at 2e-7 for safety.
        # At epsilon_acc=1e-3 this gives 1e-5 (prior behaviour); at 1e-6 gives 2e-7,
        # ensuring ≥1 fixed-point correction fires before the loop exits.
        # (cf. Drake integrator_base.cc — solver convergence separate from error ctrl)
        self.newton_tol = max(epsilon_acc * 0.01, 2e-7)
        self.newton_max_iter = 15
        # Fix 2+3: minimum step size guard and force-accept flag
        # (cf. Drake ValidateSmallerStepSize lines 556-572, CalcAdjustedStepSize 344-355)
        self.minimum_step_size = minimum_step_size
        self.throw_on_minimum_step_size_violation = throw_on_minimum_step_size_violation

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
        self.min_dt_accepted = np.inf
        self.max_dt_accepted = 0.0

        # Final state (always stored, even without history)
        self.final_theta1 = None
        self.final_omega1 = None
        self.final_theta2 = None
        self.final_omega2 = None

    def _implicit_euler_step_gpu(self, theta1, omega1, theta2, omega2, dt):
        """Implicit Euler step — entire fixed-point loop runs on-device."""
        wp.launch(implicit_euler_solve,
                  dim=self.num_pendulums,
                  inputs=[theta1, omega1, theta2, omega2,
                          np.float32(dt),
                          self.length1, self.length2, self.mass1, self.mass2, self.gravity,
                          np.float32(self.newton_tol), np.int32(self.newton_max_iter)])
        return theta1, omega1, theta2, omega2

    def _calc_adjusted_step_size(self, err, dt, at_minimum_step_size=False):
        safety = 0.9
        min_shrink = 0.1
        max_grow = 5.0
        hysteresis_low = 0.9
        hysteresis_high = 1.2

        if np.isnan(err) or np.isinf(err):
            dt_new = dt * min_shrink
        elif err == 0:
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

        if dt_new < self.minimum_step_size:
            if at_minimum_step_size:
                # Already clamped last iteration — force-accept so sim can progress
                return dt, True, True
            if self.throw_on_minimum_step_size_violation:
                raise RuntimeError(
                    f"Requested step size {dt_new:.3e} s is below the minimum "
                    f"permitted step size {self.minimum_step_size:.3e} s. "
                    "Consider reducing minimum_step_size or increasing epsilon_acc."
                )
            return self.minimum_step_size, True, False

        return dt_new, False, False

    def _step_doubling(self, theta1, omega1, theta2, omega2, dt,
                       at_minimum_step_size=False):
        """Step-doubling error estimation.

        Returns (t1, w1, t2, w2, dt_new, max_error, new_at_minimum, force_accept).
        """
        n = self.num_pendulums

        # Full step — on-device clone
        t1_full = wp.clone(theta1);  w1_full = wp.clone(omega1)
        t2_full = wp.clone(theta2);  w2_full = wp.clone(omega2)
        t1_full, w1_full, t2_full, w2_full = self._implicit_euler_step_gpu(
            t1_full, w1_full, t2_full, w2_full, dt)

        # Two half steps — on-device clone
        t1_half = wp.clone(theta1);  w1_half = wp.clone(omega1)
        t2_half = wp.clone(theta2);  w2_half = wp.clone(omega2)
        t1_half, w1_half, t2_half, w2_half = self._implicit_euler_step_gpu(
            t1_half, w1_half, t2_half, w2_half, dt / 2)
        t1_half, w1_half, t2_half, w2_half = self._implicit_euler_step_gpu(
            t1_half, w1_half, t2_half, w2_half, dt / 2)

        # Error computation in float64: dynamics stay in fp32 on GPU (RTX 4070 Ti SUPER
        # has 1/64 fp64 throughput), only cast at comparison time.  Eliminates the
        # ~1e-7 float32 noise floor that caused dt to stall at epsilon_acc=1e-6.
        wp.synchronize()
        t1f = t1_full.numpy().astype(np.float64);  w1f = w1_full.numpy().astype(np.float64)
        t2f = t2_full.numpy().astype(np.float64);  w2f = w2_full.numpy().astype(np.float64)
        t1h = t1_half.numpy().astype(np.float64);  w1h = w1_half.numpy().astype(np.float64)
        t2h = t2_half.numpy().astype(np.float64);  w2h = w2_half.numpy().astype(np.float64)
        errors_f64 = np.sqrt((t1f - t1h)**2 + (w1f - w1h)**2
                             + (t2f - t2h)**2 + (w2f - w2h)**2)
        max_error = float(np.max(errors_f64))
        dt_new, new_at_minimum, force_accept = self._calc_adjusted_step_size(
            max_error, dt, at_minimum_step_size)
        return t1_half, w1_half, t2_half, w2_half, dt_new, max_error, new_at_minimum, force_accept

    def run(self, verbose=False, log_every=0):
        """Run adaptive simulation"""
        start_time = time.perf_counter()

        theta1 = wp.array(self.initial_theta1, dtype=wp.float32)
        omega1 = wp.array(self.initial_omega1, dtype=wp.float32)
        theta2 = wp.array(self.initial_theta2, dtype=wp.float32)
        omega2 = wp.array(self.initial_omega2, dtype=wp.float32)

        current_time = 0.0
        dt = self.initial_dt
        at_minimum_step_size = False  # Fix 3: track force-accept state across steps

        if self.store_history:
            self.time = [current_time]
            self.states_theta1 = [theta1.numpy().copy()]
            self.states_omega1 = [omega1.numpy().copy()]
            self.states_theta2 = [theta2.numpy().copy()]
            self.states_omega2 = [omega2.numpy().copy()]
            self.dt_history = [dt]
            self.errors = [0.0]
        self.accepted_steps = 0
        self.rejected_steps = 0
        _total_attempts = 0

        while current_time < self.end_time:
            dt_used = min(dt, self.end_time - current_time)
            # Fix 4: flag when dt was clamped to hit end_time exactly so we don't
            # feed the artificially-small step size back as the ideal next step
            # (cf. Drake StepOnceErrorControlledAtMost lines 100-111, 154-158)
            h_was_artificially_limited = (dt_used < 0.95 * dt)

            t1_new, w1_new, t2_new, w2_new, dt_new, error, new_at_min, force_accept = \
                self._step_doubling(theta1, omega1, theta2, omega2,
                                    dt_used, at_minimum_step_size)

            _total_attempts += 1
            if log_every > 0 and _total_attempts % log_every == 0:
                print(f"\r    t={current_time:.4f}/{self.end_time:.1f}s  dt={dt_used:.3e}  acc={self.accepted_steps}  rej={self.rejected_steps}   ",
                      end='', flush=True)

            if error <= self.epsilon_acc or force_accept:
                theta1 = t1_new
                omega1 = w1_new
                theta2 = t2_new
                omega2 = w2_new
                current_time += dt_used

                if self.store_history:
                    self.time.append(current_time)
                    self.states_theta1.append(theta1.numpy().copy())
                    self.states_omega1.append(omega1.numpy().copy())
                    self.states_theta2.append(theta2.numpy().copy())
                    self.states_omega2.append(omega2.numpy().copy())
                    self.dt_history.append(dt_used)
                    self.errors.append(error)

                self.accepted_steps += 1
                if dt_used < self.min_dt_accepted: self.min_dt_accepted = dt_used
                if dt_used > self.max_dt_accepted: self.max_dt_accepted = dt_used
                at_minimum_step_size = new_at_min
                if not h_was_artificially_limited:
                    dt = dt_new
            else:
                self.rejected_steps += 1
                dt = dt_new
                at_minimum_step_size = new_at_min

        if log_every > 0:
            print()  # newline after final \r status

        # Store final state
        self.final_theta1 = theta1.numpy().copy()
        self.final_omega1 = omega1.numpy().copy()
        self.final_theta2 = theta2.numpy().copy()
        self.final_omega2 = omega2.numpy().copy()

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
