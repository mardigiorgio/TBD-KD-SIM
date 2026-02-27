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
def implicit_euler_solve_pt(
    theta1: wp.array(dtype=wp.float32), omega1: wp.array(dtype=wp.float32),
    theta2: wp.array(dtype=wp.float32), omega2: wp.array(dtype=wp.float32),
    dt: wp.array(dtype=wp.float32),
    L1: wp.float32, L2: wp.float32, m1: wp.float32, m2: wp.float32, g: wp.float32,
    newton_tol: wp.float32, newton_max_iter: wp.int32,
):
    """Run the full fixed-point loop on-device with per-thread dt — no CPU round-trips."""
    tid = wp.tid()
    h = dt[tid]
    t1_prev = theta1[tid];  w1_prev = omega1[tid]
    t2_prev = theta2[tid];  w2_prev = omega2[tid]

    # Initial guess: explicit Euler
    acc = pendulum_dynamics(t1_prev, w1_prev, t2_prev, w2_prev, L1, L2, m1, m2, g)
    t1 = t1_prev + h * w1_prev;  w1 = w1_prev + h * acc[0]
    t2 = t2_prev + h * w2_prev;  w2 = w2_prev + h * acc[1]

    # Fixed-point loop — per-thread convergence, no CPU round-trip
    for _iter in range(newton_max_iter):
        acc = pendulum_dynamics(t1, w1, t2, w2, L1, L2, m1, m2, g)
        r0 = t1 - t1_prev - h * w1;  r1 = w1 - w1_prev - h * acc[0]
        r2 = t2 - t2_prev - h * w2;  r3 = w2 - w2_prev - h * acc[1]
        if wp.sqrt(r0*r0 + r1*r1 + r2*r2 + r3*r3) < newton_tol:
            break
        t1 = t1_prev + h * w1;  w1 = w1_prev + h * acc[0]
        t2 = t2_prev + h * w2;  w2 = w2_prev + h * acc[1]

    theta1[tid] = t1;  omega1[tid] = w1
    theta2[tid] = t2;  omega2[tid] = w2


@wp.kernel
def selective_update_state(
    theta1:  wp.array(dtype=wp.float32),
    omega1:  wp.array(dtype=wp.float32),
    theta2:  wp.array(dtype=wp.float32),
    omega2:  wp.array(dtype=wp.float32),
    t1_new:  wp.array(dtype=wp.float32),
    w1_new:  wp.array(dtype=wp.float32),
    t2_new:  wp.array(dtype=wp.float32),
    w2_new:  wp.array(dtype=wp.float32),
    accepted: wp.array(dtype=wp.int32),
):
    """In-place state update: only copy from new arrays where accepted[tid] == 1."""
    tid = wp.tid()
    if accepted[tid]:
        theta1[tid] = t1_new[tid]
        omega1[tid] = w1_new[tid]
        theta2[tid] = t2_new[tid]
        omega2[tid] = w2_new[tid]


class AdaptiveDoublePendulumWarpPT:
    def __init__(self, num_pendulums, initial_theta1, initial_omega1,
                 initial_theta2, initial_omega2, epsilon_acc,
                 end_time=10.0, initial_dt=0.1,
                 gravity=9.81, length1=1.0, length2=1.0,
                 mass1=1.0, mass2=1.0,
                 minimum_step_size=1e-10,
                 throw_on_minimum_step_size_violation=False,
                 quiet=False, store_history=True):
        """
        Per-thread adaptive double pendulum simulator with WARP GPU acceleration.

        Each pendulum independently controls its own time step based on its
        own local truncation error, rather than being constrained by the
        worst-case pendulum across the batch.

        Parameters:
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
        # ensuring >=1 fixed-point correction fires before the loop exits.
        # (cf. Drake integrator_base.cc — solver convergence separate from error ctrl)
        self.newton_tol = max(epsilon_acc * 0.01, 2e-7)
        self.newton_max_iter = 15
        # Fix 2+3: minimum step size guard and force-accept flag
        # (cf. Drake ValidateSmallerStepSize lines 556-572, CalcAdjustedStepSize 344-355)
        self.minimum_step_size = minimum_step_size
        self.throw_on_minimum_step_size_violation = throw_on_minimum_step_size_violation

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
        self.min_dt_accepted = np.full(n, np.inf)
        self.max_dt_accepted = np.zeros(n)

        # Final state (always stored, even without history)
        self.final_theta1 = None
        self.final_omega1 = None
        self.final_theta2 = None
        self.final_omega2 = None

    def _implicit_euler_step_gpu_pt(self, theta1, omega1, theta2, omega2, dt_wp):
        """Implicit Euler step with per-thread dt — entire fixed-point loop runs on-device.

        dt_wp: wp.array(dtype=wp.float32) — per-thread step sizes, already on GPU.
        """
        wp.launch(implicit_euler_solve_pt,
                  dim=self.num_pendulums,
                  inputs=[theta1, omega1, theta2, omega2,
                          dt_wp,
                          self.length1, self.length2, self.mass1, self.mass2, self.gravity,
                          np.float32(self.newton_tol), np.int32(self.newton_max_iter)])
        return theta1, omega1, theta2, omega2

    def _calc_adjusted_step_sizes_vec(self, errors, dt_arr, at_minimum):
        """Vectorized step size adjustment for all N pendulums.

        Replaces the O(N) Python for-loop that called _calc_adjusted_step_size
        per pendulum.  All operations are numpy-level (no Python loop).

        Parameters
        ----------
        errors    : float64 array [N] — per-pendulum step-doubling error
        dt_arr    : float64 array [N] — current per-pendulum step sizes
        at_minimum: bool    array [N] — True if pendulum is already at minimum dt

        Returns
        -------
        dt_new      : float64 array [N]
        at_min_new  : bool    array [N]
        force_accept: bool    array [N]
        """
        safety         = 0.9
        min_shrink     = 0.1
        max_grow       = 5.0
        hysteresis_low = 0.9
        hysteresis_high = 1.2

        errors  = np.asarray(errors,   dtype=np.float64)
        dt_arr  = np.asarray(dt_arr,   dtype=np.float64)

        nan_inf = np.isnan(errors) | np.isinf(errors)
        zero_e  = (~nan_inf) & (errors == 0.0)
        normal  = (~nan_inf) & (~zero_e)

        dt_new = np.empty(len(errors), dtype=np.float64)
        dt_new[nan_inf] = dt_arr[nan_inf] * min_shrink
        dt_new[zero_e]  = dt_arr[zero_e]  * max_grow
        dt_new[normal]  = (safety * dt_arr[normal]
                           * (self.epsilon_acc / errors[normal]) ** (1.0 / self.error_order))

        # Hysteresis: suppress negligible growth
        growing = dt_new > dt_arr
        dt_new = np.where(growing & (dt_new < hysteresis_high * dt_arr), dt_arr, dt_new)

        # Hysteresis: don't shrink if already within tolerance; apply floor otherwise
        shrinking = dt_new < dt_arr
        dt_new = np.where(shrinking & (errors <= self.epsilon_acc), dt_arr,
                 np.where(shrinking & (errors >  self.epsilon_acc),
                          np.minimum(dt_new, hysteresis_low * dt_arr), dt_new))

        # Hard bounds: [min_shrink, max_grow] × dt
        dt_new = np.clip(dt_new, dt_arr * min_shrink, dt_arr * max_grow)

        # Minimum step size guard
        # (cf. Drake ValidateSmallerStepSize / CalcAdjustedStepSize lines 344-355)
        below_min = dt_new < self.minimum_step_size

        if self.throw_on_minimum_step_size_violation:
            violators = below_min & ~at_minimum
            if np.any(violators):
                raise RuntimeError(
                    f"Requested step size below minimum permitted "
                    f"{self.minimum_step_size:.3e} s. "
                    "Consider reducing minimum_step_size or increasing epsilon_acc."
                )

        # Already at minimum last step → force-accept, keep current dt
        force_accept = below_min & at_minimum
        dt_new = np.where(force_accept, dt_arr, dt_new)

        # First time below minimum → clamp to minimum (will force-accept next step)
        dt_new = np.where(below_min & ~at_minimum, self.minimum_step_size, dt_new)

        at_min_new = at_minimum.copy()
        at_min_new[below_min]  = True
        at_min_new[~below_min] = False

        return dt_new, at_min_new, force_accept

    def _step_doubling_pt(self, theta1, omega1, theta2, omega2, dt_arr,
                          at_minimum=None, active=None):
        """Step-doubling error estimation with per-thread dt.

        dt_arr:     numpy float32 array [n].
        at_minimum: numpy bool array [n] — True if pendulum is already at minimum
                    step size from the previous iteration (Fix 3).
        active:     numpy bool array [n] — skip dt adjustment for inactive pendulums
                    so they cannot spuriously trigger minimum-step violations.
        Returns: (t1_half, w1_half, t2_half, w2_half, errors, dt_new,
                  at_min_new, force_accept)
        """
        n = self.num_pendulums

        if at_minimum is None:
            at_minimum = np.zeros(n, dtype=bool)

        # Build warp dt arrays once — avoids 2 redundant CPU→GPU uploads
        dt_wp      = wp.array(dt_arr.astype(np.float32), dtype=wp.float32)
        half_dt_wp = wp.array((dt_arr / 2.0).astype(np.float32), dtype=wp.float32)

        # Full step — on-device clone
        t1_full = wp.clone(theta1);  w1_full = wp.clone(omega1)
        t2_full = wp.clone(theta2);  w2_full = wp.clone(omega2)
        t1_full, w1_full, t2_full, w2_full = self._implicit_euler_step_gpu_pt(
            t1_full, w1_full, t2_full, w2_full, dt_wp)

        # Two half steps — on-device clone, reuse half_dt_wp both times
        t1_half = wp.clone(theta1);  w1_half = wp.clone(omega1)
        t2_half = wp.clone(theta2);  w2_half = wp.clone(omega2)
        t1_half, w1_half, t2_half, w2_half = self._implicit_euler_step_gpu_pt(
            t1_half, w1_half, t2_half, w2_half, half_dt_wp)
        t1_half, w1_half, t2_half, w2_half = self._implicit_euler_step_gpu_pt(
            t1_half, w1_half, t2_half, w2_half, half_dt_wp)

        # Per-pendulum error in float64: dynamics stay in fp32 on GPU (RTX 4070 Ti SUPER
        # has 1/64 fp64 throughput), only cast at comparison time.  Eliminates the
        # ~1e-7 float32 noise floor that caused dt to stall at epsilon_acc=1e-6.
        wp.synchronize()
        t1f = t1_full.numpy().astype(np.float64);  w1f = w1_full.numpy().astype(np.float64)
        t2f = t2_full.numpy().astype(np.float64);  w2f = w2_full.numpy().astype(np.float64)
        t1h = t1_half.numpy().astype(np.float64);  w1h = w1_half.numpy().astype(np.float64)
        t2h = t2_half.numpy().astype(np.float64);  w2h = w2_half.numpy().astype(np.float64)
        errors = np.sqrt((t1f - t1h)**2 + (w1f - w1h)**2
                         + (t2f - t2h)**2 + (w2f - w2h)**2)

        # Per-pendulum step size adjustment — vectorized over active pendulums only
        # to avoid spurious minimum-step violations on finished trajectories.
        # Replaces an O(N) Python for-loop that was the dominant wall-time cost at large N.
        dt_new       = dt_arr.astype(np.float64).copy()
        at_min_new   = at_minimum.copy()
        force_accept = np.zeros(n, dtype=bool)

        active_mask = np.ones(n, dtype=bool) if active is None else active
        if np.any(active_mask):
            dt_new[active_mask], at_min_new[active_mask], force_accept[active_mask] = \
                self._calc_adjusted_step_sizes_vec(
                    errors[active_mask], dt_arr[active_mask], at_minimum[active_mask])

        return t1_half, w1_half, t2_half, w2_half, errors, dt_new, at_min_new, force_accept

    def run(self, verbose=False, log_every=0):
        """Run per-thread adaptive simulation"""
        start_time = time.perf_counter()
        n = self.num_pendulums

        # Persistent GPU state — no per-step CPU round-trips
        theta1 = wp.array(self.initial_theta1, dtype=wp.float32)
        omega1 = wp.array(self.initial_omega1, dtype=wp.float32)
        theta2 = wp.array(self.initial_theta2, dtype=wp.float32)
        omega2 = wp.array(self.initial_omega2, dtype=wp.float32)

        current_time = np.zeros(n, dtype=np.float64)
        dt = np.full(n, self.initial_dt, dtype=np.float64)
        active = np.ones(n, dtype=bool)
        at_minimum = np.zeros(n, dtype=bool)  # Fix 3: per-pendulum force-accept state
        _loop_count = 0

        # Record initial state (one pull at start)
        if self.store_history:
            t1_np0 = theta1.numpy()
            w1_np0 = omega1.numpy()
            t2_np0 = theta2.numpy()
            w2_np0 = omega2.numpy()
            for i in range(n):
                self.times[i].append(0.0)
                self.states_theta1[i].append(float(t1_np0[i]))
                self.states_omega1[i].append(float(w1_np0[i]))
                self.states_theta2[i].append(float(t2_np0[i]))
                self.states_omega2[i].append(float(w2_np0[i]))
                self.dt_histories[i].append(float(dt[i]))
                self.error_histories[i].append(0.0)

        while np.any(active):
            _loop_count += 1
            if log_every > 0 and _loop_count % log_every == 0:
                n_active = int(np.sum(active))
                t_min = float(np.min(current_time[active])) if n_active else self.end_time
                dt_min = float(np.min(dt[active])) if n_active else 0.0
                print(f"\r    t_min={t_min:.4f}/{self.end_time:.1f}s  dt_min={dt_min:.3e}  active={n_active}/{n}   ",
                      end='', flush=True)

            # Clamp dt so we don't overshoot end_time
            remaining = self.end_time - current_time
            dt_clamped = np.where(active, np.minimum(dt, remaining), dt)
            # Fix 4: per-pendulum artificial-limiting flag — don't feed a
            # clamped-to-end_time step size back as the ideal next dt
            # (cf. Drake StepOnceErrorControlledAtMost lines 100-111, 154-158)
            h_was_limited = active & (dt_clamped < 0.95 * dt)

            t1_new, w1_new, t2_new, w2_new, errors, dt_new, at_min_new, force_accept_arr = \
                self._step_doubling_pt(theta1, omega1, theta2, omega2,
                                       dt_clamped.astype(np.float32), at_minimum, active)

            # Accept/reject per pendulum
            accepted = active & ((errors <= self.epsilon_acc) | force_accept_arr)
            rejected = active & ~accepted

            # GPU-side selective state update — one small int push replaces 8 large float round-trips
            accepted_wp = wp.array(accepted.astype(np.int32), dtype=wp.int32)
            wp.launch(selective_update_state, dim=n,
                      inputs=[theta1, omega1, theta2, omega2,
                               t1_new, w1_new, t2_new, w2_new, accepted_wp])

            # Advance clock for accepted pendulums
            current_time[accepted] += dt_clamped[accepted]

            # History storage: one pull per step only when needed
            if self.store_history and np.any(accepted):
                wp.synchronize()
                t1_np = theta1.numpy()
                w1_np = omega1.numpy()
                t2_np = theta2.numpy()
                w2_np = omega2.numpy()
                for i in np.where(accepted)[0]:
                    self.times[i].append(float(current_time[i]))
                    self.states_theta1[i].append(float(t1_np[i]))
                    self.states_omega1[i].append(float(w1_np[i]))
                    self.states_theta2[i].append(float(t2_np[i]))
                    self.states_omega2[i].append(float(w2_np[i]))
                    self.dt_histories[i].append(float(dt_clamped[i]))
                    self.error_histories[i].append(float(errors[i]))

            self.accepted_steps[accepted] += 1
            self.rejected_steps[rejected] += 1
            self.min_dt_accepted = np.where(accepted,
                np.minimum(self.min_dt_accepted, dt_clamped), self.min_dt_accepted)
            self.max_dt_accepted = np.where(accepted,
                np.maximum(self.max_dt_accepted, dt_clamped), self.max_dt_accepted)

            # Mark done
            active[current_time >= self.end_time] = False

            # Fix 3: propagate per-pendulum at_minimum state (active only, so
            # finished pendulums don't corrupt their stale flag)
            at_minimum[active] = at_min_new[active]

            # Fix 4: dt update — keep ideal dt when step was clamped to end_time
            accepted_not_limited = accepted & ~h_was_limited
            dt[accepted_not_limited] = dt_new[accepted_not_limited]
            dt[rejected] = dt_new[rejected]
            # accepted & h_was_limited: dt unchanged (ideal dt preserved)

        if log_every > 0:
            print()  # newline after final \r status

        # Store final state — one pull at the end
        wp.synchronize()
        self.final_theta1 = theta1.numpy().copy()
        self.final_omega1 = omega1.numpy().copy()
        self.final_theta2 = theta2.numpy().copy()
        self.final_omega2 = omega2.numpy().copy()

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
