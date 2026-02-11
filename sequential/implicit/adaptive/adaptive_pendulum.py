"""
Adaptive Time Stepping Pendulum - Step-Doubling Error Control

Pure implementation of the step-doubling method (also called half-stepping).
The integrator computes two estimates:
  - x̂^{n+1}: computed with one full step of size δt (equation 24)
  - x^{n+1}: computed with two half-steps of size δt/2 (equations 22-23)

The local truncation error is estimated as:
  e^{n+1} = ||x̂^{n+1} - x^{n+1}|| ≈ c̄δt^p

where p is the order of the error estimate. The step size is then adjusted to
maintain a user-specified accuracy ε_acc using the pure formula:
  δt_new ← δt(ε_acc/e^{n+1})^{1/p}
"""

import numpy as np
import time

class AdaptivePendulum:
    def __init__(self, initial_theta, initial_omega, epsilon_acc,
                 end_time=10.0, initial_dt=0.1,
                 gravity=9.81, length=1.0):
        """
        Initialize adaptive pendulum simulator

        Parameters:
            initial_theta: Initial angle (rad), θ₀
            initial_omega: Initial angular velocity (rad/s), ω₀
            epsilon_acc: Target accuracy ε_acc for error control
            end_time: Simulation end time (s)
            initial_dt: Initial time step size δt₀ (s)
            gravity: Gravitational acceleration g (m/s²)
            length: Pendulum length L (m)
        """
        self.initial_state = np.array([initial_theta, initial_omega])
        self.epsilon_acc = epsilon_acc
        self.end_time = end_time
        self.initial_dt = initial_dt
        self.gravity = gravity
        self.length = length

        self.error_order = 1  # p: order of error estimate (1 for Euler)
        self.newton_tol = 1e-10  # Newton-Raphson convergence tolerance
        self.newton_max_iter = 10  # Maximum Newton iterations per step

        # Simulation results
        self.time = []  # t^n at each accepted step
        self.states = []  # [θ^n, ω^n] at each accepted step
        self.dt_history = []  # δt used at each accepted step
        self.errors = []  # e^{n+1} at each accepted step
        self.accepted_steps = 0  # Number of accepted steps
        self.rejected_steps = 0  # Number of rejected steps
        self.total_newton_iterations = 0  # Total Newton iterations (computational cost)
        self.wall_clock_time = 0.0  # Wall clock time (s)

    def dynamics(self, state):
        """Compute state derivative: θ̇ = ω, ω̇ = -(g/L)sin(θ)"""
        theta, omega = state
        theta_dot = omega
        omega_dot = -(self.gravity / self.length) * np.sin(theta)
        return np.array([theta_dot, omega_dot])

    def jacobian(self, state):
        """Compute Jacobian: J = [[0, 1], [-(g/L)cos(θ), 0]]"""
        theta, omega = state
        return np.array([
            [0, 1],
            [-(self.gravity / self.length) * np.cos(theta), 0]
        ])

    def implicit_euler_step(self, state, dt):
        """Solve implicit Euler x_{n+1} = x_n + δt·f(x_{n+1}) via Newton-Raphson"""
        state_next = state.copy()
        iterations = 0

        for iterations in range(1, self.newton_max_iter + 1):
            residual = state_next - state - dt * self.dynamics(state_next)
            jac = np.eye(2) - dt * self.jacobian(state_next)
            delta = np.linalg.solve(jac, -residual)
            state_next = state_next + delta

            if np.linalg.norm(residual) < self.newton_tol:
                break

        self.total_newton_iterations += iterations
        return state_next

    def step_doubling(self, state, dt):
        """Compute error via step-doubling: e = ||x̂(δt) - x(2×δt/2)||
        Step size: δt_new = δt(ε_acc/e)^{1/p}"""
        x_full = self.implicit_euler_step(state, dt)
        x_half_1 = self.implicit_euler_step(state, dt / 2)
        x_half_2 = self.implicit_euler_step(x_half_1, dt / 2)

        error = np.linalg.norm(x_full - x_half_2)

        if error > 0:
            dt_new = dt * (self.epsilon_acc / error) ** (1.0 / self.error_order)
        else:
            dt_new = dt * 2.0

        return x_half_2, dt_new, error

    def run(self, verbose=False):
        """Run adaptive simulation: accept if e ≤ ε_acc, reject and retry otherwise"""
        start_time = time.perf_counter()

        state = self.initial_state.copy()
        current_time = 0.0
        dt = self.initial_dt

        self.time = [current_time]
        self.states = [state.copy()]
        self.dt_history = [dt]
        self.errors = [0.0]
        self.accepted_steps = 0
        self.rejected_steps = 0
        self.total_newton_iterations = 0

        while current_time < self.end_time:
            if current_time + dt > self.end_time:
                dt = self.end_time - current_time

            state_next, dt_new, error = self.step_doubling(state, dt)

            if error <= self.epsilon_acc:
                state = state_next
                current_time += dt

                self.time.append(current_time)
                self.states.append(state.copy())
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

        print(f"\nSimulation completed:")
        print(f"  Accepted steps: {self.accepted_steps}")
        print(f"  Rejected steps: {self.rejected_steps}")
        print(f"  Rejection rate: {rejection_rate:.1f}%")
        print(f"  Total Newton iterations: {self.total_newton_iterations}")
        print(f"  Wall clock time: {self.wall_clock_time:.4f}s")
        print(f"  Final time: {self.time[-1]:.6f}s")
        print(f"  Min step size: {min(self.dt_history):.2e}s")
        print(f"  Max step size: {max(self.dt_history):.2e}s")
        print(f"  Mean step size: {np.mean(self.dt_history):.2e}s")
        print(f"  Max error: {max(self.errors):.2e}")

    def get_states_array(self):
        """Return states as numpy array [N×2]"""
        return np.array(self.states)


if __name__ == "__main__":
    integrator = AdaptivePendulum(
        initial_theta=np.pi / 4,
        initial_omega=0.0,
        epsilon_acc=1e-5,
        end_time=10.0,
        initial_dt=0.1
    )

    integrator.run(verbose=True)
