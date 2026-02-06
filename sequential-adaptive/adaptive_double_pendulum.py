"""
Adaptive Time Stepping Double Pendulum - Step-Doubling Error Control

Pure implementation of the step-doubling method for a double pendulum system.
The integrator computes two estimates:
  - x̂^{n+1}: computed with one full step of size δt (equation 24)
  - x^{n+1}: computed with two half-steps of size δt/2 (equations 22-23)

The local truncation error is estimated as:
  e^{n+1} = ||x̂^{n+1} - x^{n+1}|| ≈ c̄δt^p

where p is the order of the error estimate. The step size is then adjusted to
maintain a user-specified accuracy ε_acc using the pure formula:
  δt_new ← δt(ε_acc/e^{n+1})^{1/p}

State vector: [θ₁, ω₁, θ₂, ω₂] (4-state system)
"""

import numpy as np
import time

class AdaptiveDoublePendulum:
    def __init__(self, initial_theta1, initial_omega1,
                 initial_theta2, initial_omega2, epsilon_acc,
                 end_time=10.0, initial_dt=0.1,
                 gravity=9.81, length1=1.0, length2=1.0,
                 mass1=1.0, mass2=1.0):
        """
        Initialize adaptive double pendulum simulator

        Parameters:
            initial_theta1: Initial angle of first pendulum (rad), θ₁₀
            initial_omega1: Initial angular velocity of first pendulum (rad/s), ω₁₀
            initial_theta2: Initial angle of second pendulum (rad), θ₂₀
            initial_omega2: Initial angular velocity of second pendulum (rad/s), ω₂₀
            epsilon_acc: Target accuracy ε_acc for error control
            end_time: Simulation end time (s)
            initial_dt: Initial time step size δt₀ (s)
            gravity: Gravitational acceleration g (m/s²)
            length1: First pendulum length L₁ (m)
            length2: Second pendulum length L₂ (m)
            mass1: First pendulum mass m₁ (kg)
            mass2: Second pendulum mass m₂ (kg)
        """
        self.initial_state = np.array([initial_theta1, initial_omega1,
                                       initial_theta2, initial_omega2])
        self.epsilon_acc = epsilon_acc
        self.end_time = end_time
        self.initial_dt = initial_dt
        self.gravity = gravity
        self.length1 = length1
        self.length2 = length2
        self.mass1 = mass1
        self.mass2 = mass2

        self.error_order = 1  # p: order of error estimate (1 for Euler)
        self.newton_tol = 1e-10  # Newton-Raphson convergence tolerance
        self.newton_max_iter = 20  # Maximum Newton iterations per step

        # Simulation results
        self.time = []  # t^n at each accepted step
        self.states = []  # [θ₁^n, ω₁^n, θ₂^n, ω₂^n] at each accepted step
        self.dt_history = []  # δt used at each accepted step
        self.errors = []  # e^{n+1} at each accepted step
        self.accepted_steps = 0  # Number of accepted steps
        self.rejected_steps = 0  # Number of rejected steps
        self.total_newton_iterations = 0  # Total Newton iterations (computational cost)
        self.wall_clock_time = 0.0  # Wall clock time (s)

    def dynamics(self, state):
        """
        Compute state derivative for double pendulum using Lagrangian mechanics

        State: [θ₁, ω₁, θ₂, ω₂]
        Returns: [θ̇₁, ω̇₁, θ̇₂, ω̇₂] = [ω₁, α₁, ω₂, α₂]
        """
        theta1, omega1, theta2, omega2 = state
        g = self.gravity
        L1 = self.length1
        L2 = self.length2
        m1 = self.mass1
        m2 = self.mass2

        delta_theta = theta1 - theta2
        sin_delta = np.sin(delta_theta)
        cos_delta = np.cos(delta_theta)

        # Denominator for both angular accelerations
        denom = L1 * (2 * m1 + m2 - m2 * np.cos(2 * delta_theta))

        # Angular acceleration of first pendulum (α₁ = ω̇₁)
        num1 = (-g * (2 * m1 + m2) * np.sin(theta1)
                - m2 * g * np.sin(theta1 - 2 * theta2)
                - 2 * sin_delta * m2 * (omega2**2 * L2 + omega1**2 * L1 * cos_delta))
        alpha1 = num1 / denom

        # Angular acceleration of second pendulum (α₂ = ω̇₂)
        num2 = (2 * sin_delta * (omega1**2 * L1 * (m1 + m2)
                + g * (m1 + m2) * np.cos(theta1)
                + omega2**2 * L2 * m2 * cos_delta))
        alpha2 = num2 / (L2 * (2 * m1 + m2 - m2 * np.cos(2 * delta_theta)))

        return np.array([omega1, alpha1, omega2, alpha2])

    def jacobian(self, state):
        """
        Compute 4×4 Jacobian matrix J = ∂f/∂x for Newton-Raphson

        J[i,j] = ∂f_i/∂x_j where f = [ω₁, α₁, ω₂, α₂], x = [θ₁, ω₁, θ₂, ω₂]
        """
        theta1, omega1, theta2, omega2 = state
        g = self.gravity
        L1 = self.length1
        L2 = self.length2
        m1 = self.mass1
        m2 = self.mass2

        delta_theta = theta1 - theta2
        sin_delta = np.sin(delta_theta)
        cos_delta = np.cos(delta_theta)
        sin_2delta = np.sin(2 * delta_theta)
        cos_2delta = np.cos(2 * delta_theta)

        denom1 = L1 * (2 * m1 + m2 - m2 * cos_2delta)
        denom2 = L2 * (2 * m1 + m2 - m2 * cos_2delta)

        # Numerical differentiation for complex derivatives (more stable)
        eps = 1e-8
        jac = np.zeros((4, 4))

        # Row 0: ∂θ̇₁/∂x = [0, 1, 0, 0]
        jac[0, 0] = 0
        jac[0, 1] = 1
        jac[0, 2] = 0
        jac[0, 3] = 0

        # Row 2: ∂θ̇₂/∂x = [0, 0, 0, 1]
        jac[2, 0] = 0
        jac[2, 1] = 0
        jac[2, 2] = 0
        jac[2, 3] = 1

        # Rows 1 and 3: numerical differentiation for α₁ and α₂
        for j in range(4):
            state_plus = state.copy()
            state_plus[j] += eps
            state_minus = state.copy()
            state_minus[j] -= eps

            f_plus = self.dynamics(state_plus)
            f_minus = self.dynamics(state_minus)

            jac[1, j] = (f_plus[1] - f_minus[1]) / (2 * eps)  # ∂α₁/∂x_j
            jac[3, j] = (f_plus[3] - f_minus[3]) / (2 * eps)  # ∂α₂/∂x_j

        return jac

    def implicit_euler_step(self, state, dt):
        """Solve implicit Euler x_{n+1} = x_n + δt·f(x_{n+1}) via Newton-Raphson"""
        state_next = state.copy()
        iterations = 0

        for iterations in range(1, self.newton_max_iter + 1):
            residual = state_next - state - dt * self.dynamics(state_next)
            jac = np.eye(4) - dt * self.jacobian(state_next)
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

        # Safety bounds on step size changes
        dt_new = max(dt_new, dt * 0.1)  # Don't shrink more than 10x
        dt_new = min(dt_new, dt * 2.0)  # Don't grow more than 2x
        dt_new = max(dt_new, 1e-10)     # Minimum step size

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

        print(f"\nDouble Pendulum Simulation completed:")
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
        """Return states as numpy array [N×4]"""
        return np.array(self.states)


if __name__ == "__main__":
    integrator = AdaptiveDoublePendulum(
        initial_theta1=np.pi / 4,
        initial_omega1=0.0,
        initial_theta2=np.pi / 2,
        initial_omega2=0.0,
        epsilon_acc=1e-5,
        end_time=10.0,
        initial_dt=0.1
    )

    integrator.run(verbose=True)
