"""Fixed time step implicit Euler pendulum with Newton-Raphson solver"""

import numpy as np
import time

class ImplicitEulerPendulum:
    def __init__(self, initial_theta, initial_omega, dt, end_time=10.0,
                 gravity=9.81, length=1.0):
        """
        Initialize implicit Euler pendulum simulator

        Parameters:
            initial_theta: Initial angle (rad), θ₀
            initial_omega: Initial angular velocity (rad/s), ω₀
            dt: Fixed time step size (s)
            end_time: Simulation end time (s)
            gravity: Gravitational acceleration g (m/s²)
            length: Pendulum length L (m)
        """
        self.initial_state = np.array([initial_theta, initial_omega])
        self.dt = dt
        self.end_time = end_time
        self.gravity = gravity
        self.length = length

        self.newton_tol = 1e-10  # Newton-Raphson convergence tolerance
        self.newton_max_iter = 10  # Maximum Newton iterations per step

        # Simulation results
        self.time = []  # t^n at each time step
        self.states = []  # [θ^n, ω^n] at each time step
        self.newton_iterations_per_step = []  # Newton iterations at each time step
        self.residuals_per_step = []  # Residual history for each time step
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

    def implicit_euler_step(self, state):
        """Solve implicit Euler x_{n+1} = x_n + dt·f(x_{n+1}) via Newton-Raphson"""
        state_next = state.copy()
        iterations = 0
        residuals = []

        for iterations in range(1, self.newton_max_iter + 1):
            residual = state_next - state - self.dt * self.dynamics(state_next)
            residual_norm = np.linalg.norm(residual)
            residuals.append(residual_norm)

            jac = np.eye(2) - self.dt * self.jacobian(state_next)
            delta = np.linalg.solve(jac, -residual)
            state_next = state_next + delta

            if residual_norm < self.newton_tol:
                break

        self.total_newton_iterations += iterations
        return state_next, iterations, residuals

    def run(self, verbose=False):
        """Run fixed time step simulation"""
        start_time = time.perf_counter()

        state = self.initial_state.copy()
        current_time = 0.0

        self.time = [current_time]
        self.states = [state.copy()]
        self.newton_iterations_per_step = []
        self.residuals_per_step = []
        self.total_newton_iterations = 0

        while current_time < self.end_time:
            state, iterations, residuals = self.implicit_euler_step(state)
            current_time += self.dt

            self.time.append(current_time)
            self.states.append(state.copy())
            self.newton_iterations_per_step.append(iterations)
            self.residuals_per_step.append(residuals)

        self.wall_clock_time = time.perf_counter() - start_time

        if verbose:
            self.print_summary()

        return self

    def print_summary(self):
        """Print simulation statistics"""
        print(f"\nSimulation completed:")
        print(f"  Total steps: {len(self.newton_iterations_per_step)}")
        print(f"  Total Newton iterations: {self.total_newton_iterations}")
        print(f"  Wall clock time: {self.wall_clock_time:.4f}s")
        print(f"  Final time: {self.time[-1]:.6f}s")

    def get_states_array(self):
        """Return states as numpy array [N×2]"""
        return np.array(self.states)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    integrator = ImplicitEulerPendulum(
        initial_theta=np.pi / 4,
        initial_omega=0.0,
        dt=0.01,
        end_time=10.0
    )

    integrator.run(verbose=False)

    states = integrator.get_states_array()

    plt.figure(figsize=(10, 7))
    plt.plot(integrator.time, states[:, 0], linewidth=2.5, color='#1f77b4')
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Angle (rad)', fontsize=14)
    plt.title('Pendulum Angle Over Time', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()