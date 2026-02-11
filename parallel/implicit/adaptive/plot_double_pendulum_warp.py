"""
Global dt vs Per-Thread dt — 30s Chaos Comparison

Runs 5 pendulums with slightly perturbed ICs through both integrators,
then plots side-by-side: trajectories, dt histories, and step-count stats.
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from adaptive_double_pendulum_warp import AdaptiveDoublePendulumWarp
from adaptive_double_pendulum_warp_pt import AdaptiveDoublePendulumWarpPT

# --- Shared configuration ---
num = 5
end_time = 30
epsilon_acc = 1e-3
initial_dt = epsilon_acc

theta1_init = np.array([np.pi / 4 + i * 0.05 for i in range(num)], dtype=np.float32)
theta2_init = np.array([np.pi / 2 + i * 0.05 for i in range(num)], dtype=np.float32)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# --- Run global-dt integrator ---
global_sim = AdaptiveDoublePendulumWarp(
    num_pendulums=num,
    initial_theta1=theta1_init,
    initial_omega1=0.0,
    initial_theta2=theta2_init,
    initial_omega2=0.0,
    epsilon_acc=epsilon_acc,
    end_time=end_time,
    initial_dt=initial_dt,
)
global_sim.run(verbose=True)

# --- Run per-thread-dt integrator ---
pt_sim = AdaptiveDoublePendulumWarpPT(
    num_pendulums=num,
    initial_theta1=theta1_init,
    initial_omega1=0.0,
    initial_theta2=theta2_init,
    initial_omega2=0.0,
    epsilon_acc=epsilon_acc,
    end_time=end_time,
    initial_dt=initial_dt,
)
pt_sim.run(verbose=True)

# --- Plot comparison (4 rows × 2 columns) ---
fig, axes = plt.subplots(4, 2, figsize=(16, 16))

# Row 1: θ₁ trajectories
ax = axes[0, 0]
for i in range(num):
    states = global_sim.get_states_array(i)
    ax.plot(global_sim.time, states[:, 0], linewidth=1.5, color=colors[i],
            label=f'Pendulum {i+1}')
ax.set_xlabel('Time (s)')
ax.set_ylabel('θ₁ (rad)')
ax.set_title('Global dt — θ₁ Trajectories')
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3, linestyle='--')

ax = axes[0, 1]
for i in range(num):
    t = pt_sim.get_time(i)
    states = pt_sim.get_states_array(i)
    ax.plot(t, states[:, 0], linewidth=1.5, color=colors[i],
            label=f'Pendulum {i+1}')
ax.set_xlabel('Time (s)')
ax.set_ylabel('θ₁ (rad)')
ax.set_title('Per-Thread dt — θ₁ Trajectories')
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3, linestyle='--')

# Row 2: dt history
ax = axes[1, 0]
ax.semilogy(global_sim.time, global_sim.dt_history, linewidth=1.5, color='#ff7f0e',
            label='Shared dt')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Step size dt (s)')
ax.set_title('Global dt — Step Size History')
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3, which='both', linestyle='--')

ax = axes[1, 1]
for i in range(num):
    t = pt_sim.get_time(i)
    dt_hist = pt_sim.get_dt_history(i)
    ax.semilogy(t, dt_hist, linewidth=1.5, color=colors[i],
                label=f'Pendulum {i+1}')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Step size dt (s)')
ax.set_title('Per-Thread dt — Step Size History')
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3, which='both', linestyle='--')

# Row 3: error vs time
ax = axes[2, 0]
ax.semilogy(global_sim.time, global_sim.errors, linewidth=1.5, color='#d62728',
            label='Max error across pendulums')
ax.axhline(y=epsilon_acc, color='#2ca02c', linestyle='--', linewidth=1.5,
           label=f'ε = {epsilon_acc:.0e}')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Error estimate')
ax.set_title('Global dt — Local Truncation Error')
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3, which='both', linestyle='--')

ax = axes[2, 1]
for i in range(num):
    t = pt_sim.get_time(i)
    errs = pt_sim.get_errors(i)
    ax.semilogy(t, np.maximum(errs, 1e-16), linewidth=1.5, color=colors[i],
                label=f'Pendulum {i+1}')
ax.axhline(y=epsilon_acc, color='#2ca02c', linestyle='--', linewidth=1.5,
           label=f'ε = {epsilon_acc:.0e}')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Error estimate')
ax.set_title('Per-Thread dt — Local Truncation Error')
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3, which='both', linestyle='--')

# Row 4: accepted steps bar chart
ax = axes[3, 0]
x = np.arange(num)
width = 0.35
global_accepted = [global_sim.accepted_steps] * num  # same for all pendulums
global_rejected = [global_sim.rejected_steps] * num
bars1 = ax.bar(x - width/2, global_accepted, width, label='Accepted', color='#2ca02c')
bars2 = ax.bar(x + width/2, global_rejected, width, label='Rejected', color='#d62728')
ax.set_xlabel('Pendulum')
ax.set_ylabel('Steps')
ax.set_title('Global dt — Step Counts')
ax.set_xticks(x)
ax.set_xticklabels([f'P{i+1}' for i in range(num)])
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

ax = axes[3, 1]
bars1 = ax.bar(x - width/2, pt_sim.accepted_steps, width, label='Accepted', color='#2ca02c')
bars2 = ax.bar(x + width/2, pt_sim.rejected_steps, width, label='Rejected', color='#d62728')
ax.set_xlabel('Pendulum')
ax.set_ylabel('Steps')
ax.set_title('Per-Thread dt — Step Counts')
ax.set_xticks(x)
ax.set_xticklabels([f'P{i+1}' for i in range(num)])
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

fig.suptitle('Global dt vs Per-Thread dt — ' + str(end_time) + '(s) Double Pendulum Comparison',
             fontsize=14, fontweight='bold')
fig.tight_layout()
plt.show()
