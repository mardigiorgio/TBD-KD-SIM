# Numerical Integrator Research

Research repository comparing numerical integration methods for rigid contact dynamics with sequential (CPU) and parallel (GPU) implementations using NVIDIA Warp.

Features adaptive and fixed step-size integrators (explicit/implicit Euler) with Drake-style step doubling error control. Includes scalability and work-precision benchmarking.

## Installation

This project uses [UV](https://github.com/astral-sh/uv) for fast, reliable dependency management.

```bash
# Install dependencies
uv sync

# Run a simulation
uv run python parallel/implicit/adaptive/adaptive_double_pendulum_warp.py
```
