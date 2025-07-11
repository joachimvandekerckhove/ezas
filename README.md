# EZAS: EZ-Diffusion Analysis Suite

A Python library for analyzing decision-making data using the EZ-diffusion model. EZAS helps researchers estimate cognitive parameters from behavioral experiments in psychology and neuroscience.

## What it does

The EZ-diffusion model is a simplified version of the drift-diffusion model that lets you extract three key cognitive parameters from behavioral data:

- **Boundary** - How cautious people are (higher = more cautious, slower but more accurate)
- **Drift** - How good people are at the task (higher = better performance) 
- **Non-decision time** - How long encoding and motor response take (time not spent deciding)

EZAS provides multiple ways to estimate these parameters:
- **Analytical estimation** - Fast mathematical formulas
- **Bayesian estimation** - Full uncertainty quantification using MCMC
- **Quick-and-dirty estimation** - Bootstrap methods for speed
- **Design matrix approaches** - Handle complex multi-condition experiments

Most files can be run directly with `--demo` to see examples or `--test` to run unit tests.
