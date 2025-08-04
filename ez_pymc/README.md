# PyMC Bayesian Estimation

This directory contains full Bayesian estimation methods for EZ-diffusion model parameters using PyMC. These methods provide complete uncertainty quantification through MCMC sampling.

## What it does

PyMC methods use Markov Chain Monte Carlo (MCMC) to sample from the full posterior distribution of parameters, giving you the most complete uncertainty information possible:

- **`pymc_single.py`** - Full Bayesian estimation for a single participant or condition
- **`pymc_multiple.py`** - Handle multiple participants/conditions simultaneously
- **`pymc_beta_weights.py`** - Complex experimental designs with design matrices and beta weights

The approach uses proper Bayesian priors and likelihood functions to estimate not just parameter values but their full posterior distributions. This gives you credible intervals, posterior correlations, and diagnostic information about convergence. It's slower than QND methods but provides the gold standard for uncertainty quantification.

All scripts can be run with `--demo` to see examples, `--test` to run unit tests, or `--simulation` to run validation studies. 
