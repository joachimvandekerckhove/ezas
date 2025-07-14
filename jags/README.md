# JAGS Bayesian Estimation

This directory contains JAGS-based Bayesian estimation methods for EZ-diffusion model parameters. These methods use Just Another Gibbs Sampler (JAGS) to provide robust Bayesian inference with excellent diagnostic capabilities.

## What it does

JAGS methods provide an alternative to PyMC for Bayesian parameter estimation, offering reliable convergence diagnostics and efficient sampling for EZ-diffusion models:

- **`jags_single.py`** - Single-condition parameter estimation using JAGS
- **`jags_multiple.py`** - Multiple-condition parameter estimation for group studies
- **`jags_beta_weights.py`** - Complex experimental designs with design matrices and regression coefficients

The approach uses JAGS's robust MCMC sampling with proper convergence diagnostics (R-hat statistics) to ensure reliable parameter estimates. JAGS is particularly strong at handling complex model structures and provides excellent diagnostic information about chain convergence and mixing.

## Key Features

JAGS estimation offers several advantages:

- **Robust convergence diagnostics** - R-hat statistics and effective sample size calculations
- **Efficient sampling** - JAGS's adaptive sampling often converges faster than other MCMC methods
- **Matrix operations** - Native support for design matrix regression using `%*%` operator
- **Flexible model specification** - Easy to modify and extend model structures
- **Parallel chain execution** - Multiple chains run in parallel for better convergence assessment
- **Speed** - Compilation of models is much faster than alternative engines

All scripts can be run with `--demo` to see examples, `--test` to run unit tests, or `--simulation` to run validation studies. 