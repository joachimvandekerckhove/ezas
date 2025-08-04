# MCMC Fallback Implementation

This directory contains a fallback MCMC implementation for EZ-diffusion parameter estimation. This provides a pure Python alternative when JAGS is not available, using a Metropolis-Hastings sampler with adaptive proposal tuning.

## What it does

The MCMC fallback provides Bayesian parameter estimation using a custom Metropolis-Hastings implementation:

- **`ez_mcmc.py`** - Complete MCMC implementation with adaptive proposal covariance, convergence diagnostics, and uncertainty quantification
- **`__init__.py`** - Module interface that exports the fallback implementation

This implementation maintains the same interface as the JAGS version while providing a self-contained solution that doesn't require external dependencies beyond standard scientific Python libraries.

## Key Features

The fallback MCMC implementation offers several important capabilities:

- **Adaptive proposal tuning** - Automatically adjusts proposal covariance during burn-in to achieve optimal acceptance rates
- **Multiple chain execution** - Runs multiple chains in parallel for robust convergence assessment
- **R-hat diagnostics** - Computes Gelman-Rubin R-hat statistics to assess chain convergence
- **Progress tracking** - Uses tqdm for real-time progress monitoring during long runs
- **Comprehensive testing** - Includes unit tests and parameter recovery validation studies
- **Simulation studies** - Built-in functions for evaluating estimator performance across different parameter regimes

The implementation uses proper Bayesian priors and handles parameter bounds correctly, ensuring reliable parameter estimates even when JAGS is unavailable.

## Usage

The script can be run with different modes:
- `--demo` - Demonstrates basic functionality with synthetic data
- `--test` - Runs comprehensive unit tests
- `--simulation` - Performs parameter recovery validation studies

The main function `mcmc` provides the same interface as other EZAS estimation methods, making it a drop-in replacement when JAGS is not available. 