# QND (Quick and Dirty) Estimation

This directory contains "Quick and Dirty" (QND) estimation methods for EZ-diffusion model parameters. These bootstrap methods are much faster than Bayesian approaches but still provide uncertainty estimates.

## What it does

QND methods use bootstrap resampling to estimate parameter uncertainty without needing full Bayesian inference. They're designed for speed while still giving you credibility intervals:

- **`qnd_single.py`** - Estimate parameters for a single participant or condition
- **`qnd_beta_weights.py`** - Handle complex experimental designs with multiple conditions  
- **`qnd_inference.py`** - Parameter inference and comparison methods

The basic approach is simple: resample your data many times, estimate parameters for each resample, then compute statistics across all the estimates. This gives you means, standard deviations, and credible intervals much faster than MCMC.  The unique structure of EZ-diffusion makes the data resampling highly efficient.

All scripts can be run with `--demo` to see examples, `--test` to run unit tests, or `--simulation` to run validation studies.