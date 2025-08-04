# Base Module

This module contains the core mathematical functions for the EZ-diffusion model. It implements the forward and inverse equations that let you convert between model parameters and behavioral data.

## What it does

The EZ-diffusion model is a simplified version of the drift-diffusion model used in cognitive psychology. This module provides three main functions:

- **`forward(params)`** - Takes model parameters (boundary, drift, ndt) and predicts what behavioral data should look like (accuracy, mean response time, response time variance)
- **`inverse(moments)`** - Takes observed behavioral data and estimates what the underlying model parameters should be  
- **`random(lower, upper)`** - Generates random model parameters within specified bounds

These functions are mathematical inverses of each other - you can go from parameters to data and back again. The equations are based on analytical solutions that make parameter estimation much faster than full drift-diffusion model fitting. 