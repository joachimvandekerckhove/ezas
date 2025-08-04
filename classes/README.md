# Classes Module

This module defines the main data structures used throughout the EZAS library. These classes represent the different types of data you work with when analyzing decision-making experiments.

## What it contains

The module provides five main classes that represent different aspects of decision-making data:

- **`Parameters`** - Model parameters like boundary, drift rate, and non-decision time. These represent the underlying cognitive processes
- **`Moments`** - Summary statistics from behavioral data like accuracy, mean response time, and response time variance
- **`Observations`** - Raw experimental data with individual response times and accuracy values for each trial
- **`DesignMatrix`** - Experimental design specifications that let you handle multi-condition studies
- **`BetaWeights`** - Regression coefficients that show how different experimental conditions affect each parameter

These classes work together to handle the flow from raw data to model parameters. You typically start with `Observations` (raw data), compute `Moments` (summary statistics), then estimate `Parameters` (model parameters). For complex experiments, you use `DesignMatrix` and `BetaWeights` to handle multiple conditions simultaneously. 