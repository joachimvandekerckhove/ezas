# EZAS: EZ-Diffusion Analysis Suite

A Python library for analyzing decision-making data using the EZ-diffusion model. EZAS helps researchers estimate cognitive parameters from behavioral experiments in psychology and neuroscience.

## What it does

The EZ-diffusion model is a simplified version of the drift-diffusion model that lets you extract three key cognitive parameters from behavioral data:

- **Boundary** - How cautious people are (higher = more cautious, slower but more accurate)
- **Drift** - How good people are at the task (higher = better performance) 
- **Non-decision time** - How long encoding and motor response take (time not spent deciding)

## Module Structure

EZAS provides multiple complementary approaches to parameter estimation:

- **[`base/`](./base/README.md)`** - Core mathematical equations (forward/inverse transformations)
- **[`classes/`](./classes/README.md)`** - Data structures for observations, parameters, and design matrices
- **[`utils/`](./utils/README.md)** - Utility functions for linear algebra, simulation, and result formatting
- **[`qnd/`](./qnd/README.md)** - Quick-and-dirty bootstrap methods for fast uncertainty estimates
- **[`ez_pymc/`](./ez_pymc/README.md)** - Full Bayesian estimation using PyMC for complete uncertainty quantification

These modules have separate README files for further information:

Most files can be run directly with `--demo` to see examples, `--test` to run unit tests, or `--simulation` to run validation studies.

There are also global `demo.py` and `test.py` scripts.  To run a specific demo, call for example:
`demos.py --engine qnd --demo single`

## System Requirements

- **Python**: 3.11.6
- **Linux**: Ubuntu 5.15.0-141-generic
- **PyMC**: 5.10.0

