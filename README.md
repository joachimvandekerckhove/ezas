# EZAS: EZ-Diffusion Analysis Suite

A Python library for analyzing decision-making data using the EZ-diffusion model. EZAS provides tools for parameter estimation, Bayesian inference, and simulation studies in cognitive psychology and neuroscience research.

## Overview

The EZ-diffusion model is a simplified version of the drift-diffusion model that allows for efficient parameter estimation from behavioral data (accuracy and response time statistics). This library implements the EZ-diffusion equations and provides multiple estimation methods including:

- **Analytical estimation** using the EZ equations
- **Bayesian parameter estimation** using PyMC
- **Quick-and-dirty estimation** using a bootstrap method
- **Design matrix** approaches for linear experimental designs

## Features

### Core Functionality
- **Forward equations**: Compute predicted mean accuracy, mean RT, and RT variance from EZ-diffusion parameters
- **Inverse equations**: Estimate EZ-diffusion parameters from observed behavioral statistics
- **Parameter classes**: Structured representation of boundary, drift, and non-decision time parameters
- **Moments and Observations**: Handle summary statistics and raw behavioral data

### Estimation Methods
- **Single-subject estimation**: Estimate parameters for individual participants
- **Multi-subject estimation**: Pool data across multiple participants
- **Design matrix estimation**: Handle complex experimental designs with multiple conditions
- **Bootstrap estimation**: For real-time estimation with correct uncertainty
- **Bayesian inference**: Full posterior distributions with uncertainty quantification

### Utilities
- **Simulation tools**: Generate synthetic data for method validation
- **Convergence diagnostics**: MCMC convergence checking (R-hat statistics)
- **Statistical summaries**: Parameter means, standard deviations, and credible intervals

### Demos, tests, simulations
Most files in the module can be executed directly with a `--demo` flag to demonstrate functionality.
Most additionally have a `--test` flag for unit testing.  Some have a `--simulation` flag that will
run a short simulation study (which may take a few minutes), and a few have a `--parallel` flag to
run a simulation study on multiple cores.

## Installation

### Prerequisites
This module was developed with
- Python `3.11.6`
- NumPy `1.25.2`
- PyMC `5.10.0`
- ArviZ `0.20.0`

### Dependencies
```bash
pip install numpy pymc arviz
```

## Quick Start

### Basic Usage

```python
from vendor.ezas import Parameters, Moments, forward, inverse

# Define EZ-diffusion parameters
params = Parameters(boundary=1.5, drift=0.8, ndt=0.2)

# Forward equations: predict behavioral statistics
moments = forward(params)
print(f"Predicted accuracy: {moments.accuracy:.3f}")
print(f"Predicted mean RT: {moments.mean_rt:.3f}")
print(f"Predicted RT variance: {moments.var_rt:.3f}")

# Inverse equations: estimate parameters from data
estimated_params = inverse(moments)
print(f"Estimated boundary: {estimated_params.boundary():.3f}")
print(f"Estimated drift: {estimated_params.drift():.3f}")
print(f"Estimated NDT: {estimated_params.ndt():.3f}")
```

### Bayesian Parameter Estimation

```python
from vendor.ezas.bayesian.bayes_single import bayesian_parameter_estimation
from vendor.ezas.classes.moments import Observations

# Create observations from behavioral data
obs = Observations(
    accuracy=75,      # Number of correct responses
    mean_rt=0.45,     # Mean response time
    var_rt=0.12,      # Response time variance
    sample_size=100   # Total number of trials
)

# Estimate parameters using Bayesian inference
estimated_params = bayesian_parameter_estimation(
    observations=obs,
    n_samples=2000,   # MCMC samples
    n_tune=1000,      # Tuning steps
    verbosity=2       # Output level
)

# Results include uncertainty estimates
print(f"Boundary: {estimated_params.boundary():.3f} ± {estimated_params.boundary_sd():.3f}")
print(f"Drift: {estimated_params.drift():.3f} ± {estimated_params.drift_sd():.3f}")
print(f"NDT: {estimated_params.ndt():.3f} ± {estimated_params.ndt_sd():.3f}")
```

### Design Matrix Analysis

```python
import numpy as np
from vendor.ezas.classes.design_matrix import DesignMatrix
from vendor.ezas.bayesian.bayes_dmat import bayesian_design_matrix_parameter_estimation

# Define experimental design
design_matrix = DesignMatrix(
    boundary_design=np.array([[1, 0], [0, 1]]),  # Two conditions
    drift_design=np.array([[1, 0], [0, 1]]),
    ndt_design=np.array([[1, 0], [0, 1]])
)

# Estimate parameters for multiple conditions
estimated_weights = bayesian_design_matrix_parameter_estimation(
    observations_list=[obs1, obs2],  # Observations for each condition
    sample_sizes=[100, 100],         # Sample size for each condition
    design_matrix=design_matrix      # The DesignMatrix object
)
```

## Project Structure

```
vendor/ezas/
├── base/                    # Core EZ-diffusion equations
│   └── ez_equations.py     # Forward and inverse equations
├── classes/                 # Data structures
│   ├── parameters.py       # Parameter representation
│   ├── moments.py          # Summary statistics
│   └── design_matrix.py    # Experimental design handling
├── bayesian/               # Bayesian estimation methods
│   ├── bayes_single.py     # Single-subject estimation
│   ├── bayes_multiple.py   # Multi-subject estimation
│   └── bayes_dmat.py       # Design matrix estimation
├── qnd/                    # Quasi-Newton Descent methods
│   ├── qnd_single.py       # Single-subject QND
│   └── qnd_beta_weights.py # QND with design matrices
├── utils/                  # Utility functions
│   ├── simulation.py       # Data simulation
│   ├── posterior.py        # Posterior analysis
│   └── debug.py           # Debugging utilities
├── demos.py               # Usage examples
└── test.py                # Unit tests
```

## API Reference

### Core Classes

#### `Parameters(boundary, drift, ndt, ...)`
Represents EZ-diffusion model parameters with optional uncertainty estimates.

**Attributes:**
- `boundary`: Decision boundary (positive float)
- `drift`: Drift rate (float)
- `ndt`: Non-decision time (positive float)
- `boundary_sd`, `drift_sd`, `ndt_sd`: Standard deviations (optional)
- `boundary_lower_bound`, `boundary_upper_bound`, etc.: 95% credible intervals (optional)

#### `Moments(accuracy, mean_rt, var_rt)`
Represents predicted or observed behavioral statistics.

**Attributes:**
- `accuracy`: Proportion correct (0-1)
- `mean_rt`: Mean response time
- `var_rt`: Response time variance

#### `Observations(accuracy, mean_rt, var_rt, sample_size)`
Represents observed behavioral data with sample size information.

### Core Functions

#### `forward(params)`
Compute predicted behavioral statistics from EZ-diffusion parameters.

#### `inverse(moments)`
Estimate EZ-diffusion parameters from behavioral statistics.

#### `bayesian_parameter_estimation(observations, n_samples=2000, n_tune=1000)`
Estimate parameters using Bayesian inference with MCMC sampling.

## Examples

### Simulation Study
```python
from vendor.ezas.utils.simulation import run_simulation
from vendor.ezas import Parameters

# Run simulation across different sample sizes
lower_bound = Parameters(0.5, 1.0, 0.2)
upper_bound = Parameters(2.0, 2.0, 0.5)

for N in [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240]:
    results = run_simulation(N, lower_bound, upper_bound)
    print(f"N = {N}: {results}")
```

### Parameter Recovery
```python
from vendor.ezas import Parameters, forward, inverse

# True parameters
true_params = Parameters(boundary=1.5, drift=0.8, ndt=0.2)

# Generate predicted moments
predicted_moments = forward(true_params)

# Estimate parameters
estimated_params = inverse(predicted_moments)

# Check recovery
print(f"True boundary: {true_params.boundary():.3f}")
print(f"Estimated boundary: {estimated_params.boundary():.3f}")
```

## Testing

Run the test suite to verify installation:

```bash
# Run all tests
python -m pytest vendor/ezas/

# Run specific test modules
python vendor/ezas/base/ez_equations.py --test
python vendor/ezas/classes/parameters.py --test
python vendor/ezas/bayesian/bayes_single.py --test
```

## Demos

Explore usage examples:

```bash
# Basic demo
python vendor/ezas/base/ez_equations.py --demo

# Bayesian estimation demo
python vendor/ezas/bayesian/bayes_single.py --demo

# QND estimation demo
python vendor/ezas/qnd_single.py --demo
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Citation

If you use EZAS in your research, please cite:

```bibtex
@software{ezas2024,
  title={EZAS: EZ-Diffusion Analysis Suite},
  author={Joachim Vandekerckhove},
  year={2025},
  url={https://github.com/joachimvandekerckhove/ezas}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, issues, or feature requests, please open an issue on the project repository.

## References

  - Chávez De la Peña, A. F., & Vandekerckhove, J. (in press). An EZ Bayesian hierarchical drift diffusion model for response time and accuracy. *Psychonomic Bulletin & Review.*
  - Wagenmakers, E. J., van der Maas, H. L., & Grasman, R. P. (2007). An EZ-diffusion model for response time and accuracy. *Psychonomic Bulletin & Review*, 14(1), 3-22.
