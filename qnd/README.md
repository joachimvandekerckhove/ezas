# QND (Quick and Dirty) Estimation Methods

This directory contains "Quick and Dirty" (QND) estimation methods for EZ-diffusion model parameters using bootstrap resampling. These methods provide fast, approximate parameter estimation with uncertainty quantification.

## Overview

QND methods use bootstrap resampling to estimate parameter uncertainty without requiring full Bayesian inference. They are designed for:
- **Speed**: Much faster than MCMC-based methods
- **Simplicity**: Easy to implement and understand
- **Robustness**: Works well across different parameter ranges
- **Calibration**: Provides well-calibrated credible intervals

## Scripts

### 1. `qnd_single.py` - Single Parameter Estimation

**Purpose**: Estimate EZ-diffusion parameters for a single participant/condition using bootstrap resampling.

**Key Functions**:
- `qnd_single_estimation(observations, n_repetitions=1000)`: Main estimation function
- `demo()`: Demonstration with default parameters
- `simulation()`: Run simulation study to test calibration

**Usage**:
```bash
# Run demo
python qnd_single.py --demo

# Run simulation study
python qnd_single.py --simulation

# Run tests
python qnd_single.py --test
```

**Example**:
```python
from vendor.ezas.qnd.qnd_single import qnd_single_estimation
from vendor.ezas.classes.moments import Observations

# Create observations from behavioral data
obs = Observations(accuracy=75, mean_rt=0.45, var_rt=0.12, sample_size=100)

# Estimate parameters with uncertainty
estimated_params = qnd_single_estimation(obs, n_repetitions=1000)
print(f"Boundary: {estimated_params.boundary():.3f} ± {estimated_params.boundary_sd():.3f}")
```

### 2. `qnd_beta_weights.py` - Design Matrix Estimation

**Purpose**: Estimate parameters for complex experimental designs using design matrices and beta weights.

**Key Functions**:
- `qnd_beta_weights_estimation(observations, working_matrix)`: Main estimation function
- `demo()`: Demonstration with example design matrix
- `simulation()`: Run simulation study
- `simulation_parallel()`: Parallel version for faster execution

**Usage**:
```bash
# Run demo
python qnd_beta_weights.py --demo

# Run simulation
python qnd_beta_weights.py --simulation

# Run parallel simulation
python qnd_beta_weights.py --simulation-parallel

# Run tests
python qnd_beta_weights.py --test
```

**Example**:
```python
from vendor.ezas.qnd.qnd_beta_weights import qnd_beta_weights_estimation
from vendor.ezas.classes.design_matrix import DesignMatrix

# Define experimental design
design_matrix = DesignMatrix(
    boundary_design=np.array([[1, 0], [0, 1]]),
    drift_design=np.array([[1, 0], [0, 1]]),
    ndt_design=np.array([[1, 0], [0, 1]]),
    boundary_weights=np.array([1.0, 1.5]),
    drift_weights=np.array([0.5, 0.8]),
    ndt_weights=np.array([0.2, 0.3])
)

# Estimate beta weights
estimated_weights, estimated_params = qnd_beta_weights_estimation(
    observations_list, working_matrix
)
```

### 3. `qnd_accuracy_demo.py` - Calibration Analysis

**Purpose**: Comprehensive analysis of QND estimation accuracy and calibration with visualizations.

**Key Features**:
- Parameter recovery analysis
- Coverage rate assessment
- Timing analysis
- Design matrix evaluation
- Publication-ready figures

**Usage**:
```bash
# Quick demo (fewer simulations, faster)
python qnd_accuracy_demo.py --quick

# Full demo (comprehensive analysis)
python qnd_accuracy_demo.py --demo
```

**Output**:
- `qnd_calibration_quick_demo.png`: Quick calibration figure
- `qnd_calibration_demo.png`: Full calibration analysis
- Console output with summary statistics

## Key Features

### Bootstrap Resampling
All QND methods use bootstrap resampling to estimate parameter uncertainty:
1. **Resample** the observed data multiple times
2. **Estimate parameters** for each resampled dataset
3. **Compute statistics** (mean, std, quantiles) across estimates

### Calibration
The methods provide well-calibrated 95% credible intervals:
- **Marginal coverage**: ~95% for individual parameters
- **Joint coverage**: ~0.95^3 ≈ 86% for all three parameters together

### Speed
QND methods are significantly faster than Bayesian alternatives:
- **Single estimation**: ~15ms for 1000 bootstrap samples
- **Design matrix**: ~100ms for complex designs
- **Parallel execution**: Available for large simulation studies

## Performance Characteristics

### Accuracy
- **Parameter recovery**: Excellent across wide parameter ranges
- **Coverage rates**: Close to nominal 95% levels
- **Bias**: Minimal systematic bias
- **Precision**: Improves with sample size

### Computational Efficiency
- **Time complexity**: O(bootstrap_samples)
- **Memory usage**: O(design_cells)
- **Scalability**: Parallel execution available

## Best Practices

### Sample Sizes
- **Minimum**: 20 trials per condition
- **Recommended**: 100+ trials per condition
- **Optimal**: 200+ trials for complex designs

### Bootstrap Repetitions
- **Quick estimates**: 100-500 repetitions
- **Standard analysis**: 1000 repetitions
- **High precision**: 2000+ repetitions

### Design Matrix Usage
- **Simple designs**: Use `qnd_single.py`
- **Complex designs**: Use `qnd_beta_weights.py`
- **Multiple conditions**: Always use design matrix approach

## Integration with EZAS

QND methods integrate with the broader EZAS module:

```python
from vendor.ezas import Parameters, Observations
from vendor.ezas.qnd.qnd_single import qnd_single_estimation

# Use with EZAS data structures
params = Parameters(boundary=1.5, drift=0.8, ndt=0.2)
moments = ez.forward(params)
observations = moments.sample(100)

# Estimate with QND
estimated = qnd_single_estimation(observations)
```

## Testing

All scripts include custom test suites:

```bash
# Test individual scripts
python qnd_single.py --test
python qnd_beta_weights.py --test

# Run calibration analysis
python qnd_accuracy_demo.py --quick
```

## Output Files

The scripts generate various output files:
- **Figures**: PNG and PDF calibration plots
- **Cache**: Pickled simulation results for faster re-runs
- **Console**: Detailed analysis reports

For more flexible but slower estimation, consider using the Bayesian methods in the `bayesian/` directory. 

(c)2025 Joachim Vandekerckhove