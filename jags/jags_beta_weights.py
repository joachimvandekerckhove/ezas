#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import unittest
import argparse
import tempfile
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Tuple
from scipy.stats import truncnorm
from vendor.ezas.base import ez_equations as ez
from vendor.ezas.classes.moments import Observations
from vendor.ezas.classes.parameters import Parameters
from vendor.ezas.classes.design_matrix import DesignMatrix, BetaWeights
from vendor.ezas.qnd.qnd_beta_weights import qnd_beta_weights_estimation
from vendor.ezas.utils.debug import announce

import vendor.py2jags.src as p2
from vendor.py2jags.src.mcmc_samples import MCMCSamples

_MAX_R_HAT = 1.1

# Example design matrix, we'll use this for the demo and simulation
_EXAMPLE_DESIGN_MATRIX = DesignMatrix(
        boundary_design = np.array([[0, 0, 1], 
                                    [0, 1, 0], 
                                    [1, 0, 0], 
                                    [1, 0, 0]]),
        drift_design    = np.array([[0, 0, 1], 
                                    [0, 1, 0], 
                                    [1, 0, 0], 
                                    [0, 1, 0]]),
        ndt_design      = np.array([[0, 0, 1], 
                                    [0, 1, 0], 
                                    [1, 0, 0], 
                                    [0, 0, 1]]),
        boundary_weights = np.array([2.0, 1.5, 1.0]),
        drift_weights    = np.array([1.2, 0.8, 0.4]),
        ndt_weights      = np.array([0.4, 0.3, 0.2])
)

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10



def mcmc_samples_to_beta_weights(
    mcmc_samples: MCMCSamples, 
    working_matrix: DesignMatrix,
    verbosity: int = 0
) -> Tuple[BetaWeights, list[Parameters]]:
    """Convert MCMC samples to BetaWeights and Parameters objects."""
    
    # Get design matrices
    boundary_design = working_matrix.boundary_design()
    drift_design = working_matrix.drift_design()
    ndt_design = working_matrix.ndt_design()
    
    
    # Extract boundary weights statistics (samples are on original scale)
    boundary_weights_mean = []
    boundary_weights_sd = []
    boundary_weights_lower = []
    boundary_weights_upper = []
    
    for i in range(boundary_design.shape[1]):
        param_name = f'boundary_weights_{i+1}'
        stats = mcmc_samples.stats[param_name]
        # Use original scale values directly
        boundary_weights_mean.append(stats['mean'])
        boundary_weights_sd.append(stats['std'])
        boundary_weights_lower.append(stats['q025'])
        boundary_weights_upper.append(stats['q975'])
    
    # Extract drift weights statistics (samples are on original scale)
    drift_weights_mean = []
    drift_weights_sd = []
    drift_weights_lower = []
    drift_weights_upper = []
    
    for i in range(drift_design.shape[1]):
        param_name = f'drift_weights_{i+1}'
        stats = mcmc_samples.stats[param_name]
        # Use original scale values directly
        drift_weights_mean.append(stats['mean'])
        drift_weights_sd.append(stats['std'])
        drift_weights_lower.append(stats['q025'])
        drift_weights_upper.append(stats['q975'])
    
    # Extract ndt weights statistics (samples are on original scale)
    ndt_weights_mean = []
    ndt_weights_sd = []
    ndt_weights_lower = []
    ndt_weights_upper = []
    
    for i in range(ndt_design.shape[1]):
        param_name = f'ndt_weights_{i+1}'
        stats = mcmc_samples.stats[param_name]
        # Use original scale values directly
        ndt_weights_mean.append(stats['mean'])
        ndt_weights_sd.append(stats['std'])
        ndt_weights_lower.append(stats['q025'])
        ndt_weights_upper.append(stats['q975'])
    
    # Create BetaWeights object
    beta_weights = BetaWeights(
        beta_boundary_mean=np.array(boundary_weights_mean),
        beta_drift_mean=np.array(drift_weights_mean),
        beta_ndt_mean=np.array(ndt_weights_mean),
        beta_boundary_sd=np.array(boundary_weights_sd),
        beta_drift_sd=np.array(drift_weights_sd),
        beta_ndt_sd=np.array(ndt_weights_sd),
        beta_boundary_lower=np.array(boundary_weights_lower),
        beta_drift_lower=np.array(drift_weights_lower),
        beta_ndt_lower=np.array(ndt_weights_lower),
        beta_boundary_upper=np.array(boundary_weights_upper),
        beta_drift_upper=np.array(drift_weights_upper),
        beta_ndt_upper=np.array(ndt_weights_upper)
    )
    
    # Compute parameters from design matrices and weights
    boundary_params = boundary_design @ beta_weights.beta_boundary_mean()
    drift_params = drift_design @ beta_weights.beta_drift_mean()
    ndt_params = ndt_design @ beta_weights.beta_ndt_mean()
    
    # Create Parameters objects
    parameters = [
        Parameters(boundary=b, drift=d, ndt=n)
        for b, d, n in zip(boundary_params, drift_params, ndt_params)
    ]
    
    return beta_weights, parameters

def bayesian_design_matrix_parameter_estimation(
    observations: list[Observations],
    working_matrix: DesignMatrix,
    n_samples: int = 2000,
    n_tune: int = 1000,
    n_chains: int = 4,
    verbosity: int = 2,
    jags_executable: str = 'jags'
) -> Tuple[BetaWeights, list[Parameters], MCMCSamples]:
    """Estimate EZ-diffusion beta weights using JAGS given observed statistics and design matrices.
    
    Args:
        observations: Observed summary statistics (list of Observations)
        working_matrix: Design matrix for parameter estimation
        n_samples: Number of MCMC samples to draw
        n_tune: Number of tuning steps for MCMC
        n_chains: Number of MCMC chains
        verbosity: Verbosity level
        jags_executable: Path to JAGS executable
        
    Returns:
        Tuple containing:
        - Beta weights (BetaWeights)
        - Estimated parameters (list of Parameters)
        - MCMC samples (MCMCSamples)
    """
    if not isinstance(observations, list):
        raise TypeError("observations must be a list")
    if not all(isinstance(x, Observations) for x in observations):
        raise TypeError("All elements in observations must be Observations instances")
    
    # Use the inverse equations to get quick and dirty estimates of the weights
    qnd_weights, qnd_parameters, _ = qnd_beta_weights_estimation(
        observations=observations,
        working_matrix=working_matrix,
        n_bootstrap=1000
    )
    
    if verbosity > 1:
        print("\n# QND parameters:\n")
        [print(p) for p in qnd_parameters]
        print("\n# QND weights:\n")
        print(qnd_weights)
    
    n_conditions = len(observations)

    N        = np.array([o.sample_size() for o in observations])
    accuracy = np.array([o.accuracy()    for o in observations])
    mean_rt  = np.array([o.mean_rt()     for o in observations])
    var_rt   = np.array([o.var_rt()      for o in observations])
    
    # Validate design matrices
    bound_mtx = working_matrix.boundary_design()
    drift_mtx = working_matrix.drift_design()
    nondt_mtx = working_matrix.ndt_design()
    
    if bound_mtx.shape[0] != n_conditions:
        raise ValueError(f"boundary_design must have {n_conditions} rows")
    if drift_mtx.shape[0] != n_conditions:
        raise ValueError(f"drift_design must have {n_conditions} rows")
    if nondt_mtx.shape[0] != n_conditions:
        raise ValueError(f"ndt_design must have {n_conditions} rows")
    
    # Convert to JAGS data format - direct translation from PyMC  
    jags_data = {
        'n_conditions'       : n_conditions,
        'n_boundary_weights' : bound_mtx.shape[1],
        'n_drift_weights'    : drift_mtx.shape[1], 
        'n_ndt_weights'      : nondt_mtx.shape[1],
        'N'                  : N.tolist(),
        'acc_obs'            : accuracy.tolist(),
        'rt_obs'             : mean_rt.tolist(),
        'var_obs'            : var_rt.tolist(),
        'boundary_design'    : bound_mtx.tolist(),
        'drift_design'       : drift_mtx.tolist(),
        'ndt_design'         : nondt_mtx.tolist()
    }
    
    # JAGS model string - exactly like PyMC with unconstrained normal priors
    model_string = """
    model {
        # Priors for weights - unconstrained normal priors (exactly like PyMC)
        # sigma=2.0 -> precision=1/4=0.25
        for (i in 1:n_boundary_weights) {
            boundary_weights[i] ~ dnorm(0, 0.25)
        }
        for (i in 1:n_drift_weights) {
            drift_weights[i] ~ dnorm(0, 0.25)
        }
        for (i in 1:n_ndt_weights) {
            ndt_weights[i] ~ dnorm(0, 0.25)
        }
        
        # Linear design matrix regression
        boundary <- boundary_design %*% boundary_weights
        drift    <- drift_design    %*% drift_weights
        ndt      <- ndt_design      %*% ndt_weights
        
        for (j in 1:n_conditions) {                      
            # Helper calculation for accuracy
            y[j] <- exp(-boundary[j] * drift[j])
            pred_acc[j] <- 1 / (y[j] + 1)

            # Likelihood for accuracy (binomial)
            acc_obs[j] ~ dbin(pred_acc[j], N[j])
            
            pred_mean[j] <- ndt[j] + (boundary[j] / (2 * drift[j])) * ((1 - y[j]) / (1 + y[j]))
            
            pred_var[j] <- (boundary[j] / (2 * pow(drift[j], 3))) * 
                           ((1 - 2*boundary[j]*drift[j]*y[j] - pow(y[j], 2)) / 
                            pow((y[j] + 1), 2))
            
            pred_var_safe[j] <- max(pred_var[j], 1e-6)
            pred_sigma[j] <- sqrt(pred_var_safe[j] / N[j])
            pred_precision[j] <- 1 / pow(pred_sigma[j], 2)
            
            rt_obs[j] ~ dnorm(pred_mean[j], pred_precision[j])

            var_obs[j] ~ dgamma((N[j]-1)/2, (N[j]-1)/(2*pred_var_safe[j]))
        }
    }
    """
    
    # Set up initial values using QND estimates - convert to JAGS format
    def format_jags_vector(values):
        """Convert Python list to JAGS c() format."""
        return f"c({', '.join(map(str, values))})"
    
    init_values = []
    for _ in range(n_chains):
        bound_init_mean  = qnd_weights.beta_boundary_mean()
        drift_init_mean  = qnd_weights.beta_drift_mean()
        nondt_init_mean  = qnd_weights.beta_ndt_mean()
        bound_init_sd    = qnd_weights.beta_boundary_sd()
        drift_init_sd    = qnd_weights.beta_drift_sd()
        nondt_init_sd    = qnd_weights.beta_ndt_sd()
        bound_init_lower = qnd_weights.beta_boundary_lower()
        drift_init_lower = qnd_weights.beta_drift_lower()
        nondt_init_lower = qnd_weights.beta_ndt_lower()
        bound_init_upper = qnd_weights.beta_boundary_upper()
        drift_init_upper = qnd_weights.beta_drift_upper()
        nondt_init_upper = qnd_weights.beta_ndt_upper()
        
        # Sample initial values from a truncated normal distribution
        def sample_truncated_normal(mean, sd, lower, upper, size):
            """Sample from truncated normal using scipy."""
            a = (lower - mean) / sd  # Normalized lower bound
            b = (upper - mean) / sd  # Normalized upper bound
            return truncnorm.rvs(a, b, loc=mean, scale=sd, size=size)
        
        bound_init = sample_truncated_normal(
            mean  = bound_init_mean,
            sd    = bound_init_sd,
            lower = bound_init_lower,
            upper = bound_init_upper,
            size  = bound_init_mean.shape
        )
        drift_init = sample_truncated_normal(
            mean  = drift_init_mean,
            sd    = drift_init_sd,
            lower = drift_init_lower,
            upper = drift_init_upper,
            size  = drift_init_mean.shape
        )
        nondt_init = sample_truncated_normal(
            mean  = nondt_init_mean,
            sd    = nondt_init_sd,
            lower = nondt_init_lower,
            upper = nondt_init_upper,
            size  = nondt_init_mean.shape
        )
        
        init_dict = {
            'boundary_weights' : format_jags_vector(bound_init),
            'drift_weights'    : format_jags_vector(drift_init),
            'ndt_weights'      : format_jags_vector(nondt_init)
        }
        
        init_values.append(init_dict)
    
    if verbosity > 1:
        print("\n# Running JAGS...\n")
    
    # Run JAGS
    mcmc_samples = p2.run_jags(
        model_string  = model_string,
        data_dict     = jags_data,
        nchains       = n_chains,
        init          = init_values,
        nadapt        = n_tune,
        nburnin       = n_tune,
        nsamples      = n_samples,
        monitorparams = ['boundary_weights', 'drift_weights', 'ndt_weights'],
        verbosity     = verbosity
    )
    
    # Check convergence
    rhat_values = [mcmc_samples.stats[param]['rhat'] for param in mcmc_samples.stats.keys()]
    max_rhat = max(rhat_values)
    
    if max_rhat > _MAX_R_HAT:
        print(f"Warning: Maximum R-hat ({max_rhat:.3f}) exceeds threshold ({_MAX_R_HAT})")
        print("Consider increasing n_tune or n_samples for better convergence.")
    
    # Convert MCMC samples to beta weights and parameters
    beta_weights, parameters = mcmc_samples_to_beta_weights(
        mcmc_samples   = mcmc_samples, 
        working_matrix = working_matrix, 
        verbosity      = verbosity
    )
    
    if verbosity > 1:
        print(f"\n# Convergence Summary:")
        print(f"Maximum R-hat: {max_rhat:.4f}")
        print(f"Converged: {'✅' if max_rhat <= _MAX_R_HAT else '❌'}")
        
        print(f"\n# Estimated Beta Weights:")
        print(beta_weights)
        
        print(f"\n# Estimated Parameters:")
        for i, p in enumerate(parameters):
            print(f"Condition {i+1}: {p}")
    
    return beta_weights, parameters, mcmc_samples

def demo():
    """Demonstrate design matrix parameter estimation."""
    N = 200
    
    design_matrix = _EXAMPLE_DESIGN_MATRIX
    
    true_bounds = design_matrix.boundary()
    true_drifts = design_matrix.drift()
    true_ndts = design_matrix.ndt()

    true_params_list = [
        Parameters(boundary=bound, drift=drift, ndt=ndt)
        for bound, drift, ndt in zip(true_bounds, true_drifts, true_ndts)
    ]

    pred_stats_list = ez.forward(true_params_list)

    obs_stats_list = [
        s.sample(sample_size=N)
        for s in pred_stats_list
    ]
    
    # Create a working design matrix
    working_matrix = DesignMatrix(
        boundary_design = design_matrix.boundary_design(),
        drift_design    = design_matrix.drift_design(),
        ndt_design      = design_matrix.ndt_design()
    )

    est_beta, est_params, mcmc_samples = bayesian_design_matrix_parameter_estimation(
        observations   = obs_stats_list,
        working_matrix = working_matrix,
        n_samples      = 5000,
        n_tune         = 10000,
        n_chains       = 4,
        verbosity      = 2
    )

    print("\n# True parameters:\n")
    [print(t) for t in true_params_list]

    print("\n# Estimated parameters:\n")
    [print(e) for e in est_params]

    print("\n# Weights:\n")
    print(est_beta)

    print("\n# True weights:\n")
    print(f"Boundary weights : {design_matrix.boundary_weights()}")
    print(f"Drift weights    : {design_matrix.drift_weights()}")
    print(f"NDT weights      : {design_matrix.ndt_weights()}")

    def pretty(covered):
        return "✅" if covered else "❌"

    print(f"\n--------------------------------------------\n")

    print(f"Coverage of boundary weights :", end=" ")
    for i in range(est_beta.beta_boundary_lower().shape[0]):
        covered = (
            est_beta.beta_boundary_lower()[i] <= 
            design_matrix.boundary_weights()[i] <= 
            est_beta.beta_boundary_upper()[i]
        )
        print(f"{pretty(covered)}", end=" ")
    
    print(f"\nCoverage of drift weights    :", end=" ")
    for i in range(est_beta.beta_drift_lower().shape[0]):
        covered = (
            est_beta.beta_drift_lower()[i] <= 
            design_matrix.drift_weights()[i] <= 
            est_beta.beta_drift_upper()[i]
        )
        print(f"{pretty(covered)}", end=" ")
    
    print(f"\nCoverage of ndt weights      :", end=" ")
    for i in range(est_beta.beta_ndt_lower().shape[0]):
        covered = (
            est_beta.beta_ndt_lower()[i] <= 
            design_matrix.ndt_weights()[i] <= 
            est_beta.beta_ndt_upper()[i]
        )
        print(f"{pretty(covered)}", end=" ")
    print()
    
    # Print convergence summary
    mcmc_samples.show_diagnostics()
    
    rhat_values = [mcmc_samples.stats[param]['rhat'] for param in mcmc_samples.stats.keys()]
    max_rhat = max(rhat_values)
    
    print(f"\n# Convergence Summary:")
    print(f"Maximum R-hat: {max_rhat:.4f}")
    print(f"Converged: {'✅' if max_rhat <= _MAX_R_HAT else '❌'}")
    
    # Make traceplots of the weights
    fig = mcmc_samples.trace_plot(
        parameters = mcmc_samples.filter_parameters('weights'),
        ncols = 3
    )
    fig.savefig('traceplots.png')
    
    print(f"\n")

def simulation(repetitions: int = 100):
    """Simulate design matrix parameter estimation."""
    if repetitions is None:
        repetitions = 100
    
    if not isinstance(repetitions, int):
        raise TypeError("repetitions must be an integer")
    
    N = 200
    verbosity = 0

    design_matrix = _EXAMPLE_DESIGN_MATRIX
    
    true_bounds = design_matrix.boundary()
    true_drifts = design_matrix.drift()
    true_nondts = design_matrix.ndt()

    true_params_list = [
        Parameters(boundary=bound, drift=drift, ndt=ndt)
        for bound, drift, ndt in zip(true_bounds, true_drifts, true_nondts)
    ]

    pred_stats_list = ez.forward(true_params_list)
    
    # Create a working design matrix
    working_matrix = DesignMatrix(
        boundary_design = design_matrix.boundary_design(),
        drift_design    = design_matrix.drift_design(),
        ndt_design      = design_matrix.ndt_design()
    )
    
    # Initialize a progress bar
    progress_bar = tqdm(total=repetitions, desc="Simulation progress")
    
    bound_weights_coverage = 0
    drift_weights_coverage = 0
    nondt_weights_coverage = 0

    n_bound_weights = design_matrix.boundary_design().shape[1]
    n_drift_weights = design_matrix.drift_design().shape[1]
    n_nondt_weights = design_matrix.ndt_design().shape[1]

    for i in range(repetitions):
        
        observations = [s.sample(sample_size=N) for s in pred_stats_list]
        est_beta, est_params, mcmc_samples = bayesian_design_matrix_parameter_estimation(
            observations   = observations,
            working_matrix = working_matrix,
            n_samples      = 2000,
            n_tune         = 1000,
            n_chains       = 4,
            verbosity      = verbosity
        )

        for j in range(n_bound_weights):
            covered = (
                est_beta.beta_boundary_lower()[j] <= 
                design_matrix.boundary_weights()[j] <= 
                est_beta.beta_boundary_upper()[j]
            )
            bound_weights_coverage += covered
            
        for j in range(n_drift_weights):
            covered = (
                est_beta.beta_drift_lower()[j] <= 
                design_matrix.drift_weights()[j] <= 
                est_beta.beta_drift_upper()[j]
            )
            drift_weights_coverage += covered
            
        for j in range(n_nondt_weights):
            covered = (
                est_beta.beta_ndt_lower()[j] <= 
                design_matrix.ndt_weights()[j] <= 
                est_beta.beta_ndt_upper()[j]
            )
            nondt_weights_coverage += covered
            
        progress_bar.update(1)
    progress_bar.close()

    bound_weights_coverage *= (100 / (repetitions * n_bound_weights))
    drift_weights_coverage *= (100 / (repetitions * n_drift_weights))
    nondt_weights_coverage *= (100 / (repetitions * n_nondt_weights))

    print("Coverage:")
    print(f" > Boundary : {bound_weights_coverage:5.1f}%  (should be ≈ 95%)")
    print(f" > Drift    : {drift_weights_coverage:5.1f}%  (should be ≈ 95%)")
    print(f" > NDT      : {nondt_weights_coverage:5.1f}%  (should be ≈ 95%)")

    # Print brief report to file, with simulation settings and results
    this_path = os.path.dirname(os.path.abspath(__file__))
    report_file = os.path.join(this_path, "jags_beta_weights_simulation_report.txt")
    with open(report_file, "w") as f:
        f.write(f"Simulation settings:\n")
        f.write(f" > Repetitions :  {repetitions}\n")
        f.write(f" > Sample size :  {N}\n")
        f.write(f" > MCMC samples:  5000\n")
        f.write(f" > MCMC burnin :  10000\n")
        f.write(f" > MCMC chains :  4\n")
        f.write(f"\n")
        f.write(f"Coverage:\n")
        f.write(f" > Boundary : {bound_weights_coverage:5.1f}%  (should be ≈ 95%)\n")
        f.write(f" > Drift    : {drift_weights_coverage:5.1f}%  (should be ≈ 95%)\n")
        f.write(f" > NDT      : {nondt_weights_coverage:5.1f}%  (should be ≈ 95%)\n")

    print(f"\nSimulation report saved to: {report_file}")

class TestSuite(unittest.TestCase):
    
    def setUp(self):
        self.design_matrix = _EXAMPLE_DESIGN_MATRIX
        self.working_matrix = DesignMatrix(
            boundary_design=self.design_matrix.boundary_design(),
            drift_design=self.design_matrix.drift_design(),
            ndt_design=self.design_matrix.ndt_design()
        )
    
    def tearDown(self):
        pass
    
    def test_demo_execution_1(self):
        """Test that demo runs without error."""
        demo()
    
    def test_demo_execution_2(self):
        """Test that simulation runs without error."""
        simulation(repetitions=2)


"""
Main
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JAGS Beta Weights Recovery")
    parser.add_argument("--test", action="store_true", help="Run the test suite")
    parser.add_argument("--demo", action="store_true", help="Run the demo")
    parser.add_argument("--simulation", action="store_true", help="Run the simulation study")
    parser.add_argument("--repetitions", type=int, default=1000, help="Number of repetitions for the simulation study")
    args = parser.parse_args()
    
    if args.test:
        unittest.main(argv=['first-arg-is-ignored'], exit=False)
        sys.exit(0)
    
    if args.demo:
        demo()
    elif args.simulation:
        simulation(repetitions=args.repetitions)
    else:
        print("Use --demo for basic demonstration or --simulation for simulation study") 