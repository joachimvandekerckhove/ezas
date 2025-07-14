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
from vendor.ezas.base import ez_equations as ez
from vendor.ezas.classes.moments import Observations
from vendor.ezas.classes.parameters import Parameters
from vendor.ezas.utils.debug import announce

import vendor.py2jags.src as p2
from vendor.py2jags.src.mcmc_samples import MCMCSamples

_DEMO_DEFAULT_PARAMETERS = Parameters(1.0, 0.5, 0.2)
_DEMO_DEFAULT_SAMPLE_SIZE = 100

_MAX_R_HAT = 1.1

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def observations_to_jags_data(observations: Observations) -> dict:
    """Convert Observations object to JAGS data format."""
    return {
        'N': int(observations.sample_size()),
        'acc_obs': int(observations.accuracy()),
        'rt_obs': float(observations.mean_rt()),
        'var_obs': float(observations.var_rt())
    }

def mcmc_samples_to_parameters(mcmc_samples: MCMCSamples) -> Parameters:
    """Convert MCMC samples to Parameters object."""
    
    # Extract statistics from MCMCSamples object
    boundary_stats = mcmc_samples.stats['boundary']
    drift_stats = mcmc_samples.stats['drift']
    ndt_stats = mcmc_samples.stats['ndt']
    
    # Extract means
    boundary_mean = boundary_stats['mean']
    drift_mean = drift_stats['mean']
    ndt_mean = ndt_stats['mean']
    
    # Extract standard deviations
    boundary_sd = boundary_stats['std']
    drift_sd = drift_stats['std']
    ndt_sd = ndt_stats['std']
    
    # Extract quantiles
    boundary_q025 = boundary_stats['q025']
    boundary_q975 = boundary_stats['q975']
    drift_q025 = drift_stats['q025']
    drift_q975 = drift_stats['q975']
    ndt_q025 = ndt_stats['q025']
    ndt_q975 = ndt_stats['q975']
    
    return Parameters(
        boundary=float(boundary_mean),
        drift=float(drift_mean),
        ndt=float(ndt_mean),
        boundary_sd=float(boundary_sd),
        drift_sd=float(drift_sd),
        ndt_sd=float(ndt_sd),
        boundary_lower_bound=float(boundary_q025),
        boundary_upper_bound=float(boundary_q975),
        drift_lower_bound=float(drift_q025),
        drift_upper_bound=float(drift_q975),
        ndt_lower_bound=float(ndt_q025),
        ndt_upper_bound=float(ndt_q975)
    )

def bayesian_parameter_estimation(observations: Observations, 
                                n_samples: int = 400, 
                                n_tune: int = 100, 
                                verbosity: int = 2) -> Parameters:
    """Estimate EZ-diffusion parameters using JAGS given observed statistics.
    
    Args:
        obs_stats: Observed summary statistics
        N: Number of trials used to generate the statistics
        n_samples: Number of MCMC samples to draw
        n_tune: Number of tuning steps for MCMC
        
    Returns:
        Estimated parameters
    """
    if not isinstance(observations, Observations):
        raise TypeError("observations must be an instance of Observations")
    
    # Create JAGS model string
    jags_model = """
    model {
        # Priors for parameters
        boundary ~ dunif(0.1, 5.0)
        drift ~ dnorm(0, 0.25)T(0.01, )  # precision = 1/sigma^2 = 1/4 = 0.25
        ndt ~ dunif(0.0, 1.0)
        
        # Helper calculation for accuracy
        y <- exp(-boundary * drift)
        pred_acc <- 1 / (y + 1)
        
        # Likelihood for accuracy (binomial)
        acc_obs ~ dbin(pred_acc, N)
        
        # Likelihood for mean RT (normal)
        pred_mean <- ndt + (boundary / (2 * drift)) * ((1 - y) / (1 + y))
        pred_var <- (boundary / (2 * pow(drift, 3))) * ((1 - 2*boundary*drift*y - pow(y, 2)) / pow((y + 1), 2))
        
        rt_obs ~ dnorm(pred_mean, N / pred_var)  # precision = N / variance
        
        # Likelihood for variance (gamma approximation)
        var_obs ~ dgamma((N-1)/2, (N-1)/(2*pred_var))
    }
    """
    
    # Data for JAGS
    jags_data = observations_to_jags_data(observations)
    
    # Initial values for multiple chains
    jags_inits = [
        {
            'boundary': 1.5,
            'drift': 0.75,
            'ndt': 0.3
        },
        {
            'boundary': 2.0,
            'drift': 1.0,
            'ndt': 0.2
        }
    ]
    
    # Initialize JAGS model
    mcmc_samples = p2.run_jags(
        model_string=jags_model,
        data_dict=jags_data,
        nchains=4,
        init=jags_inits,
        nadapt=n_tune,
        nburnin=n_tune,
        nsamples=n_samples,
        monitorparams=['boundary', 'drift', 'ndt'],
        verbosity=verbosity
    )
    
    # Print summary if verbose
    if verbosity >= 2:
        mcmc_samples.summary()
    
    # Check convergence
    if verbosity >= 1:
        mcmc_samples.show_diagnostics()
    
    # Convert to Parameters object using the helper function
    return mcmc_samples_to_parameters(mcmc_samples)




def demo(true_parameters: Parameters|None = None,
         sample_size: int|None = None):
    """
    Demo function for the Bayesian parameter estimation.
    
    Args:
        parameters: The true parameters to use for the demo.
        sample_size: The sample size to use for the demo.
    
    Returns:
        None
    """
    
    if true_parameters is None:
        true_parameters = _DEMO_DEFAULT_PARAMETERS
    
    if sample_size is None:
        sample_size = 100
    
    if not isinstance(true_parameters, Parameters):
        raise TypeError("true_parameters must be an instance of Parameters")
    
    if not isinstance(sample_size, int):
        raise TypeError("sample_size must be an instance of int")

    if sample_size <= 0:
        raise ValueError("sample_size must be greater than 0")
    
    print("True parameters:")
    print(true_parameters)
    
    # Compute the moments
    moments = ez.forward(true_parameters)
    print("True moments   :")
    print(moments)
    
    # Generate some observations
    observations = moments.sample(sample_size=sample_size)
    print("Observations   :")
    print(observations)
    
    # Estimate the parameters
    estimated_parameters = bayesian_parameter_estimation(observations, verbosity=0)
    print("Estimated parameters:")
    print(estimated_parameters)
    
    # Check if the true parameters are in the bounds of the estimated parameters and print colorful emoji checkmark or X
    boundary_in_bounds, drift_in_bounds, ndt_in_bounds = \
        true_parameters.is_within_bounds_of(estimated_parameters)
    print(f"Boundary in bounds : {'✅' if boundary_in_bounds else '❌'}")
    print(f"Drift in bounds    : {'✅' if drift_in_bounds else '❌'}")
    print(f"NDT in bounds      : {'✅' if ndt_in_bounds else '❌'}")

"""
Simulation
"""
def simulation(repetitions: int|None = None):
    """
    Run a comprehensive demonstration of Bayesian parameter recovery accuracy.
    """
    
    if repetitions is None:
        repetitions = 100

    if not isinstance(repetitions, int):
        raise TypeError("simulation_repetitions must be of type int")
    
    true_parameters = _DEMO_DEFAULT_PARAMETERS
    sample_size = _DEMO_DEFAULT_SAMPLE_SIZE
            
    # Initialize a progress bar
    progress_bar = tqdm(total=repetitions, 
                        desc="Simulation progress")

    boundary_coverage = 0
    drift_coverage = 0
    ndt_coverage = 0
    total_coverage = 0
        
    for _ in range(repetitions):
        moments = ez.forward(true_parameters)
        observations = moments.sample(sample_size=sample_size)
        estimate = bayesian_parameter_estimation(
            observations,
            n_samples=1000,
            n_tune=1000,
            verbosity=0
        )
        
        (boundary_covered, drift_covered, ndt_covered) = \
            true_parameters.is_within_bounds_of(estimate)
        
        if boundary_covered: 
            boundary_coverage += 1
        if drift_covered: 
            drift_coverage += 1
        if ndt_covered: 
            ndt_coverage += 1    
        if boundary_covered and drift_covered and ndt_covered: 
            total_coverage += 1
            
        # Update the progress bar
        progress_bar.update(1)
        
    # Close the progress bar
    progress_bar.close()
    
    boundary_coverage *= 100 / repetitions
    drift_coverage *= 100 / repetitions
    ndt_coverage *= 100 / repetitions
    total_coverage *= 100 / repetitions
    
    print("Coverage:")
    print(f" > Boundary: {boundary_coverage:.1f}%  (should be ≈ 95%)")
    print(f" > Drift   : {drift_coverage:.1f}%  (should be ≈ 95%)")
    print(f" > NDT     : {ndt_coverage:.1f}%  (should be ≈ 95%)")
    print(f" > Total   : {total_coverage:.1f}%  (should be ≈ 86%)") # .95^3



"""
Test suite
"""
class TestSuite(unittest.TestCase):
    def setUp(self):
        import io
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
    def tearDown(self):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

    def test_demo_execution_1(self):
        demo()
        self.assertIn("True parameters:", sys.stdout.getvalue())
        self.assertIn("Estimated parameters:", sys.stdout.getvalue())
        
    def test_demo_execution_2(self):
        demo(true_parameters=Parameters(
            boundary=2.0,
            drift=0.1,
            ndt=0.2
        ), sample_size=50)  # Use smaller sample size for faster testing
        self.assertIn("True parameters:", sys.stdout.getvalue())
        self.assertIn("Estimated parameters:", sys.stdout.getvalue())


# if __name__ == "__main__":
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--test", action="store_true", help="Run the test suite")
#     parser.add_argument("--demo", action="store_true", help="Run the demo")
#     parser.add_argument("--simulation", action="store_true", help="Run the simulation")
#     parser.add_argument("--iterations", type=int, default=100, help="Number of iterations for the simulation")
#     args = parser.parse_args()
    
#     if args.test:
#         unittest.main(argv=[__file__], verbosity=0, failfast=True)

#     if args.demo:
#         demo()
        
#     if args.simulation:
#         simulation(args.iterations)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian Parameter Estimation")
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