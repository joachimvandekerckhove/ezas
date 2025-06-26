#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import unittest
import argparse
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from vendor.ezas.base import ez_equations as ez
from vendor.ezas.classes.moments import Observations
from vendor.ezas.classes.parameters import Parameters
from vendor.ezas.utils.debug import announce

_DEMO_DEFAULT_PARAMETERS = Parameters(1.0, 0.5, 0.2)
_DEMO_DEFAULT_SAMPLE_SIZE = 100

_MAX_R_HAT = 1.1

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def bayesian_parameter_estimation(observations: Observations, 
                                n_samples: int = 400, 
                                n_tune: int = 100, 
                                verbosity: int = 2) -> Parameters:
    """Estimate EZ-diffusion parameters using PyMC given observed statistics.
    
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
    
    N = observations.sample_size()
    accuracy = observations.accuracy()
    mean_rt = observations.mean_rt()
    var_rt = observations.var_rt()
    
    # Define the Bayesian model
    with pm.Model() as model:
        # Priors for parameters
        boundary = pm.Uniform('boundary', lower=0.1, upper=5.0)
        drift = pm.Normal('drift', mu=0, sigma=2.0)
        ndt = pm.Uniform('ndt', lower=0.0, upper=1.0)
        
        # Helper calculation for accuracy
        y = pm.math.exp(-boundary * drift)
        pred_acc = 1 / (y + 1)
        
        # Likelihood for accuracy (binomial)
        acc_obs = pm.Binomial('acc_obs', 
                            n=N, 
                            p=pred_acc, 
                            observed=accuracy)
        
        # Likelihood for mean RT (normal)
        pred_mean = (ndt + 
                    (boundary / (2 * drift)) * 
                    ((1 - y) / (1 + y)))
        
        pred_var = ((boundary / (2 * drift**3)) * 
                    ((1 - 2*boundary*drift*y - y**2) / 
                     ((y + 1)**2)))
        
        rt_obs = pm.Normal('rt_obs',
                          mu=pred_mean,
                          sigma=pm.math.sqrt(pred_var/N),
                          observed=mean_rt)
        
        # Likelihood for variance (gamma approximation)
        var_obs = pm.Gamma('var_obs',
                          alpha=(N-1)/2,
                          beta=(N-1)/(2*pred_var),
                          observed=var_rt)
        
        # Run MCMC
        trace = pm.sample(n_samples, tune=n_tune, return_inferencedata=True, 
                         compute_convergence_checks=False, progressbar=False)
                
        summary = az.summary(trace)
        
        rhats = summary['r_hat']
        
        if verbosity >= 2: 
            # Print the summary table
            print(summary)
        
        if verbosity >= 1:
            # Check convergence
            max_rhat = np.max(rhats)
            if max_rhat > _MAX_R_HAT:
                print(f"Warning: Maximum Rhat is {max_rhat}, which is greater than {_MAX_R_HAT}.")
            else:
                print(f"Maximum Rhat is {max_rhat}, which is less than {_MAX_R_HAT}.")

        # Get posterior means and standard deviations
        post_mean = summary['mean']
        post_sd = summary['sd']
        
        lower_quantile = 0.025
        upper_quantile = 0.975
        
        posterior = trace.posterior

        boundary_q025 = posterior['boundary'].quantile(lower_quantile).values
        boundary_q975 = posterior['boundary'].quantile(upper_quantile).values

        drift_q025 = posterior['drift'].quantile(lower_quantile).values
        drift_q975 = posterior['drift'].quantile(upper_quantile).values

        ndt_q025 = posterior['ndt'].quantile(lower_quantile).values
        ndt_q975 = posterior['ndt'].quantile(upper_quantile).values
        
        # Return Parameters object with means and standard deviations
        return Parameters(
            boundary=float(post_mean['boundary']),
            drift=float(post_mean['drift']),
            ndt=float(post_mean['ndt']),
            boundary_sd=float(post_sd['boundary']),
            drift_sd=float(post_sd['drift']),
            ndt_sd=float(post_sd['ndt']),
            boundary_lower_bound=float(boundary_q025),
            boundary_upper_bound=float(boundary_q975),
            drift_lower_bound=float(drift_q025),
            drift_upper_bound=float(drift_q975),
            ndt_lower_bound=float(ndt_q025),
            ndt_upper_bound=float(ndt_q975)
        )

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
    
    print("True parameters:", end=" ")
    print(true_parameters)
    
    # Compute the moments
    moments = ez.forward(true_parameters)
    print("True moments   :", end=" ")
    print(moments)
    
    # Generate some observations
    observations = moments.sample(sample_size=sample_size)
    print("Observations   :", end=" ")
    print(observations)
    
    # Estimate the parameters
    estimated_parameters = bayesian_parameter_estimation(observations)
    print("Estimated parameters:", end=" ")
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
def simulation(simulation_repetitions: int|None = None):
    """
    Run a comprehensive demonstration of Bayesian parameter recovery accuracy.
    """
    
    if simulation_repetitions is None:
        simulation_repetitions = 100

    if not isinstance(simulation_repetitions, int):
        raise TypeError("simulation_repetitions must be of type int")
    
    true_parameters = _DEMO_DEFAULT_PARAMETERS
    sample_size = _DEMO_DEFAULT_SAMPLE_SIZE
            
    # Initialize a progress bar
    progress_bar = tqdm(total=simulation_repetitions, 
                        desc="Simulation progress")

    boundary_coverage = 0
    drift_coverage = 0
    ndt_coverage = 0
    total_coverage = 0
        
    for _ in range(simulation_repetitions):
        moments = ez.forward(true_parameters)
        observations = moments.sample(sample_size=sample_size)
        estimate = bayesian_parameter_estimation(observations)
        
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
    
    boundary_coverage *= 100 / simulation_repetitions
    drift_coverage *= 100 / simulation_repetitions
    ndt_coverage *= 100 / simulation_repetitions
    total_coverage *= 100 / simulation_repetitions
    
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
        import logging
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        # Suppress PyMC logging during tests
        self.old_pymc_logger = logging.getLogger('pymc')
        self.old_pymc_level = self.old_pymc_logger.level
        self.old_pymc_logger.setLevel(logging.ERROR)
        
        # Also suppress arviz logging
        self.old_arviz_logger = logging.getLogger('arviz')
        self.old_arviz_level = self.old_arviz_logger.level
        self.old_arviz_logger.setLevel(logging.ERROR)
        
    def tearDown(self):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        
        # Restore PyMC logging
        self.old_pymc_logger.setLevel(self.old_pymc_level)
        self.old_arviz_logger.setLevel(self.old_arviz_level)

    def test_demo_execution_1(self):
        demo()
        self.assertIn("Sample size: 1000", sys.stdout.getvalue())
        self.assertIn("True parameters:", sys.stdout.getvalue())
        
    def test_demo_execution_2(self):
        demo(parameters=Parameters(
            boundary=2.0,
            drift=0.1,
            ndt=0.2
        ), sample_size=2000)
        self.assertIn("Sample size: 2000", sys.stdout.getvalue())


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run the test suite")
    parser.add_argument("--demo", action="store_true", help="Run the demo")
    parser.add_argument("--simulation", action="store_true", help="Run the simulation")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations for the simulation")
    args = parser.parse_args()
    
    if args.test:
        unittest.main(argv=[__file__], verbosity=0, failfast=True)

    if args.demo:
        demo()
        
    if args.simulation:
        simulation(args.iterations)
    