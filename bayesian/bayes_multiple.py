#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from typing import Tuple, List
import unittest
import argparse
import pymc as pm
import arviz as az
from vendor.ezas.base import ez_equations as ez
from vendor.ezas.classes.moments import Observations
from vendor.ezas.classes.parameters import Parameters

MAX_R_HAT = 1.1


def bayesian_multiple_parameter_estimation(
    observations: List[Observations], 
    n_samples: int = 2000, 
    n_tune: int = 1000,
    n_chains: int = 4,
    verbosity: int = 2
) -> Tuple[Parameters, Parameters]:
    """Estimate EZ-diffusion parameters using PyMC given observed statistics.
    
    Args:
        observations: Observed summary statistics (array of Observations)
        n_samples: Number of MCMC samples to draw
        n_tune: Number of tuning steps for MCMC
        
    Returns:
        Posterior means of the parameters
        Posterior standard deviations of the parameters
    """
    if not isinstance(observations, list):
        raise TypeError("observations must be a list")
    if not all(isinstance(x, Observations) for x in observations):
        raise TypeError("All elements in observations must be Observations instances")
    
    N = np.array([o.sample_size() for o in observations])
    accuracy = np.array([o.accuracy() for o in observations])
    mean_rt = np.array([o.mean_rt() for o in observations])
    var_rt = np.array([o.var_rt() for o in observations])
    n_conditions = len(observations)
    
    # Define the Bayesian model
    with pm.Model() as model:
        # Priors for parameters
        boundary = pm.Uniform('boundary', lower=0.1, upper=5.0, shape=n_conditions)
        drift = pm.Normal('drift', mu=0, sigma=2.0, shape=n_conditions)
        ndt = pm.Uniform('ndt', lower=0.0, upper=1.0, shape=n_conditions)
        
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
        trace = pm.sample(n_samples, tune=n_tune, return_inferencedata=True, chains=n_chains,
                         compute_convergence_checks=False)
        
        summary = az.summary(trace)
        
        rhats = summary['r_hat']
        
        if verbosity >= 2: 
            # Print the summary table
            print(summary)
        
        if verbosity >= 1:
            # Check convergence
            max_rhat = np.max(rhats)
            if max_rhat > MAX_R_HAT:
                print(f"Warning: Maximum Rhat is {max_rhat}, which is greater than {MAX_R_HAT}.")
            else:
                print(f"Maximum Rhat is {max_rhat}, which is less than {MAX_R_HAT}.")
        
        # Get posterior means
        post_mean = summary['mean']
        post_sd = summary['sd']

        lower_quantile = 0.025
        upper_quantile = 0.975
        
        posterior = trace.posterior

        # Get quantiles for the whole array at once
        boundary_q025 = posterior['boundary'].quantile(lower_quantile, dim=("chain", "draw")).values
        boundary_q975 = posterior['boundary'].quantile(upper_quantile, dim=("chain", "draw")).values

        drift_q025 = posterior['drift'].quantile(lower_quantile, dim=("chain", "draw")).values
        drift_q975 = posterior['drift'].quantile(upper_quantile, dim=("chain", "draw")).values

        ndt_q025 = posterior['ndt'].quantile(lower_quantile, dim=("chain", "draw")).values
        ndt_q975 = posterior['ndt'].quantile(upper_quantile, dim=("chain", "draw")).values
        
        return [Parameters(
            boundary=float(post_mean[f'boundary[{i}]']),
            drift=float(post_mean[f'drift[{i}]']),
            ndt=float(post_mean[f'ndt[{i}]']),
            boundary_sd=float(post_sd[f'boundary[{i}]']),
            drift_sd=float(post_sd[f'drift[{i}]']),
            ndt_sd=float(post_sd[f'ndt[{i}]']),
            boundary_lower_bound=float(boundary_q025[i]),
            boundary_upper_bound=float(boundary_q975[i]),
            drift_lower_bound=float(drift_q025[i]),
            drift_upper_bound=float(drift_q975[i]),
            ndt_lower_bound=float(ndt_q025[i]),
            ndt_upper_bound=float(ndt_q975[i])
        ) for i in range(n_conditions)]

def demo():

    # Set some parameters
    parameters = [
        Parameters(
            boundary=2.0,
            drift=d,
            ndt=0.2
        ) for d in [0.5, 1.5, 2.5]
    ]
    
    sample_size = 1000
    
    print("True parameters:")
    [print(p) for p in parameters]
    
    # Compute the moments
    moments = [p.to_moments() for p in parameters]
    print("True moments:")
    [print(m) for m in moments]
    
    # Sample the moments
    observations = [m.sample(sample_size=sample_size) for m in moments]
    print("Observations:")
    [print(o) for o in observations]
    
    # Estimate the parameters
    estimated_parameters = bayesian_multiple_parameter_estimation(observations)
    
    print("Estimated parameters:")
    [print(p) for p in estimated_parameters]
    
    # Check if the true parameters are in the bounds of the estimated parameters and print colorful emoji checkmark or X
    for i, p in enumerate(parameters):
        boundary_in_bounds, drift_in_bounds, ndt_in_bounds = p.is_in_bounds(estimated_parameters[i])
        print(f"Boundary {i} in bounds : {'✅' if boundary_in_bounds else '❌'}")
        print(f"Drift {i} in bounds    : {'✅' if drift_in_bounds else '❌'}")
        print(f"NDT {i} in bounds      : {'✅' if ndt_in_bounds else '❌'}")

"""
Test suite
"""
class TestSuite(unittest.TestCase):    
    def test_bayesian_multiple_parameter_estimation(self):
        demo()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run the test suite")
    parser.add_argument("--demo", action="store_true", help="Run the demo")
    args = parser.parse_args()
    
    if args.test:
        unittest.main(argv=[__file__], verbosity=0, failfast=True)

    if args.demo or len(sys.argv) == 1:
        demo()