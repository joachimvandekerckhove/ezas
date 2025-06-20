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
from vendor.ezas.utils import announce

MAX_R_HAT = 1.1

def bayesian_parameter_estimation(observations: Observations, 
                                n_samples: int = 2000, 
                                n_tune: int = 1000, 
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
        trace = pm.sample(n_samples, tune=n_tune, return_inferencedata=True)
                
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

def demo():
    # Set some parameters
    parameters = Parameters(
        boundary=2.0,
        drift=0.1,
        ndt=0.2
    )
    sample_size = 1000
    
    print("True parameters:", end=" ")
    print(parameters)
    
    # Compute the moments
    moments = parameters.to_moments()
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
    boundary_in_bounds, drift_in_bounds, ndt_in_bounds = parameters.is_in_bounds(estimated_parameters)
    print(f"Boundary in bounds : {'✅' if boundary_in_bounds else '❌'}")
    print(f"Drift in bounds    : {'✅' if drift_in_bounds else '❌'}")
    print(f"NDT in bounds      : {'✅' if ndt_in_bounds else '❌'}")

"""
Test suite
"""
class TestSuite(unittest.TestCase):
    def test_bayesian_parameter_estimation(self):
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