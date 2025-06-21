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
from vendor.ezas.utils import PosteriorSummary
from vendor.ezas.classes.design_matrix import DesignMatrix
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
    
    N = observations.sample_size
    accuracy = observations.accuracy
    mean_rt = observations.mean_rt
    var_rt = observations.var_rt
    
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
        
        if verbosity >= 1:
            # Print the summary table
            print(summary)
            if verbosity >= 2:
                # Check convergence
                rhats = summary['r_hat']
                max_rhat = np.max(rhats)
                if max_rhat > MAX_R_HAT:
                    print(f"Warning: Maximum Rhat is {max_rhat}, which is greater than {MAX_R_HAT}")
                else:
                    print(f"Maximum Rhat is {max_rhat}, which is less than {MAX_R_HAT}")
        
        # Get posterior means
        post_mean = summary['mean']
        
        return Parameters(
            boundary=float(post_mean['boundary']),
            drift=float(post_mean['drift']),
            ndt=float(post_mean['ndt'])
        )

def bayesian_multiple_parameter_estimation(
    observations: List[Observations], 
    n_samples: int = 2000, 
    n_tune: int = 1000,
    n_chains: int = 4 
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
    
    N = [o.sample_size for o in observations]
    accuracy = [o.accuracy for o in observations]
    mean_rt = [o.mean_rt for o in observations]
    var_rt = [o.var_rt for o in observations]
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
        trace = pm.sample(n_samples, tune=n_tune, return_inferencedata=True, chains=n_chains)
        
        print(az.summary(trace))
        
        # Get posterior means
        post_mean = az.summary(trace)['mean']
        
        # Return mean parameters across all cells
        mean_parameters = Parameters(
            boundary=float(np.mean([post_mean[f'boundary[{i}]'] for i in range(n_conditions)])),
            drift=float(np.mean([post_mean[f'drift[{i}]'] for i in range(n_conditions)])),
            ndt=float(np.mean([post_mean[f'ndt[{i}]'] for i in range(n_conditions)]))
        )
        sd_parameters = Parameters(
            boundary=float(np.std([post_mean[f'boundary[{i}]'] for i in range(n_conditions)])),
            drift=float(np.std([post_mean[f'drift[{i}]'] for i in range(n_conditions)])),
            ndt=float(np.std([post_mean[f'ndt[{i}]'] for i in range(n_conditions)]))
        )

        return mean_parameters, sd_parameters

"""
Demo
"""
def demo():
    print("Demo for bayes:")
    print("No demo available")

"""
Test suite
"""
class TestSuite(unittest.TestCase):
    def test_bayesian_parameter_estimation(self):
        pass
    
    def test_bayesian_multiple_parameter_estimation(self):
        pass




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run the test suite")
    parser.add_argument("--demo", action="store_true", help="Run the demo")
    args = parser.parse_args()
    
    if args.test:
        unittest.main(argv=[__file__], verbosity=0, failfast=True)

    if args.demo:
        demo()