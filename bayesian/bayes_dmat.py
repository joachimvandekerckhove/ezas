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
from vendor.ezas.classes.design_matrix import DesignMatrix
from vendor.ezas.lists import DesignList, WeightList, bigger, smaller
from vendor.ezas.utils.debug import announce
from vendor.ezas.utils.posterior import PosteriorSummary

MAX_R_HAT = 1.1

def bayesian_design_matrix_parameter_estimation(
    observations: List[Observations], 
    designList: DesignList,
    n_samples: int = 2000, 
    n_tune: int = 1000,
    n_chains: int = 4,
    verbosity: int = 2
) -> Tuple[WeightList, List[Parameters], az.InferenceData]:
    """Estimate EZ-diffusion parameters using PyMC given observed statistics and design matrices.
    
    Args:
        observations: Observed summary statistics (array of Observations)
        design_list: Design list (DesignList)
        n_samples: Number of MCMC samples to draw
        n_tune: Number of tuning steps for MCMC
        
    Returns:
        Tuple containing:
        - Weight list (WeightList)
        - Estimated parameters (Parameters)
        - Trace (az.InferenceData)
    """
    if not isinstance(observations, list):
        raise TypeError("observations must be a list")
    if not all(isinstance(x, Observations) for x in observations):
        raise TypeError("All elements in observations must be Observations instances")
    
    # Announce yourself
    announce()
    
    # Use the inverse equations to get quick and dirty estimates of the parameters
    qnd_params_list = [ez.inverse(observations) for observations in observations]
    print("\n# QND parameters:\n")
    [print(qnd_params) for qnd_params in qnd_params_list]
    
    # Parameter estimates
    qnd_boundaries = np.array([qnd_params.boundary for qnd_params in qnd_params_list])
    qnd_drifts = np.array([qnd_params.drift for qnd_params in qnd_params_list])
    qnd_ndts = np.array([qnd_params.ndt for qnd_params in qnd_params_list])
    
    # Solve the linear system of equations to get the weights for each parameter
    qnd_weights = smaller(designList, qnd_params_list)

    print("\n# QND weights:\n")
    print(qnd_weights)
    
    # Print the implied constrained parameters (transform into List[Parameters])
    print("\n# Implied constrained parameters:\n")
    implied_parameters = bigger(designList, qnd_weights)
 
    [print(p) for p in implied_parameters]
    
    # Use the quick and dirty weights as initial values for the Bayesian model
    init_values = {
        'boundary_weights': qnd_weights.boundary,
        'drift_weights': qnd_weights.drift,
        'ndt_weights': qnd_weights.ndt
    }
    
    n_conditions = len(observations)
    N = np.array([o.sample_size() for o in observations])
    accuracy = np.array([o.accuracy() for o in observations])
    mean_rt = np.array([o.mean_rt() for o in observations])
    var_rt = np.array([o.var_rt() for o in observations])
    
    # Validate design matrices
    if designList.boundary.shape[0] != n_conditions:
        raise ValueError(f"boundary_design must have {n_conditions} rows")
    if designList.drift.shape[0] != n_conditions:
        raise ValueError(f"drift_design must have {n_conditions} rows")
    if designList.ndt.shape[0] != n_conditions:
        raise ValueError(f"ndt_design must have {n_conditions} rows")
    
    print("\n# Running PyMC...\n")
    
    # Define the Bayesian model
    with pm.Model() as model:
        # Priors for weights
        boundary_weights = pm.Normal('boundary_weights', 
                                   mu=0, 
                                   sigma=2.0, 
                                   shape=designList.boundary.shape[1])
        drift_weights = pm.Normal('drift_weights', 
                                mu=0, 
                                sigma=2.0, 
                                shape=designList.drift.shape[1])
        ndt_weights = pm.Normal('ndt_weights', 
                              mu=0, 
                              sigma=2.0, 
                              shape=designList.ndt.shape[1])
        
        # Compute parameters from design matrices and weights
        boundary = pm.Deterministic('boundary', 
                                  pm.math.dot(designList.boundary, boundary_weights))
        drift = pm.Deterministic('drift', 
                               pm.math.dot(designList.drift, drift_weights))
        ndt = pm.Deterministic('ndt', 
                             pm.math.dot(designList.ndt, ndt_weights))
        
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
        
        # Run MCMC with more samples and longer tuning
        trace = pm.sample(n_samples, 
                         tune=n_tune, 
                         return_inferencedata=True, 
                         initvals=init_values,
                         chains=n_chains,
                         progressbar=verbosity > 0,
                         target_accept=0.95)  # Higher target acceptance rate
        
        # Print summary table for only those nodes with weight in their name
        summary = az.summary(trace)
        mask = summary.index.str.contains('weights')
        print(summary.loc[mask])
        
        # Get posterior means
        post_mean = summary['mean']
        
        # Get posterior standard deviations
        post_std = summary['sd']
        
        # Get central credible intervals (quantiles) for weights
        quantiles = trace.posterior.quantile([0.025, 0.975])
        
        # Extract weights
        weightList = WeightList(
            boundary=np.array([post_mean[f'boundary_weights[{i}]'] for i in range(designList.boundary.shape[1])]),
            drift=np.array([post_mean[f'drift_weights[{i}]'] for i in range(designList.drift.shape[1])]),
            ndt=np.array([post_mean[f'ndt_weights[{i}]'] for i in range(designList.ndt.shape[1])]),
            boundary_sd=np.array([post_std[f'boundary_weights[{i}]'] for i in range(designList.boundary.shape[1])]),
            drift_sd=np.array([post_std[f'drift_weights[{i}]'] for i in range(designList.drift.shape[1])]),
            ndt_sd=np.array([post_std[f'ndt_weights[{i}]'] for i in range(designList.ndt.shape[1])]),
            boundary_q025=quantiles['boundary_weights'].sel(quantile=0.025).values,
            drift_q025=quantiles['drift_weights'].sel(quantile=0.025).values,
            ndt_q025=quantiles['ndt_weights'].sel(quantile=0.025).values,
            boundary_q975=quantiles['boundary_weights'].sel(quantile=0.975).values,
            drift_q975=quantiles['drift_weights'].sel(quantile=0.975).values,
            ndt_q975=quantiles['ndt_weights'].sel(quantile=0.975).values)
            
        return weightList, bigger(designList, weightList), trace
    
def bayesian_design_matrix_parameter_estimation_with_hierarchical_structure(
    observations: List[Observations],
    boundary_design: np.ndarray,
    drift_design: np.ndarray,
    ndt_design: np.ndarray,
    person_id: List[int],
    n_samples: int = 2000, 
    n_tune: int = 1000,
    n_chains: int = 4 
) -> Tuple[Parameters, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate EZ-diffusion parameters using PyMC given observed statistics and design matrices.
    
    Args:
        observations: Observed summary statistics (array of Observations)
        boundary_design: Design matrix for boundary parameter (n_conditions x n_boundary_weights)
        drift_design: Design matrix for drift parameter (n_conditions x n_drift_weights)
        ndt_design: Design matrix for NDT parameter (n_conditions x n_ndt_weights)
        person_id: List of person IDs (array of ints)
        n_samples: Number of MCMC samples to draw
        n_tune: Number of tuning steps for MCMC
        
    Returns:
        Tuple containing:
        - Estimated parameters (Parameters)
        - Boundary weights (np.ndarray)
        - Drift weights (np.ndarray)
        - NDT weights (np.ndarray)
    """
    if not isinstance(observations, list):
        raise TypeError("observations must be a list")
    if not all(isinstance(x, Observations) for x in observations):
        raise TypeError("All elements in observations must be Observations instances")
    if len(observations) != len(person_id):
        raise ValueError("observations and person_id must have the same length")
    
    # Use the inverse equations to get quick and dirty estimates of the parameters
    qnd_params_list = [ez.inverse(observations) for observations in observations]
    
    print("\n# QND parameters:\n")
    [print(qnd_params) for qnd_params in qnd_params_list]
    
    # Make the design matrices into numpy matrices
    boundary_design_matrix = np.array(boundary_design)
    drift_design_matrix    = np.array(drift_design)
    ndt_design_matrix      = np.array(ndt_design)
    
    # Computed parameters
    qnd_boundaries = np.array([qnd_params.boundary for qnd_params in qnd_params_list])
    qnd_drifts = np.array([qnd_params.drift for qnd_params in qnd_params_list])
    qnd_ndts = np.array([qnd_params.ndt for qnd_params in qnd_params_list])
    
    # Solve the linear system of equations to get the weights for each parameter
    qnd_bound_weights = np.linalg.pinv(boundary_design_matrix) @ qnd_boundaries.T
    qnd_drift_weights = np.linalg.pinv(drift_design_matrix) @ qnd_drifts.T
    qnd_ndt_weights   = np.linalg.pinv(ndt_design_matrix) @ qnd_ndts.T

    print("\n# QND weights:\n")
    print(qnd_bound_weights)
    print(qnd_drift_weights)
    print(qnd_ndt_weights)
    
    # Print the implied constrained parameters (transform into List[Parameters])
    print("\n# Implied constrained parameters:\n")
    implied_boundaries = boundary_design_matrix @ qnd_bound_weights
    implied_drifts = drift_design_matrix @ qnd_drift_weights
    implied_ndts = ndt_design_matrix @ qnd_ndt_weights
    
    [
        print(
            Parameters(
                boundary=boundary, 
                drift=drift, 
                ndt=ndt
            ) for boundary, drift, ndt in zip(implied_boundaries, implied_drifts, implied_ndts)
        )
    ]
    # Use the quick and dirty weights as initial values for the Bayesian model
    init_values = {
        'boundary_weights': qnd_bound_weights,
        'drift_weights': qnd_drift_weights,
        'ndt_weights': qnd_ndt_weights
    }
    
    n_conditions = len(observations)
    
    # Validate design matrices
    if boundary_design.shape[0] != n_conditions:
        raise ValueError(f"boundary_design must have {n_conditions} rows")
    if drift_design.shape[0] != n_conditions:
        raise ValueError(f"drift_design must have {n_conditions} rows")
    if ndt_design.shape[0] != n_conditions:
        raise ValueError(f"ndt_design must have {n_conditions} rows")
    
    # Get the number of trials for each condition
    N = np.array([o.sample_size() for o in observations])
    accuracy = np.array([o.accuracy() for o in observations])
    mean_rt = np.array([o.mean_rt() for o in observations])
    var_rt = np.array([o.var_rt() for o in observations])
    
    print("\n# Running PyMC...\n")
    
    # Define the Bayesian model
    with pm.Model() as model:
        # Priors for weights
        boundary_weights = pm.Normal('boundary_weights', 
                                   mu=0, 
                                   sigma=2.0, 
                                   shape=boundary_design.shape[1])
        drift_weights = pm.Normal('drift_weights', 
                                mu=0, 
                                sigma=2.0, 
                                shape=drift_design.shape[1])
        ndt_weights = pm.Normal('ndt_weights', 
                              mu=0, 
                              sigma=2.0, 
                              shape=ndt_design.shape[1])
        
        # Compute parameters from design matrices and weights
        boundary_mean = pm.Deterministic('boundary_mean', 
                                  pm.math.dot(boundary_design, boundary_weights))
        drift_mean = pm.Deterministic('drift_mean', 
                               pm.math.dot(drift_design, drift_weights))
        ndt_mean = pm.Deterministic('ndt_mean', 
                             pm.math.dot(ndt_design, ndt_weights))
        
        # Population standard deviations
        boundary_sigma = pm.HalfNormal('boundary_sigma', 
                                      sigma=1.0)
        drift_sigma = pm.HalfNormal('drift_sigma', 
                                    sigma=1.0)
        ndt_sigma = pm.HalfNormal('ndt_sigma', 
                                  sigma=1.0)
        
        # Person-specific parameters
        boundary_offset = pm.Normal('boundary_offset', 
                                  mu=0, 
                                  sigma=1.0, 
                                  shape=len(person_id))
        drift_offset = pm.Normal('drift_offset', 
                                mu=0, 
                                sigma=1.0, 
                                shape=len(person_id))
        ndt_offset = pm.Normal('ndt_offset', 
                              mu=0, 
                              sigma=1.0, 
                              shape=len(person_id))
        
        boundary_p = pm.Deterministic('boundary_p', 
                                  boundary_mean + boundary_sigma * boundary_offset)
        drift_p = pm.Deterministic('drift_p', 
                                drift_mean + drift_sigma * drift_offset)
        ndt_p = pm.Deterministic('ndt_p', 
                              ndt_mean + ndt_sigma * ndt_offset)
        
        # Helper calculation for accuracy
        y = pm.math.exp(-boundary_p * drift_p)
        pred_acc = 1 / (y + 1)
        
        # Likelihood for mean RT (normal)
        pred_mean = (ndt_p + 
                    (boundary_p / (2 * drift_p)) * 
                    ((1 - y) / (1 + y)))
        
        pred_var = ((boundary_p / (2 * drift_p**3)) * 
                    ((1 - 2*boundary_p*drift_p*y - y**2) / 
                     ((y + 1)**2)))
        
        # Likelihood for accuracy (binomial)
        acc_obs = pm.Binomial('acc_obs', 
                            n=N, 
                            p=pred_acc, 
                            observed=accuracy)
        
        rt_obs = pm.Normal('rt_obs',
                          mu=pred_mean,
                          sigma=pm.math.sqrt(pred_var/N),
                          observed=mean_rt)
        
        # Likelihood for variance (gamma approximation)
        var_obs = pm.Gamma('var_obs',
                          alpha=(N-1)/2,
                          beta=(N-1)/(2*pred_var),
                          observed=var_rt)
        
        # Run MCMC with more samples and longer tuning
        trace = pm.sample(n_samples, 
                         tune=n_tune, 
                         return_inferencedata=True, 
                         initvals=init_values,
                         target_accept=0.95)  # Higher target acceptance rate
        
        # Print summary table for only those nodes with weight in their name
        mask = az.summary(trace).index.str.contains('weights')
        print(az.summary(trace).loc[mask])
        
        # Get posterior means
        post_mean = az.summary(trace)['mean']
        
        # Extract weights
        boundary_weights_est = np.array([post_mean[f'boundary_weights[{i}]'] 
                                       for i in range(boundary_design.shape[1])])
        drift_weights_est = np.array([post_mean[f'drift_weights[{i}]'] 
                                    for i in range(drift_design.shape[1])])
        ndt_weights_est = np.array([post_mean[f'ndt_weights[{i}]'] 
                                  for i in range(ndt_design.shape[1])])
        
        # Compute final parameter estimates
        final_boundary = np.dot(boundary_design, boundary_weights_est)
        final_drift = np.dot(drift_design, drift_weights_est)
        final_ndt = np.dot(ndt_design, ndt_weights_est)
        
        # Transform trace into List[Parameters]
        trace_parameters = [
            Parameters(
                boundary=boundary, 
                drift=drift, 
                ndt=ndt
            ) for boundary, drift, ndt in zip(final_boundary, final_drift, final_ndt)
        ]
        
        # Return mean parameters and weights
        return (boundary_weights_est, drift_weights_est, ndt_weights_est, trace_parameters)


"""
Test suite
"""
class TestSuite(unittest.TestCase):
    def test_bayesian_parameter_estimation(self):
        pass
    
    def test_bayesian_multiple_parameter_estimation(self):
        pass


"""
Demonstrate design matrix parameter estimation
"""
def demonstrate_design_matrix_parameter_estimation():
    N = 56

    design_matrix = DesignMatrix(
        boundary_design  = np.array([[1, 0, 0], 
                                     [0, 1, 0], 
                                     [0, 0, 1], 
                                     [0, 0, 1]]),
        drift_design     = np.array([[1, 0, 0], 
                                     [0, 1, 0], 
                                     [0, 0, 1], 
                                     [0, 1, 0]]),
        ndt_design       = np.array([[1, 0, 0], 
                                     [0, 1, 0], 
                                     [0, 0, 1], 
                                     [1, 0, 0]]),
        boundary_weights = np.array([1.0, 1.5, 2.0]),
        drift_weights    = np.array([0.4, 0.8, 1.2]),
        ndt_weights      = np.array([0.3, 0.4, 0.5]))
    
    true_bounds = design_matrix.boundary()
    true_drifts = design_matrix.drift()
    true_ndts   = design_matrix.ndt()

    true_params_list = [
        Parameters(
            boundary=bound, 
            drift=drift, 
            ndt=ndt
        ) for bound, drift, ndt in zip(true_bounds, true_drifts, true_ndts)
    ]

    pred_stats_list = ez.forward(true_params_list) 

    obs_stats_list = [
        s.sample(sample_size=N) 
        for s in pred_stats_list
    ]

    output, est_params, trace = bayesian_design_matrix_parameter_estimation(
        obs_stats_list, 
        design_matrix.boundary_nd(), 
        design_matrix.drift_nd(), 
        design_matrix.ndt_nd(), 
        n_samples=5000, 
        n_tune=2500,
        n_chains=4,
        verbosity=0
    )

    print("\n# True parameters:\n")
    [print(t) for t in true_params_list]

    print("\n# Estimated parameters:\n")
    [print(e) for e in est_params]

    print("\n# Weights:\n")
    print(f"Boundary weights : {output['boundary_weights_est']} ({output['boundary_weights_std']})")
    print(f"Drift weights    : {output['drift_weights_est']} ({output['drift_weights_std']})")
    print(f"NDT weights      : {output['ndt_weights_est']} ({output['ndt_weights_std']})")

    print("\n# True weights:\n")
    print(f"Boundary weights : {design_matrix.boundary_weights()}")
    print(f"Drift weights    : {design_matrix.drift_weights()}")
    print(f"NDT weights      : {design_matrix.ndt_weights()}")

    print(f"\n--------------------------------------------\n")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run the test suite")
    parser.add_argument("--demo", action="store_true", help="Run the demo")
    args = parser.parse_args()
    
    if args.test:
        unittest.main(argv=[__file__], verbosity=0, failfast=True)

    if args.demo:
        demonstrate_design_matrix_parameter_estimation()