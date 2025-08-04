#!/usr/bin/env python3

import numpy as np
from typing import Tuple, List
import unittest
import sys
import pymc as pm
import arviz as az
from base import SummaryStats, Parameters, inverse_equations

def bayesian_parameter_estimation(obs_stats: SummaryStats, 
                                  N: int, 
                                n_samples: int = 2000, 
                                n_tune: int = 1000) -> Parameters:
    """Estimate EZ-diffusion parameters using PyMC given observed statistics.
    
    Args:
        obs_stats: Observed summary statistics
        N: Number of trials used to generate the statistics
        n_samples: Number of MCMC samples to draw
        n_tune: Number of tuning steps for MCMC
        
    Returns:
        Estimated parameters
    """
    if not isinstance(obs_stats, SummaryStats):
        raise TypeError("obs_stats must be an instance of SummaryStats")
    
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
                            observed=int(obs_stats.accuracy * N))
        
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
                          observed=obs_stats.mean_rt)
        
        # Likelihood for variance (gamma approximation)
        var_obs = pm.Gamma('var_obs',
                          alpha=(N-1)/2,
                          beta=(N-1)/(2*pred_var),
                          observed=obs_stats.var_rt)
        
        # Run MCMC
        trace = pm.sample(n_samples, tune=n_tune, return_inferencedata=True)
        
        print(az.summary(trace))
        
        # Get posterior means
        post_mean = az.summary(trace)['mean']
        
        return Parameters(
            boundary=float(post_mean['boundary']),
            drift=float(post_mean['drift']),
            ndt=float(post_mean['ndt'])
        )

def bayesian_multiple_parameter_estimation(
    obs_stats: List[SummaryStats], 
    N: List[int], 
    n_samples: int = 2000, 
    n_tune: int = 1000,
    n_chains: int = 4 
) -> Parameters:
    """Estimate EZ-diffusion parameters using PyMC given observed statistics.
    
    Args:
        obs_stats: Observed summary statistics (array of SummaryStats)
        N: Number of trials used to generate the statistics (array of ints)
        n_samples: Number of MCMC samples to draw
        n_tune: Number of tuning steps for MCMC
        
    Returns:
        Estimated parameters
    """
    if not isinstance(obs_stats, list):
        raise TypeError("obs_stats must be a list")
    if not all(isinstance(x, SummaryStats) for x in obs_stats):
        raise TypeError("All elements in obs_stats must be SummaryStats instances")
    if not isinstance(N, list):
        raise TypeError("N must be a list")
    if not all(isinstance(x, int) for x in N):
        raise TypeError("All elements in N must be integers")
    if len(obs_stats) != len(N):
        raise ValueError("obs_stats and N must have the same length")
    
    # Define the Bayesian model
    with pm.Model() as model:
        # Priors for parameters
        boundary = pm.Uniform('boundary', lower=0.1, upper=5.0, shape=len(obs_stats))
        drift = pm.Normal('drift', mu=0, sigma=2.0, shape=len(obs_stats))
        ndt = pm.Uniform('ndt', lower=0.0, upper=1.0, shape=len(obs_stats))
        
        # Helper calculation for accuracy
        y = pm.math.exp(-boundary * drift)
        pred_acc = 1 / (y + 1)
        
        # Likelihood for accuracy (binomial)
        acc_obs = pm.Binomial('acc_obs', 
                            n=N, 
                            p=pred_acc, 
                            observed=[int(s.accuracy * n) for s, n in zip(obs_stats, N)])
        
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
                          observed=[s.mean_rt for s in obs_stats])
        
        # Likelihood for variance (gamma approximation)
        var_obs = pm.Gamma('var_obs',
                          alpha=[(n-1)/2 for n in N],
                          beta=[(n-1)/(2*v) for n, v in zip(N, pred_var)],
                          observed=[s.var_rt for s in obs_stats])
        
        # Run MCMC
        trace = pm.sample(n_samples, tune=n_tune, return_inferencedata=True)
        
        print(az.summary(trace))
        
        # Get posterior means
        post_mean = az.summary(trace)['mean']
        
        # Return mean parameters across all cells
        return Parameters(
            boundary=float(np.mean([post_mean[f'boundary[{i}]'] for i in range(len(obs_stats))])),
            drift=float(np.mean([post_mean[f'drift[{i}]'] for i in range(len(obs_stats))])),
            ndt=float(np.mean([post_mean[f'ndt[{i}]'] for i in range(len(obs_stats))]))
        )

    
def bayesian_design_matrix_parameter_estimation(
    obs_stats: List[SummaryStats], 
    N: List[int], 
    boundary_design: np.ndarray,
    drift_design: np.ndarray,
    ndt_design: np.ndarray,
    n_samples: int = 2000, 
    n_tune: int = 1000,
    n_chains: int = 4 
) -> Tuple[Parameters, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate EZ-diffusion parameters using PyMC given observed statistics and design matrices.
    
    Args:
        obs_stats: Observed summary statistics (array of SummaryStats)
        N: Number of trials used to generate the statistics (array of ints)
        boundary_design: Design matrix for boundary parameter (n_conditions x n_boundary_weights)
        drift_design: Design matrix for drift parameter (n_conditions x n_drift_weights)
        ndt_design: Design matrix for NDT parameter (n_conditions x n_ndt_weights)
        n_samples: Number of MCMC samples to draw
        n_tune: Number of tuning steps for MCMC
        
    Returns:
        Tuple containing:
        - Estimated parameters (Parameters)
        - Boundary weights (np.ndarray)
        - Drift weights (np.ndarray)
        - NDT weights (np.ndarray)
    """
    if not isinstance(obs_stats, list):
        raise TypeError("obs_stats must be a list")
    if not all(isinstance(x, SummaryStats) for x in obs_stats):
        raise TypeError("All elements in obs_stats must be SummaryStats instances")
    if not isinstance(N, list):
        raise TypeError("N must be a list")
    if not all(isinstance(x, int) for x in N):
        raise TypeError("All elements in N must be integers")
    if len(obs_stats) != len(N):
        raise ValueError("obs_stats and N must have the same length")
    
    # Use the inverse equations to get quick and dirty estimates of the parameters
    qnd_params_list = [inverse_equations(obs_stats) for obs_stats in obs_stats]
    print("\n# QND parameters:\n")
    [print(qnd_params) for qnd_params in qnd_params_list]
    
    # Make the design matrices into numpy matrices
    boundary_design_matrix = np.array(boundary_design)
    drift_design_matrix    = np.array(drift_design)
    ndt_design_matrix      = np.array(ndt_design)
    
    # Solve the linear system of equations to get the weights for each parameter
    qnd_bound_weights = np.linalg.pinv(boundary_design_matrix) @ np.array([qnd_params.boundary for qnd_params in qnd_params_list]).T
    qnd_drift_weights = np.linalg.pinv(drift_design_matrix) @ np.array([qnd_params.drift for qnd_params in qnd_params_list]).T
    qnd_ndt_weights   = np.linalg.pinv(ndt_design_matrix) @ np.array([qnd_params.ndt for qnd_params in qnd_params_list]).T

    print("\n# QND weights:\n")
    print(qnd_bound_weights)
    print(qnd_drift_weights)
    print(qnd_ndt_weights)
    
    # Print the implied constrained parameters (transform into List[Parameters])
    print("\n# Implied constrained parameters:\n")
    implied_boundaries = boundary_design_matrix @ qnd_bound_weights
    implied_drifts = drift_design_matrix @ qnd_drift_weights
    implied_ndts = ndt_design_matrix @ qnd_ndt_weights
    [print(Parameters(boundary, drift, ndt)) for boundary, drift, ndt in zip(implied_boundaries, implied_drifts, implied_ndts)]
    
    # Use the quick and dirty weights as initial values for the Bayesian model
    init_values = {
        'boundary_weights': qnd_bound_weights,
        'drift_weights': qnd_drift_weights,
        'ndt_weights': qnd_ndt_weights
    }
    
    n_conditions = len(obs_stats)
    
    # Validate design matrices
    if boundary_design.shape[0] != n_conditions:
        raise ValueError(f"boundary_design must have {n_conditions} rows")
    if drift_design.shape[0] != n_conditions:
        raise ValueError(f"drift_design must have {n_conditions} rows")
    if ndt_design.shape[0] != n_conditions:
        raise ValueError(f"ndt_design must have {n_conditions} rows")
    
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
        boundary = pm.Deterministic('boundary', 
                                  pm.math.dot(boundary_design, boundary_weights))
        drift = pm.Deterministic('drift', 
                               pm.math.dot(drift_design, drift_weights))
        ndt = pm.Deterministic('ndt', 
                             pm.math.dot(ndt_design, ndt_weights))
        
        # Helper calculation for accuracy
        y = pm.math.exp(-boundary * drift)
        pred_acc = 1 / (y + 1)
        
        # Likelihood for accuracy (binomial)
        acc_obs = pm.Binomial('acc_obs', 
                            n=N, 
                            p=pred_acc, 
                            observed=[int(s.accuracy * n) for s, n in zip(obs_stats, N)])
        
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
                          observed=[s.mean_rt for s in obs_stats])
        
        # Likelihood for variance (gamma approximation)
        var_obs = pm.Gamma('var_obs',
                          alpha=[(n-1)/2 for n in N],
                          beta=[(n-1)/(2*v) for n, v in zip(N, pred_var)],
                          observed=[s.var_rt for s in obs_stats])
        
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
        
        # Get posterior standard deviations
        try:
            post_std = az.summary(trace)['sd']
        except KeyError:
            # List all keys in the trace
            print(az.summary(trace).index)
            raise KeyError("sd not found in trace")
        
        # Get weight standard deviations
        boundary_weights_std = np.array([post_std[f'boundary_weights[{i}]'] 
                                       for i in range(boundary_design.shape[1])])
        drift_weights_std = np.array([post_std[f'drift_weights[{i}]'] 
                                    for i in range(drift_design.shape[1])])
        ndt_weights_std = np.array([post_std[f'ndt_weights[{i}]'] 
                                  for i in range(ndt_design.shape[1])])
        
        # Compute final parameter estimates
        final_boundary = np.dot(boundary_design, boundary_weights_est)
        final_drift = np.dot(drift_design, drift_weights_est)
        final_ndt = np.dot(ndt_design, ndt_weights_est)
        
        # Transform trace into List[Parameters]
        trace_parameters = [Parameters(boundary, drift, ndt) for boundary, drift, ndt in zip(final_boundary, final_drift, final_ndt)]
        
        # Return mean parameters and weights
        return (boundary_weights_est, drift_weights_est, ndt_weights_est,
                boundary_weights_std, drift_weights_std, ndt_weights_std,
                trace_parameters)
    
def bayesian_design_matrix_parameter_estimation_with_hierarchical_structure(
    obs_stats: List[SummaryStats], 
    N: List[int], 
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
        obs_stats: Observed summary statistics (array of SummaryStats)
        N: Number of trials used to generate the statistics (array of ints)
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
    if not isinstance(obs_stats, list):
        raise TypeError("obs_stats must be a list")
    if not all(isinstance(x, SummaryStats) for x in obs_stats):
        raise TypeError("All elements in obs_stats must be SummaryStats instances")
    if not isinstance(N, list):
        raise TypeError("N must be a list")
    if not all(isinstance(x, int) for x in N):
        raise TypeError("All elements in N must be integers")
    if len(obs_stats) != len(N):
        raise ValueError("obs_stats and N must have the same length")
    
    # Use the inverse equations to get quick and dirty estimates of the parameters
    qnd_params_list = [inverse_equations(obs_stats) for obs_stats in obs_stats]
    print("\n# QND parameters:\n")
    [print(qnd_params) for qnd_params in qnd_params_list]
    
    # Make the design matrices into numpy matrices
    boundary_design_matrix = np.array(boundary_design)
    drift_design_matrix    = np.array(drift_design)
    ndt_design_matrix      = np.array(ndt_design)
    
    # Solve the linear system of equations to get the weights for each parameter
    qnd_bound_weights = np.linalg.pinv(boundary_design_matrix) @ np.array([qnd_params.boundary for qnd_params in qnd_params_list]).T
    qnd_drift_weights = np.linalg.pinv(drift_design_matrix) @ np.array([qnd_params.drift for qnd_params in qnd_params_list]).T
    qnd_ndt_weights   = np.linalg.pinv(ndt_design_matrix) @ np.array([qnd_params.ndt for qnd_params in qnd_params_list]).T

    print("\n# QND weights:\n")
    print(qnd_bound_weights)
    print(qnd_drift_weights)
    print(qnd_ndt_weights)
    
    # Print the implied constrained parameters (transform into List[Parameters])
    print("\n# Implied constrained parameters:\n")
    implied_boundaries = boundary_design_matrix @ qnd_bound_weights
    implied_drifts = drift_design_matrix @ qnd_drift_weights
    implied_ndts = ndt_design_matrix @ qnd_ndt_weights
    [print(Parameters(boundary, drift, ndt)) for boundary, drift, ndt in zip(implied_boundaries, implied_drifts, implied_ndts)]
    
    # Use the quick and dirty weights as initial values for the Bayesian model
    init_values = {
        'boundary_weights': qnd_bound_weights,
        'drift_weights': qnd_drift_weights,
        'ndt_weights': qnd_ndt_weights
    }
    
    n_conditions = len(obs_stats)
    
    # Validate design matrices
    if boundary_design.shape[0] != n_conditions:
        raise ValueError(f"boundary_design must have {n_conditions} rows")
    if drift_design.shape[0] != n_conditions:
        raise ValueError(f"drift_design must have {n_conditions} rows")
    if ndt_design.shape[0] != n_conditions:
        raise ValueError(f"ndt_design must have {n_conditions} rows")
    
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
                            observed=[int(s.accuracy * n) for s, n in zip(obs_stats, N)])
        
        rt_obs = pm.Normal('rt_obs',
                          mu=pred_mean,
                          sigma=pm.math.sqrt(pred_var/N),
                          observed=[s.mean_rt for s in obs_stats])
        
        # Likelihood for variance (gamma approximation)
        var_obs = pm.Gamma('var_obs',
                          alpha=[(n-1)/2 for n in N],
                          beta=[(n-1)/(2*v) for n, v in zip(N, pred_var)],
                          observed=[s.var_rt for s in obs_stats])
        
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
        trace_parameters = [Parameters(boundary, drift, ndt) for boundary, drift, ndt in zip(final_boundary, final_drift, final_ndt)]
        
        # Return mean parameters and weights
        return (boundary_weights_est, drift_weights_est, ndt_weights_est, trace_parameters)
