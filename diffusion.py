#!/usr/bin/env python3

import numpy as np
from typing import Tuple, List
import unittest
import argparse
import pymc as pm
import arviz as az

class Parameters:
    """Model parameters for EZ diffusion"""
    def __init__(self, 
                 boundary: float, 
                 drift: float, 
                 ndt: float):
        if not isinstance(boundary, (int, float)):
            raise TypeError("Boundary must be a number.")
        if not isinstance(drift, (int, float)):
            raise TypeError("Drift must be a number.")
        if not isinstance(ndt, (int, float)):
            raise TypeError("NDT must be a number.")

        self.boundary = boundary
        self.drift = drift
        self.ndt = ndt
        
    def __sub__(self, other):
        if not isinstance(other, Parameters):
            raise TypeError("other must be an instance of Parameters")
        return Parameters(self.boundary - other.boundary,
                          self.drift - other.drift, 
                          self.ndt - other.ndt)

    def __str__(self):
        return f"Boundary: {self.boundary:6.2f}, " + \
               f"Drift: {self.drift:6.2f}, " + \
               f"NDT: {self.ndt:6.2f}"

class SummaryStats:
    """Summary statistics from behavioral data"""
    def __init__(self, 
                 accuracy: float,
                 mean_rt: float, 
                 var_rt: float):
        if not isinstance(accuracy, (int, float)):
            raise TypeError("Accuracy must be a number.")
        if not isinstance(mean_rt, (int, float)):
            raise TypeError("Mean RT must be a number.")
        if not isinstance(var_rt, (int, float)):
            raise TypeError("Var RT must be a number.")

        self.accuracy = accuracy
        self.mean_rt = mean_rt
        self.var_rt = var_rt
        
    def __sub__(self, other):
        if not isinstance(other, SummaryStats):
            raise TypeError("other must be an instance of SummaryStats")
        return SummaryStats(self.accuracy - other.accuracy, 
                            self.mean_rt - other.mean_rt, 
                            self.var_rt - other.var_rt)

    def __str__(self):
        return f"Accuracy: {self.accuracy:6.2f}," + \
               f"Mean RT: {self.mean_rt:6.2f}," + \
               f"Var RT: {self.var_rt:6.2f}"

class Results:
    """Results from a single simulation"""
    def __init__(self, 
                 true_params: Parameters,
                 est_params: Parameters, 
                 bias: Parameters, 
                 sq_error: Parameters, 
                 relative_bias: Parameters):
        if not isinstance(true_params, Parameters):
            raise TypeError("true_params must be an instance of Parameters")
        
        if not isinstance(est_params, Parameters):
            raise TypeError("est_params must be an instance of Parameters")
        
        if not isinstance(bias, Parameters):
            raise TypeError("bias must be an instance of Parameters")
        
        if not isinstance(sq_error, Parameters):
            raise TypeError("sq_error must be an instance of Parameters")
        
        if not isinstance(relative_bias, Parameters):
            raise TypeError("relative_bias must be an instance of Parameters")

        self._true_params = true_params
        self._est_params = est_params
        self._bias = bias
        self._sq_error = sq_error
        self._relative_bias = relative_bias
    
    def true_params(self):
        return self._true_params
    
    def est_params(self):
        return self._est_params
    
    def bias(self):
        return self._bias
    
    def sq_error(self):
        return self._sq_error
    
    def relative_bias(self):
        return self._relative_bias
    
    def __sub__(self, other):
        if not isinstance(other, Results):
            raise TypeError("other must be an instance of Results")
        return Results(self.true_params() - other.true_params(), 
                       self.est_params() - other.est_params(), 
                       self.bias() - other.bias(), 
                       self.sq_error() - other.sq_error(),
                       self.relative_bias() - other.relative_bias())

    def __str__(self):        
        return f"{'Bias':16s} : {self._bias}\n" + \
               f"{'Squared error':16s} : {self._sq_error}\n" + \
               f"{'Relative bias':16s} : {self._relative_bias}"

def forward_equations(params: Parameters) -> SummaryStats:
    """Calculate predicted summary statistics from parameters"""
    if not isinstance(params, Parameters):
        raise TypeError("params must be an instance of Parameters")
    if params.drift == 0:
        raise ValueError("Drift must be non-zero.")
    if params.boundary <= 0:
        raise ValueError("Boundary must be strictly positive.")
    
    # Helper calculation
    y = np.exp(-params.boundary * params.drift)
    
    # Calculate predicted statistics
    pred_acc = 1 / (y + 1)
    
    pred_mean = (params.ndt + 
                (params.boundary / (2 * params.drift)) * 
                ((1 - y) / (1 + y)))
    
    pred_var = ((params.boundary / (2 * params.drift**3)) * 
                ((1 - 2*params.boundary*params.drift*y - y**2) / 
                 ((y + 1)**2)))
    
    return SummaryStats(pred_acc, pred_mean, pred_var)

def sample_statistics(pred_stats: SummaryStats, N: int) -> SummaryStats:
    """Generate observed summary statistics using sampling distributions"""
    # Sample accuracy (binomial)
    obs_acc_total = np.random.binomial(N, pred_stats.accuracy)
    obs_acc = obs_acc_total / N
    
    # Sample mean RT (normal)
    obs_mean = np.random.normal(pred_stats.mean_rt, 
                              np.sqrt(pred_stats.var_rt/N))
    
    # Sample variance (gamma approximation)
    obs_var = np.random.gamma((N-1)/2, 2*pred_stats.var_rt/(N-1))
        
    return SummaryStats(obs_acc, obs_mean, obs_var)

def inverse_equations(obs_stats: SummaryStats) -> Parameters:
    """Estimate parameters from observed summary statistics"""
    if not isinstance(obs_stats, SummaryStats):
        raise TypeError("obs_stats must be an instance of SummaryStats")
    
    # Helper calculations
    if obs_stats.accuracy == 0 or obs_stats.accuracy == 1:
        return Parameters(0, 0, 0)
    
    logit_acc = np.log(obs_stats.accuracy / (1 - obs_stats.accuracy))
    
    # Calculate drift rate
    numerator = logit_acc * (
        obs_stats.accuracy**2 * logit_acc - 
        obs_stats.accuracy * logit_acc + 
        obs_stats.accuracy - 0.5
    )
    
    # Add sign based on accuracy
    sign = 1 if obs_stats.accuracy > 0.5 else -1
    
    try:
        est_drift = sign * np.power(numerator / obs_stats.var_rt, 0.25)
        if abs(est_drift) < 1e-9:
            return Parameters(1.0, 0.0, obs_stats.mean_rt)
            
        # Calculate boundary
        est_bound = abs(logit_acc / est_drift)  # Take absolute value for stability
        
        # Calculate non-decision time
        y = np.exp(-est_drift * est_bound)
        est_ndt = obs_stats.mean_rt - (
            (est_bound / (2 * est_drift)) * 
            ((1 - y) / (1 + y))
        )
                
        return Parameters(est_bound, est_drift, est_ndt)
        
    except (ValueError, RuntimeWarning, ZeroDivisionError):
        # Return reasonable defaults for numerical failures
        print("WARNING: Numerical failure for N = {N}")
        return Parameters(1.0, 0.0, obs_stats.mean_rt)

def generate_random_parameters(lower_bound: Parameters, 
                               upper_bound: Parameters) -> Parameters:
    """Generate random parameters within given bounds"""
    boundary = np.random.uniform(lower_bound.boundary, upper_bound.boundary)
    drift = np.random.uniform(lower_bound.drift, upper_bound.drift)
    ndt = np.random.uniform(lower_bound.ndt, upper_bound.ndt)
    return Parameters(boundary, drift, ndt)

def calculate_error(true: Parameters, 
                    estimated: Parameters) -> Tuple[Parameters, Parameters]:
    """Calculate bias and squared error"""
    bias = Parameters(
        true.boundary - estimated.boundary,
        true.drift - estimated.drift,
        true.ndt - estimated.ndt
    )
    
    relative_bias = Parameters(
        (true.boundary - estimated.boundary) / true.boundary,
        (true.drift - estimated.drift) / true.drift,
        (true.ndt - estimated.ndt) / true.ndt
    )
    
    squared_error = Parameters(
        bias.boundary**2,
        bias.drift**2,
        bias.ndt**2
    )
    
    return bias, squared_error, relative_bias

def mean_parameters(results: List[Results]) -> Results:
    """Calculate the mean parameters from a list of results"""
    return Results(
        Parameters(
            np.mean([result.true_params().boundary for result in results]),
            np.mean([result.true_params().drift for result in results]),
            np.mean([result.true_params().ndt for result in results])
        ),
        Parameters(
            np.mean([result.est_params().boundary for result in results]),
            np.mean([result.est_params().drift for result in results]),
            np.mean([result.est_params().ndt for result in results])
        ),
        Parameters(
            np.mean([result.bias().boundary for result in results]),
            np.mean([result.bias().drift for result in results]),
            np.mean([result.bias().ndt for result in results])
        ),
        Parameters(
            np.mean([result.sq_error().boundary for result in results]),
            np.mean([result.sq_error().drift for result in results]),
            np.mean([result.sq_error().ndt for result in results])
        ),
        Parameters(
            np.mean([result.relative_bias().boundary for result in results]),
            np.mean([result.relative_bias().drift for result in results]),
            np.mean([result.relative_bias().ndt for result in results])
        )
    )

def run_simulation(N: int, 
                   lower_bound: Parameters, 
                   upper_bound: Parameters) -> Results:
    """Run a single simulation and return true, estimated, and bias parameters"""
    results = []
    for _ in range(10000):
        true_params = generate_random_parameters(lower_bound, upper_bound)
        pred_stats = forward_equations(true_params)
        obs_stats = sample_statistics(pred_stats, N)
        est_params = inverse_equations(obs_stats)
        bias, sq_error, relative_bias = calculate_error(true_params, est_params)
        results.append(Results(true_params, est_params, bias, sq_error, relative_bias))
    
    return mean_parameters(results)

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

def design_matrix_parameter_estimation(
    obs_stats: List[SummaryStats], 
    N: List[int], 
    boundary_design: np.ndarray,
    drift_design: np.ndarray,
    ndt_design: np.ndarray
) -> Tuple[Parameters, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate EZ-diffusion parameters with uncertainty given observed statistics and design matrices.
    Args:
        obs_stats: Observed summary statistics (array of SummaryStats)
        N: Number of trials used to generate the statistics (array of ints)
        boundary_design: Design matrix for boundary parameter (n_conditions x n_boundary_weights)
        drift_design: Design matrix for drift parameter (n_conditions x n_drift_weights)
        ndt_design: Design matrix for NDT parameter (n_conditions x n_ndt_weights)
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
    boundary_design = np.array(boundary_design)
    drift_design    = np.array(drift_design)
    ndt_design      = np.array(ndt_design)
    qnd_params = []
    # Resample the observed statistics 1000 times
    for _ in range(1000):
        resampled_obs_stats = [sample_statistics(o, n) for o, n in zip(obs_stats, N)]
        weights = qnd_parameter_estimation(resampled_obs_stats, N, boundary_design, drift_design, ndt_design)
        qnd_params.append(weights)
    return qnd_params

# design_matrix_parameter_estimation(obs_stats_list, [56,56,56,56], boundary_design, drift_design, ndt_design)

def qnd_parameter_estimation(
    obs_stats: List[SummaryStats], 
    N: List[int], 
    boundary_design: np.array,
    drift_design: np.array,
    ndt_design: np.array
) -> Parameters:
    """Estimate EZ-diffusion parameters with uncertainty given observed statistics and design matrices.
    
    Args:
        obs_stats: Observed summary statistics (array of SummaryStats)
        N: Number of trials used to generate the statistics (array of ints)
        boundary_design: Design matrix for boundary parameter (n_conditions x n_boundary_weights)
        drift_design: Design matrix for drift parameter (n_conditions x n_drift_weights)
        ndt_design: Design matrix for NDT parameter (n_conditions x n_ndt_weights)
        
    Returns:
        Estimated parameters (Parameters)
        - Boundary weights (np.array)
        - Drift weights (np.array)
        - NDT weights (np.array)
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
    
    # Make the parameters into numpy arrays
    boundary_array = np.array([qnd_params.boundary for qnd_params in qnd_params_list])
    drift_array    = np.array([qnd_params.drift for qnd_params in qnd_params_list])
    ndt_array      = np.array([qnd_params.ndt for qnd_params in qnd_params_list])
    
    # Solve the linear system of equations to get the weights for each parameter
    qnd_bound_weights = np.linalg.pinv(boundary_design) @ boundary_array.T
    qnd_drift_weights = np.linalg.pinv(drift_design) @ drift_array.T
    qnd_ndt_weights   = np.linalg.pinv(ndt_design) @ ndt_array.T
        
    return qnd_bound_weights, qnd_drift_weights, qnd_ndt_weights
    
    
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
        
        # Compute final parameter estimates
        final_boundary = np.dot(boundary_design, boundary_weights_est)
        final_drift = np.dot(drift_design, drift_weights_est)
        final_ndt = np.dot(ndt_design, ndt_weights_est)
        
        # Transform trace into List[Parameters]
        trace_parameters = [Parameters(boundary, drift, ndt) for boundary, drift, ndt in zip(final_boundary, final_drift, final_ndt)]
        
        # Return mean parameters and weights
        return (trace_parameters, boundary_weights_est, drift_weights_est, ndt_weights_est)
    
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
        return (trace_parameters, boundary_weights_est, drift_weights_est, ndt_weights_est)

# Here is the test suite
class TestSuite(unittest.TestCase):
    def test_create_parameters(self):
        params = Parameters(1.0, 1.0, 0.0)
        self.assertIsInstance(params, Parameters)
        self.assertEqual(params.boundary, 1.0)
        self.assertEqual(params.drift, 1.0)
        self.assertEqual(params.ndt, 0.0)
    
    def test_create_summary_stats(self):
        stats = SummaryStats(0.66, 0.33, 0.05)
        self.assertIsInstance(stats, SummaryStats)
        self.assertEqual(stats.accuracy, 0.66)
        self.assertEqual(stats.mean_rt, 0.33)
        self.assertEqual(stats.var_rt, 0.05)
    
    def test_forward_equations(self):
        true_params = Parameters(1.0, 0.5, 0.2)
        pred_stats = forward_equations(true_params)
        self.assertIsInstance(pred_stats, SummaryStats)
        self.assertGreater(pred_stats.accuracy, 0.5)
        
        true_params = Parameters(1.0, -0.5, 0.2)
        pred_stats = forward_equations(true_params)
        self.assertIsInstance(pred_stats, SummaryStats)
        self.assertLess(pred_stats.accuracy, 0.5)

    def test_inverse_equations(self):
        obs_stats = SummaryStats(0.66, 0.33, 0.05)
        est_params = inverse_equations(obs_stats)
        self.assertIsInstance(est_params, Parameters)
    
    def test_create_parameters_input_validation(self):
        self.assertRaises(TypeError, Parameters, None, .2, .2)
        self.assertRaises(TypeError, Parameters, .2, None, .2)
        self.assertRaises(TypeError, Parameters, .2, .2, None)
        self.assertRaises(TypeError, Parameters, "a", "b", "c" )
        self.assertRaises(TypeError, Parameters, [.2, .2, .2])
        
    def test_create_summary_stats_input_validation(self):
        self.assertRaises(TypeError, SummaryStats, None, .2, .2)
        self.assertRaises(TypeError, SummaryStats, .2, None, .2)
        self.assertRaises(TypeError, SummaryStats, .2, .2, None)
        self.assertRaises(TypeError, SummaryStats, "a", "b", "c" )
        self.assertRaises(TypeError, SummaryStats, [.2, .2, .2])
    
    def test_forward_equations_input_validation(self):
        self.assertRaises(TypeError, forward_equations, SummaryStats(0.66, 0.33, 0.05))
        self.assertRaises(TypeError, forward_equations, None)
        self.assertRaises(TypeError, forward_equations, 1)
        self.assertRaises(TypeError, forward_equations, "string")
        self.assertRaises(TypeError, forward_equations, "s")
        self.assertRaises(TypeError, forward_equations, [1, 2, 3])
        self.assertRaises(ValueError, forward_equations, Parameters(1.0, 0.0, 0.0))
        self.assertRaises(ValueError, forward_equations, Parameters(0.0, 1.0, 0.0))
        
    def test_inverse_equations_input_validation(self):
        self.assertRaises(TypeError, inverse_equations, Parameters(1.0, 0.5, 0.2))
        self.assertRaises(TypeError, inverse_equations, None)
        self.assertRaises(TypeError, inverse_equations, 1)
        self.assertRaises(TypeError, inverse_equations, "string")
        self.assertRaises(TypeError, inverse_equations, "s")
        self.assertRaises(TypeError, inverse_equations, [1, 2, 3])
        
    def test_ping_pong(self):
        true_params = Parameters(1.0, 0.5, 0.2)
        difference = true_params - inverse_equations(forward_equations(true_params))
        self.assertLess(difference.boundary, 1e-6)
        self.assertLess(difference.drift, 1e-6)
        self.assertLess(difference.ndt, 1e-6)
        
        true_params = Parameters(4.0, 1.0, 0.0)
        difference = true_params - inverse_equations(forward_equations(true_params))
        self.assertLess(difference.boundary, 1e-6)
        self.assertLess(difference.drift, 1e-6)
        self.assertLess(difference.ndt, 1e-6)
        
        true_params = Parameters(1.0, -1.0, 0.0)
        difference = true_params - inverse_equations(forward_equations(true_params))
        self.assertLess(difference.boundary, 1e-6)
        self.assertLess(difference.drift, 1e-6)
        self.assertLess(difference.ndt, 1e-6)

    def test_sample_statistics(self):
        pred_stats = SummaryStats(0.66, 0.33, 0.05)
        obs_stats = sample_statistics(pred_stats, 10)
        self.assertIsInstance(obs_stats, SummaryStats)
        
    def test_bayesian_parameter_estimation(self):
        # Generate some test data
        true_params = Parameters(1.0, 0.5, 0.2)
        pred_stats = forward_equations(true_params)
        obs_stats = sample_statistics(pred_stats, 100)  # Use more trials for better estimates
        
        # Run Bayesian estimation
        est_params = bayesian_parameter_estimation(obs_stats, 100, n_samples=100, n_tune=50)
        
        # Check that estimates are reasonable
        self.assertIsInstance(est_params, Parameters)
        self.assertGreater(est_params.boundary, 0)
        self.assertGreater(est_params.ndt, 0)
        
        # Check that estimates are within reasonable range of true values
        self.assertLess(abs(est_params.boundary - true_params.boundary), 1.0)
        self.assertLess(abs(est_params.drift - true_params.drift), 1.0)
        self.assertLess(abs(est_params.ndt - true_params.ndt), 0.5)


def demonstrate_design_matrix_parameter_estimation():
    N = 56

    # Run Bayesian estimation
    boundary_design = np.array([[1, 0, 0], 
                                [0, 1, 0], 
                                [0, 0, 1], 
                                [0, 0, 1]])

    drift_design    = np.array([[1, 0, 0], 
                                [0, 1, 0], 
                                [0, 0, 1], 
                                [0, 1, 0]])

    ndt_design      = np.array([[1, 0, 0], 
                                [0, 1, 0], 
                                [0, 0, 1], 
                                [1, 0, 0]])

    # Simulate data from true weights
    true_bound_weights = np.array([1.0, 1.5, 2.0])
    true_drift_weights = np.array([0.4, 0.8, 1.2])
    true_ndt_weights   = np.array([0.3, 0.4, 0.5])

    true_bounds = boundary_design @ true_bound_weights
    true_drifts = drift_design @ true_drift_weights
    true_ndts   = ndt_design @ true_ndt_weights

    true_params_list = [Parameters(bound, drift, ndt) for bound, drift, ndt in zip(true_bounds, true_drifts, true_ndts)]
    pred_stats_list = [forward_equations(true_params) for true_params in true_params_list]
    obs_stats_list = [sample_statistics(pred_stats, N) for pred_stats in pred_stats_list]

    est_params, boundary_weights, drift_weights, ndt_weights = \
        bayesian_design_matrix_parameter_estimation(obs_stats_list, 
                                                    [N, N, N, N], 
                                                    boundary_design, 
                                                    drift_design, 
                                                    ndt_design, 
                                                    n_samples=5000, 
                                                    n_tune=2500)

    print("\n# True parameters:\n")
    [print(t) for t in true_params_list]

    print("\n# Estimated parameters:\n")
    [print(e) for e in est_params]

    print("\n# Weights:\n")
    print(f"Boundary weights : {boundary_weights}")
    print(f"Drift weights    : {drift_weights}")
    print(f"NDT weights      : {ndt_weights}")

    print(f"\n--------------------------------------------\n")

def run_ezdiffusion_simulation_study():
    # Define bounds before running simulation
    lower_bound = Parameters(0.5, 1.0, 0.2)
    upper_bound = Parameters(2.0, 2.0, 0.5)
    
    for N in [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240]:
        print(f"N = {N}")
        print(run_simulation(N, lower_bound, upper_bound))

def demonstrate_qnd_parameter_estimation():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--test", action="store_true", help="Run the test suite")
    parser.add_argument("--simstudy", action="store_true", help="Run the simulation study")
    parser.add_argument("--demo", action="store_true", help="Run the demo")
    parser.add_argument("--bayes", action="store_true", help="Run the Bayesian parameter estimation")
    parser.add_argument("--qnd", action="store_true", help="Run the QND parameter estimation")
    
    args = parser.parse_args()
    
    if args.test:
        unittest.main(argv=[__file__], verbosity=0, failfast=True)
    
    if args.simstudy:
        run_ezdiffusion_simulation_study()
    
    if args.demo:
        demonstrate_design_matrix_parameter_estimation()
    
    if args.qnd:
        demonstrate_qnd_parameter_estimation()
    
    if args.bayes:
        demonstrate_design_matrix_parameter_estimation()
    
    