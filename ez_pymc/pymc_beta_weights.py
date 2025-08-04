#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
from typing import Tuple, List
import unittest
import argparse
import pymc as pm
import arviz as az
from vendor.ezas.base import ez_equations as ez
from vendor.ezas.classes.moments import Observations
from vendor.ezas.classes.parameters import Parameters
from vendor.ezas.classes.design_matrix import DesignMatrix, BetaWeights
from vendor.ezas.utils.posterior import PosteriorSummary
from vendor.ezas.qnd.qnd_beta_weights import qnd_beta_weights_estimation
from vendor.ezas.utils.prettify import b as pretty
from tqdm import tqdm
from pathlib import Path

_MAX_R_HAT = 1.1

# Example design matrix, we'll use this for the demo and simulation
_EXAMPLE_DESIGN_MATRIX = DesignMatrix(
        boundary_design = np.array([[1, 0, 0], 
                                    [0, 1, 0], 
                                    [0, 0, 1], 
                                    [0, 0, 1]]),
        drift_design    = np.array([[1, 0, 0], 
                                    [0, 1, 0], 
                                    [0, 0, 1], 
                                    [0, 1, 0]]),
        ndt_design      = np.array([[1, 0, 0], 
                                    [0, 1, 0], 
                                    [0, 0, 1], 
                                    [1, 0, 0]]),
        boundary_weights = np.array([1.0, 1.5, 2.0]),
        drift_weights    = np.array([0.4, 0.8, 1.2]),
        ndt_weights      = np.array([0.2, 0.3, 0.4])
    )

def pymc_design_matrix_parameter_estimation(
    observations: list[Observations],
    working_matrix: DesignMatrix,
    n_samples: int = 2000, 
    n_tune: int = 1000,
    n_chains: int = 4,
    verbosity: int = 2
) -> Tuple[BetaWeights, Parameters, az.InferenceData]:
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
    
    # Use the inverse equations to get quick and dirty estimates of the parameters
    qnd_weights, qnd_parameters, _ = qnd_beta_weights_estimation(
        observations = observations,
        working_matrix = working_matrix,
        n_bootstrap = 1000
    )
    
    if verbosity > 1:
        print("\n# QND parameters:\n")
        [print(p) for p in qnd_parameters]
    
    # # Parameter estimates
    # qnd_boundaries = np.array([p.boundary for p in qnd_parameters])
    # qnd_drifts     = np.array([p.drift    for p in qnd_parameters])
    # qnd_ndts       = np.array([p.ndt      for p in qnd_parameters])

    if verbosity > 1:
        print("\n# QND weights:\n")
        print(qnd_weights)
        
    # Use the quick and dirty weights as initial values for the Bayesian model
    init_values = {
        'boundary_weights' : qnd_weights.beta_boundary_mean(),
        'drift_weights'    : qnd_weights.beta_drift_mean(),
        'ndt_weights'      : qnd_weights.beta_ndt_mean()
    }
    
    n_conditions = len(observations)

    N        = np.array([o.sample_size() for o in observations])
    accuracy = np.array([o.accuracy()    for o in observations])
    mean_rt  = np.array([o.mean_rt()     for o in observations])
    var_rt   = np.array([o.var_rt()      for o in observations])
    
    # Validate design matrices
    bound_mtx = working_matrix.boundary_design()
    drift_mtx = working_matrix.drift_design()
    nondt_mtx   = working_matrix.ndt_design()
    
    if bound_mtx.shape[0] != n_conditions:
        raise ValueError(f"boundary_design must have {n_conditions} rows")
    if drift_mtx.shape[0] != n_conditions:
        raise ValueError(f"drift_design must have {n_conditions} rows")
    if nondt_mtx.shape[0] != n_conditions:
        raise ValueError(f"ndt_design must have {n_conditions} rows")
    
    if verbosity > 1:
        print("\n# Running PyMC...\n")
    
    # Define the Bayesian model
    with pm.Model() as model:
        # Priors for weights
        boundary_weights = pm.Normal('boundary_weights', 
                                   mu=0, 
                                   sigma=2.0, 
                                   shape=bound_mtx.shape[1])
        drift_weights = pm.Normal('drift_weights', 
                                mu=0, 
                                sigma=2.0, 
                                shape=drift_mtx.shape[1])
        ndt_weights = pm.Normal('ndt_weights', 
                              mu=0, 
                              sigma=2.0, 
                              shape=nondt_mtx.shape[1])
        
        # Compute parameters from design matrices and weights
        boundary = pm.Deterministic('boundary', 
                                  pm.math.dot(bound_mtx, boundary_weights))
        drift = pm.Deterministic('drift', 
                               pm.math.dot(drift_mtx, drift_weights))
        ndt = pm.Deterministic('ndt', 
                             pm.math.dot(nondt_mtx, ndt_weights))
        
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
                         progressbar=verbosity > -90,
                         target_accept=0.95,
                         compute_convergence_checks=False)  # Higher target acceptance rate
        
        # Print summary table for only those nodes with weight in their name
        summary = az.summary(trace)
        mask = summary.index.str.contains('weights')
        if verbosity > 1:
            print(summary.loc[mask])
        
        # Get posterior means
        post_mean = summary['mean']
        
        # Get posterior standard deviations
        post_std = summary['sd']
        
        # Extract weights
        weightList = BetaWeights(
            beta_boundary_mean  = np.array([post_mean[f'boundary_weights[{i}]' ] for i in range(bound_mtx.shape[1])]),
            beta_drift_mean     = np.array([post_mean[f'drift_weights[{i}]'    ] for i in range(drift_mtx.shape[1])]),
            beta_ndt_mean       = np.array([post_mean[f'ndt_weights[{i}]'      ] for i in range(nondt_mtx.shape[1])]),
            beta_boundary_sd    = np.array([post_std [f'boundary_weights[{i}]' ] for i in range(bound_mtx.shape[1])]),
            beta_drift_sd       = np.array([post_std [f'drift_weights[{i}]'    ] for i in range(drift_mtx.shape[1])]),
            beta_ndt_sd         = np.array([post_std [f'ndt_weights[{i}]'      ] for i in range(nondt_mtx.shape[1])]),
            beta_boundary_lower = np.array([np.percentile(trace.posterior['boundary_weights'].values[:, :, i],  2.5) for i in range(bound_mtx.shape[1])]),
            beta_drift_lower    = np.array([np.percentile(trace.posterior['drift_weights'   ].values[:, :, i],  2.5) for i in range(drift_mtx.shape[1])]),
            beta_ndt_lower      = np.array([np.percentile(trace.posterior['ndt_weights'     ].values[:, :, i],  2.5) for i in range(nondt_mtx.shape[1])]),
            beta_boundary_upper = np.array([np.percentile(trace.posterior['boundary_weights'].values[:, :, i], 97.5) for i in range(bound_mtx.shape[1])]),
            beta_drift_upper    = np.array([np.percentile(trace.posterior['drift_weights'   ].values[:, :, i], 97.5) for i in range(drift_mtx.shape[1])]),
            beta_ndt_upper      = np.array([np.percentile(trace.posterior['ndt_weights'     ].values[:, :, i], 97.5) for i in range(nondt_mtx.shape[1])])
        )
        # Extract parameters
        parameters = [
            Parameters(
                boundary = b, 
                drift    = d, 
                ndt      = n
            ) for b, d, n in zip(
                bound_mtx @ weightList.beta_boundary_mean(), 
                drift_mtx @ weightList.beta_drift_mean(), 
                nondt_mtx @ weightList.beta_ndt_mean()
            )
        ]
            
        return weightList, parameters, trace
    


"""
Test suite
"""
class TestSuite(unittest.TestCase):
    def test_pymc_parameter_estimation(self):
        pass
    
    def test_pymc_multiple_parameter_estimation(self):
        pass


"""
Demonstrate design matrix parameter estimation
"""
def demo():
    N = 200
    
    design_matrix = _EXAMPLE_DESIGN_MATRIX
    
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
    
    # Create a working design matrix
    working_matrix = DesignMatrix(
        boundary_design = design_matrix.boundary_design(),
        drift_design    = design_matrix.drift_design(),
        ndt_design      = design_matrix.ndt_design()
    )

    est_beta, est_params, trace = pymc_design_matrix_parameter_estimation(
        observations   = obs_stats_list, 
        working_matrix = working_matrix, 
        n_samples      = 2000, 
        n_tune         = 2500,
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

    print(f"\n--------------------------------------------\n")

    print(f"Coverage of boundary weights :", end=" ")
    for i in range(est_beta.beta_boundary_lower().shape[0]):
        covered = est_beta.beta_boundary_lower()[i] <= true_bounds[i] <= est_beta.beta_boundary_upper()[i]
        print(f"{pretty(covered)}", end=" ")
    
    print(f"\nCoverage of drift weights    :", end=" ")
    for i in range(est_beta.beta_drift_lower().shape[0]):
        covered = est_beta.beta_drift_lower()[i] <= true_drifts[i] <= est_beta.beta_drift_upper()[i]
        print(f"{pretty(covered)}", end=" ")
    
    print(f"\nCoverage of ndt weights      :", end=" ")
    for i in range(est_beta.beta_ndt_lower().shape[0]):
        covered = est_beta.beta_ndt_lower()[i] <= true_ndts[i] <= est_beta.beta_ndt_upper()[i]
        print(f"{pretty(covered)}", end=" ")

    print(f"\n\n--------------------------------------------\n")



"""
Simulate design matrix parameter estimation
"""
def simulation(repetitions: int = 1000):
    N = 200
    
    verbosity = 0

    design_matrix = _EXAMPLE_DESIGN_MATRIX
    
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
    
    # Create a working design matrix
    working_matrix = DesignMatrix(
        boundary_design = design_matrix.boundary_design(),
        drift_design    = design_matrix.drift_design(),
        ndt_design      = design_matrix.ndt_design()
    ) 
    
    import logging
    logger = logging.getLogger("pymc")
    logger.setLevel(logging.ERROR)
        
    # Initialize a progress bar
    progress_bar = tqdm(total=repetitions, desc="Simulation progress")
    
    boundary_weights_coverage = 0
    drift_weights_coverage    = 0
    ndt_weights_coverage      = 0

    n_boundary_weights = design_matrix.boundary_design().shape[1]
    n_drift_weights    = design_matrix.drift_design().shape[1]
    n_ndt_weights      = design_matrix.ndt_design().shape[1]

    for i in range(repetitions):

        est_beta, est_params, trace = pymc_design_matrix_parameter_estimation(
            observations   = [ s.sample(sample_size=N) for s in pred_stats_list ], 
            working_matrix = working_matrix, 
            n_samples      = 2000, 
            n_tune         = 2500,
            n_chains       = 4,
            verbosity      = verbosity
        )

        for j in range(n_boundary_weights):
            covered = est_beta.beta_boundary_lower()[j] <= true_bounds[j] <= est_beta.beta_boundary_upper()[j]
            boundary_weights_coverage += covered
            
        for j in range(n_drift_weights):
            covered = est_beta.beta_drift_lower()[j] <= true_drifts[j] <= est_beta.beta_drift_upper()[j]
            drift_weights_coverage += covered
            
        for j in range(n_ndt_weights):
            covered = est_beta.beta_ndt_lower()[j] <= true_ndts[j] <= est_beta.beta_ndt_upper()[j]
            ndt_weights_coverage += covered
        progress_bar.update(1)
    progress_bar.close()
    

    boundary_weights_coverage *= (100 / (repetitions * n_boundary_weights))
    drift_weights_coverage    *= (100 / (repetitions * n_drift_weights))
    ndt_weights_coverage      *= (100 / (repetitions * n_ndt_weights))

    print("Coverage:")
    print(f" > Boundary : {boundary_weights_coverage:5.1f}%  (should be ≈ 95%)")
    print(f" > Drift    : {drift_weights_coverage:5.1f}%  (should be ≈ 95%)")
    print(f" > NDT      : {ndt_weights_coverage:5.1f}%  (should be ≈ 95%)")

    # Print brief report to file, with simulation settings and results
    filepath = Path(__file__).parent / f"pymc_beta_weights_simulation_report_{repetitions}reps_{N}N.txt"
    with open(filepath, "w") as f:
        f.write(f"Simulation settings:\n")
        f.write(f" > Repetitions :  {repetitions}\n")
        f.write(f" > Sample size :  {N}\n")
        f.write(f" > True parameters:\n")
        [f.write(f"  * {str(p)}\n") for p in true_params_list]
        f.write(f"Results:\n")
        f.write(f" > Boundary coverage :  {boundary_weights_coverage:5.1f}%\n")
        f.write(f" > Drift coverage    :  {drift_weights_coverage:5.1f}%\n")
        f.write(f" > NDT coverage      :  {ndt_weights_coverage:5.1f}%\n")


"""
Main
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QND Drift Difference Inference")
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