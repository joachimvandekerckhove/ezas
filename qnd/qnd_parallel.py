#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
from typing import Tuple, List
from vendor.ezas.base import ez_equations as ez
from vendor.ezas.classes.moments import Observations
from vendor.ezas.classes.parameters import Parameters
from vendor.ezas.classes.design_matrix import DesignMatrix, BetaWeights
import unittest
import time
import argparse
from tqdm import tqdm
import copy
import multiprocessing as mp

_LOWER_QUANTILE = 0.025
_UPPER_QUANTILE = 0.975

"""
Quick and dirty parameter estimation of beta weights
"""
def qnd_beta_weights_estimation(
    observations: list[Observations],
    design_matrix: DesignMatrix
) -> Tuple[BetaWeights, Parameters]:
    """Estimate EZ-diffusion parameters with uncertainty given observed statistics and design matrices.
    
    Args:
        observations: Observed summary statistics (Observations)
        design_matrix: Design matrix for parameter estimation (n_conditions x n_parameters)
    Returns:
        - The updated design matrix with the beta weights
    """
    
    if not isinstance(observations, list):
        raise TypeError("observations must be a list of Observations")
    
    if not isinstance(design_matrix, DesignMatrix):
        raise TypeError("design_matrix must be a DesignMatrix")
    
    if len(observations) != design_matrix.boundary_design().shape[0]:
        raise ValueError("The number of observations must match the number of rows in the design matrix")
    
    # Deep copy the design matrix
    local_matrix = copy.deepcopy(design_matrix)

    # Get the parameters
    parameters = [ ez.inverse(o.to_moments()) for o in observations ]

    # Inject the observed summary statistics into the design matrix
    local_matrix.set_parameters(parameters)
    
    # Estimate the parameters
    local_matrix.fix()
    
    # List of beta weights
    beta_weights_list = []
    parameters_list = []
    
    # Bootstrap the distribution of the parameters
    for _ in range(1000):
        parameters = [ ez.inverse(o.resample().to_moments()) for o in observations ]
        # print(f"## Resampled parameters ({_}):")
        # [print(p) for p in parameters]
        local_matrix.set_parameters(parameters)
        # print(f"## Local matrix parameters ({_}):")
        # [print(p) for p in local_matrix.get_parameters()]
        parameters_list.append(local_matrix.get_parameters())
        beta_weights_list.append(local_matrix.get_beta_weights())
        # print(f"## Beta weights ({_}):")
        # print(local_matrix.get_beta_weights())
        
    
    # Get the mean of the beta weights
    return BetaWeights.summarize_matrix(beta_weights_list), Parameters.summarize_matrix(parameters_list)
    

def demo():
    """
    Demonstrate basic QND parameter estimation with beta weights.
    """
    
    design_matrix = DesignMatrix(
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
    
    sample_sizes = [1000, 2000, 3000, 4000]
    
    observations = design_matrix.sample(sample_sizes)
    
    # Get the true beta weights
    true_parameters = design_matrix.get_parameters()
    true_beta_weights = design_matrix.get_beta_weights()
    
    # Create a working design matrix
    working_matrix = DesignMatrix(
        boundary_design = design_matrix.boundary_design(),
        drift_design    = design_matrix.drift_design(),
        ndt_design      = design_matrix.ndt_design()
    )    
    
    # Estimate the parameters
    start_time = time.time()
    estimated_beta_weights, estimated_parameters = \
        qnd_beta_weights_estimation(observations, working_matrix)
    end_time = time.time()
    print(f"## Time taken: {1000 * (end_time - start_time):.0f} ms")
    
    # Print the results for beta weights
    print(f"## True beta weights:\n\n{true_beta_weights}\n")
    
    print(f"## Estimated beta weights:\n\n{estimated_beta_weights}\n")
    
    # Print the results
    # print(f"## True parameters:\n")
    # [print(p) for p in true_parameters]
    # print(f"\n## Estimated parameters:\n")
    # [print(p) for p in estimated_parameters]
    
    # Check if the true parameters are within the 95% credible interval of the estimated parameters
    misses = 0
    for i in range(len(true_parameters)):
        true_param = true_parameters[i]
        est_param = estimated_parameters[i]
        
        # Check if true parameter is within the 95% credible interval
        boundary_in_bounds = (est_param.boundary_lower_bound() <= true_param.boundary() <= est_param.boundary_upper_bound())
        drift_in_bounds = (est_param.drift_lower_bound() <= true_param.drift() <= est_param.drift_upper_bound())
        ndt_in_bounds = (est_param.ndt_lower_bound() <= true_param.ndt() <= est_param.ndt_upper_bound())
        
        if not boundary_in_bounds:
            misses += 1
            print(f"❌", end="")
        else:
            print(f"✅", end="")

        if not drift_in_bounds:
            misses += 1
            print(f"❌", end="")
        else:
            print(f"✅", end="")

        if not ndt_in_bounds:
            misses += 1
            print(f"❌", end="")
        else:
            print(f"✅", end="")

    print(f"\n")

    if misses > 0: # print cross or check mark emoji
        print(f"## Misses: {misses} out of {3*len(true_parameters)} ❌\n")
    else:
        print(f"## Misses: {misses} out of {3*len(true_parameters)} ✅\n")
        

def _single_simulation_iteration(args):
    """Single simulation iteration for parallel processing."""
    iteration_num, design_matrix, working_matrix, sample_sizes, true_beta_weights = args
    observations = design_matrix.sample(sample_sizes)
    estimated_beta_weights, _ = qnd_beta_weights_estimation(observations, working_matrix)
    
    n_positions = len(true_beta_weights.beta_boundary_mean())
    results = []
    
    for i in range(n_positions):
        true_boundary = true_beta_weights.beta_boundary_mean()[i]
        true_drift = true_beta_weights.beta_drift_mean()[i]
        true_ndt = true_beta_weights.beta_ndt_mean()[i]
        
        # Check if true beta weight is within the 95% credible interval
        boundary_covered = (estimated_beta_weights.beta_boundary_lower()[i] <= true_boundary <= estimated_beta_weights.beta_boundary_upper()[i])
        drift_covered = (estimated_beta_weights.beta_drift_lower()[i] <= true_drift <= estimated_beta_weights.beta_drift_upper()[i])
        ndt_covered = (estimated_beta_weights.beta_ndt_lower()[i] <= true_ndt <= estimated_beta_weights.beta_ndt_upper()[i])
        
        results.append((boundary_covered, drift_covered, ndt_covered))
    
    all_covered = all(all(pos) for pos in results)
    return results, all_covered

def simulation_parallel(simulation_repetitions: int = 1000):
    """Parallel version of the simulation using multiprocessing."""
    design_matrix = DesignMatrix(
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
    
    n = 16
    sample_sizes = [n,n,n,n]
    
    # Get the true beta weights
    true_parameters = design_matrix.get_parameters()
    true_beta_weights = design_matrix.get_beta_weights()
    
    # Create a working design matrix
    working_matrix = DesignMatrix(
        boundary_design = design_matrix.boundary_design(),
        drift_design    = design_matrix.drift_design(),
        ndt_design      = design_matrix.ndt_design()
    )    

    # Calculate number of processes to use (75% of available cores)
    n_cores = mp.cpu_count()
    n_processes = max(1, int(0.75 * n_cores))
    print(f"Using {n_processes} processes out of {n_cores} available cores")
    
    # Initialize counters
    n_positions = len(true_beta_weights.beta_boundary_mean())
    boundary_coverage = np.zeros(n_positions)
    drift_coverage = np.zeros(n_positions)
    ndt_coverage = np.zeros(n_positions)
    total_coverage = 0
    
    # Run parallel simulation
    with mp.Pool(processes=n_processes) as pool:
        # Create arguments for each iteration
        args_list = [(i, design_matrix, working_matrix, sample_sizes, true_beta_weights) 
                    for i in range(simulation_repetitions)]
        
        # Use imap for progress tracking
        results = list(tqdm(
            pool.imap(_single_simulation_iteration, args_list),
            total=simulation_repetitions,
            desc=f"Parallel simulation ({n_processes} cores)"
        ))
    
    # Aggregate results
    for iteration_results, all_covered in results:
        for i, (boundary_covered, drift_covered, ndt_covered) in enumerate(iteration_results):
            if boundary_covered: boundary_coverage[i] += 1
            if drift_covered: drift_coverage[i] += 1
            if ndt_covered: ndt_coverage[i] += 1
        
        if all_covered: total_coverage += 1
    
    # Close the progress bar
    print("Coverage by beta weight position:")
    for p, c in zip(['Boundary', 'Drift', 'NDT'], [boundary_coverage, drift_coverage, ndt_coverage]):
        print(f" > {p}:")
        for i in range(n_positions):
            print(f"   - Position {i}: {100 * c[i] / simulation_repetitions:7.1f}%  (should be ≈ 95%)")
    
    print(f"\nOverall coverage (all parameters in all positions):")
    print(f" > Total   : {100 * total_coverage / simulation_repetitions:7.1f}%  (should be ≈ 63%)") # .95^9


        

"""
Simulation study for QND parameter estimation with beta weights
"""
def simulation(simulation_repetitions: int = 1000):
    design_matrix = DesignMatrix(
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
    
    n = 16
    sample_sizes = [n,n,n,n]
    
    # Get the true beta weights
    true_parameters = design_matrix.get_parameters()
    true_beta_weights = design_matrix.get_beta_weights()
    
    # Create a working design matrix
    working_matrix = DesignMatrix(
        boundary_design = design_matrix.boundary_design(),
        drift_design    = design_matrix.drift_design(),
        ndt_design      = design_matrix.ndt_design()
    )    

    # Initialize a progress bar
    progress_bar = tqdm(total=simulation_repetitions, desc="Simulation progress")

    # Counters for each beta weight position
    n_positions = len(true_beta_weights.beta_boundary_mean())
    boundary_coverage = np.zeros(n_positions)
    drift_coverage = np.zeros(n_positions)
    ndt_coverage = np.zeros(n_positions)
    total_coverage = 0
        
    for _ in range(simulation_repetitions):
        
        observations = design_matrix.sample(sample_sizes)
        
        estimated_beta_weights, _ = \
            qnd_beta_weights_estimation(observations, working_matrix)
            
        # Check coverage for each beta weight position
        all_covered = True
        for i in range(n_positions):
            true_boundary = true_beta_weights.beta_boundary_mean()[i]
            true_drift = true_beta_weights.beta_drift_mean()[i]
            true_ndt = true_beta_weights.beta_ndt_mean()[i]
            
            # Check if true beta weight is within the 95% credible interval
            boundary_covered = (estimated_beta_weights.beta_boundary_lower()[i] <= true_boundary <= estimated_beta_weights.beta_boundary_upper()[i])
            drift_covered = (estimated_beta_weights.beta_drift_lower()[i] <= true_drift <= estimated_beta_weights.beta_drift_upper()[i])
            ndt_covered = (estimated_beta_weights.beta_ndt_lower()[i] <= true_ndt <= estimated_beta_weights.beta_ndt_upper()[i])
        
            if boundary_covered: boundary_coverage[i] += 1
            if drift_covered: drift_coverage[i] += 1
            if ndt_covered: ndt_coverage[i] += 1
            
            if not (boundary_covered and drift_covered and ndt_covered):
                all_covered = False
        
        if all_covered: total_coverage += 1
        
        # Update the progress bar
        progress_bar.update(1)
        
    # Close the progress bar
    progress_bar.close()
            
    print("Coverage by beta weight position:")
    for p, c in zip(['Boundary', 'Drift', 'NDT'], [boundary_coverage, drift_coverage, ndt_coverage]):
        print(f" > {p}:")
        for i in range(n_positions):
            print(f"   - Position {i}: {100 * c[i] / simulation_repetitions:7.1f}%  (should be ≈ 95%)")
    
    print(f"\nOverall coverage (all parameters in all positions):")
    print(f" > Total   : {100 * total_coverage / simulation_repetitions:7.1f}%  (should be ≈ 63%)") # .95^9



"""
Test suite
"""
class TestSuite(unittest.TestCase):
    def test(self):
        pass


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()  
    
    parser.add_argument("--test", action="store_true", help="Run the test suite")
    parser.add_argument("--demo", action="store_true", help="Run the demo")
    parser.add_argument("--simulation", action="store_true", help="Run the serial simulation")
    parser.add_argument("--parallel", action="store_true", help="Run the parallel simulation")
    parser.add_argument("--repetitions", type=int, default=1000, help="Number of simulation repetitions (default: 1000)")
    
    args = parser.parse_args()
    
    if args.test:
        unittest.main(argv=[__file__], verbosity=0, failfast=True)
        
    if args.demo:
        demo()
        
    if args.simulation:
        simulation(args.repetitions)
        
    if args.parallel:
        simulation_parallel(args.repetitions)