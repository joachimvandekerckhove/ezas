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
_DEFAULT_N_BOOTSTRAP = 1000

"""
"Quick and dirty" parameter estimation of beta weights
"""
def qnd_beta_weights_estimation(
    observations: list[Observations],
    working_matrix: DesignMatrix,
    n_bootstrap: int = _DEFAULT_N_BOOTSTRAP
) -> Tuple[BetaWeights, Parameters]:
    """Estimate EZ-diffusion parameters with uncertainty given observed statistics and design matrices.
    
    This is a "quick and dirty" implementation of the parameter estimation.
    We use a bootstrap approach to estimate the uncertainty of the beta weights.
    
    Args:
        observations: Observed summary statistics (Observations)
        working_matrix: Design matrix for parameter estimation (n_conditions x n_parameters)
          ! working_matrix is modified in place !
    Returns:
        - The updated design matrix with the beta weights
    """
    
    if not isinstance(observations, list):
        raise TypeError("observations must be a list of Observations")
    
    if not isinstance(working_matrix, DesignMatrix):
        raise TypeError("working_matrix must be a DesignMatrix")
    
    if len(observations) != working_matrix.boundary_design().shape[0]:
        raise ValueError("The number of observations must match the number of rows in the design matrix")
    
    # Get the parameters
    parameters = [ ez.inverse(o.to_moments()) for o in observations ]

    # Inject the observed summary statistics into the design matrix
    working_matrix.set_parameters(parameters)
    
    # Estimate the parameters
    working_matrix.fix()
    
    # List of beta weights
    beta_weights_list = []
    parameters_list = []
    
    # Bootstrap the distribution of the parameters
    for _ in range(n_bootstrap):
        parameters = [
            ez.inverse(o.resample().to_moments())
            for o in observations
        ]
        working_matrix.set_parameters(parameters)
        parameters_list.append(working_matrix.get_parameters())
        beta_weights_list.append(working_matrix.get_beta_weights())
    
    # Get the mean of the beta weights
    return BetaWeights.summarize_matrix(beta_weights_list), \
        Parameters.summarize_matrix(parameters_list), \
        beta_weights_list
    

def demo(design_matrix: DesignMatrix|None = None,
         sample_sizes: list[int]|None = None):
    """
    Demonstrate basic QND parameter estimation with beta weights.
    """
    
    if design_matrix is None:
        design_matrix = _EXAMPLE_DESIGN_MATRIX
    
    if sample_sizes is None:
        sample_sizes = [100, 200, 300, 400]
    
    if not isinstance(design_matrix, DesignMatrix):
        raise TypeError("design_matrix must be a DesignMatrix")
    
    if not isinstance(sample_sizes, list) or \
        not all(isinstance(s, int) for s in sample_sizes):
        raise TypeError("sample_sizes must be a list of int")
    
    if len(sample_sizes) != design_matrix.boundary_design().shape[0]:
        raise ValueError("sample_sizes must have the same number of elements \
            as the number of rows in the design matrix")
    
    if not all(s > 0 for s in sample_sizes):
        raise ValueError("sample_sizes must be a list of positive integers")
    
    # Get the true beta weights
    true_parameters = design_matrix.get_parameters()
    true_beta_weights = design_matrix.get_beta_weights()

    print(f"\n## True beta weights:\n")
    print(true_beta_weights)
    
    print(f"\n## True parameters:\n")
    [print(p) for p in true_parameters]

    print(f"\n## Simulating data...", end="")
    start_time = time.time()
    observations = design_matrix.sample(sample_sizes)
    end_time = time.time()
    print(f" ({1000 * (end_time - start_time):.0f} ms)")
    
    print(f"\n## Observations ({len(observations)} conditions):\n")
    [print(o) for o in observations]
    
    # Create a working design matrix
    working_matrix = DesignMatrix(
        boundary_design = design_matrix.boundary_design(),
        drift_design    = design_matrix.drift_design(),
        ndt_design      = design_matrix.ndt_design()
    )    
    
    # Estimate the parameters
    print(f"\n## Estimating parameters...", end="")
    start_time = time.time()
    estimated_beta_weights, estimated_parameters, _ = \
        qnd_beta_weights_estimation(observations, working_matrix)
    end_time = time.time()
    print(f" ({1000 * (end_time - start_time):.0f} ms)")
        
    print(f"\n## Checking parameter coverage... ", end="")
    # Check if the true parameters are within the 95% credible interval of the estimated parameters
    start_time = time.time()
    misses = 0
    for i in range(len(true_parameters)):
        true_param = true_parameters[i]
        est_param = estimated_parameters[i]
        
        # Check if true parameter is within the 95% credible interval
        boundary_in_bounds = (est_param.boundary_lower_bound() <= \
            true_param.boundary() <= est_param.boundary_upper_bound())
        drift_in_bounds = (est_param.drift_lower_bound() <= \
            true_param.drift() <= est_param.drift_upper_bound())
        ndt_in_bounds = (est_param.ndt_lower_bound() <= \
            true_param.ndt() <= est_param.ndt_upper_bound())
        
        misses += int(not boundary_in_bounds)
        misses += int(not drift_in_bounds)
        misses += int(not ndt_in_bounds)
        
        print(f"✅" if boundary_in_bounds else f"❌", end="")
        print(f"✅" if drift_in_bounds    else f"❌", end="")
        print(f"✅" if ndt_in_bounds      else f"❌", end="")
    end_time = time.time()
    print(f" ({1000 * (end_time - start_time):.0f} ms)")
    
    # Print the results for beta weights
    print(f"\n## Estimated beta weights:\n\n{estimated_beta_weights}")

    print(f"\n## Misses: {misses} out of {3*len(true_parameters)} " + \
        ("❌" if misses > 0 else "✅") + "\n")
        

"""
Simulation study for QND parameter estimation with beta weights
"""
def simulation(simulation_repetitions: int|None = None):
    
    if not isinstance(simulation_repetitions, int) \
        and simulation_repetitions is not None:
        raise TypeError("simulation_repetitions must be of type int")
    
    if simulation_repetitions <= 0:
        raise ValueError("simulation_repetitions must be greater than 0")
    
    if simulation_repetitions is None:
        simulation_repetitions = 1000
    
    design_matrix = _EXAMPLE_DESIGN_MATRIX
    
    n = 160
    sample_sizes = [n,n,n,n]
    
    # Get the true beta weights
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
        
        estimated_beta_weights, _, _ = \
            qnd_beta_weights_estimation(observations, working_matrix)
            
        # Check coverage for each beta weight position
        all_covered = True
        for i in range(n_positions):
            true_boundary = true_beta_weights.beta_boundary_mean()[i]
            true_drift = true_beta_weights.beta_drift_mean()[i]
            true_ndt = true_beta_weights.beta_ndt_mean()[i]
            
            # Check if true beta weight is within the 95% credible interval
            boundary_covered = (estimated_beta_weights.beta_boundary_lower()[i] \
                <= true_boundary <= estimated_beta_weights.beta_boundary_upper()[i])
            drift_covered = (estimated_beta_weights.beta_drift_lower()[i] \
                <= true_drift <= estimated_beta_weights.beta_drift_upper()[i])
            ndt_covered = (estimated_beta_weights.beta_ndt_lower()[i] \
                <= true_ndt <= estimated_beta_weights.beta_ndt_upper()[i])
        
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
    
    f = 100 / simulation_repetitions
            
    print("Coverage by beta weight position:")
    for p, c in zip(['Boundary', 'Drift', 'NDT'], 
                    [boundary_coverage, drift_coverage, ndt_coverage]):
        print(f" > {p}:")
        for i in range(n_positions):
            print(f"   - Position {i}: {f * c[i]:7.1f}%  (should be ≈ 95%)")
    
    print(f"\nOverall coverage (all parameters in all positions):")
    print(f" > Total   : {f * total_coverage:7.1f}%  (should be ≈ 63%)") # .95^9



"""
Parallel version of the simulation
"""

def _single_simulation_iteration(args):
    """Single simulation iteration for parallel processing."""
    _, design_matrix, working_matrix, sample_sizes, true_beta_weights = args
    observations = design_matrix.sample(sample_sizes)
    estimated_beta_weights, _, _ = \
        qnd_beta_weights_estimation(observations, working_matrix)
    
    n_positions = len(true_beta_weights.beta_boundary_mean())
    results = []
    
    for i in range(n_positions):
        true_boundary = true_beta_weights.beta_boundary_mean()[i]
        true_drift = true_beta_weights.beta_drift_mean()[i]
        true_ndt = true_beta_weights.beta_ndt_mean()[i]
        
        # Check if true beta weight is within the 95% credible interval
        boundary_covered = (estimated_beta_weights.beta_boundary_lower()[i] \
            <= true_boundary <= estimated_beta_weights.beta_boundary_upper()[i])
        drift_covered = (estimated_beta_weights.beta_drift_lower()[i] \
            <= true_drift <= estimated_beta_weights.beta_drift_upper()[i])
        ndt_covered = (estimated_beta_weights.beta_ndt_lower()[i] \
            <= true_ndt <= estimated_beta_weights.beta_ndt_upper()[i])
        
        results.append((boundary_covered, drift_covered, ndt_covered))
    
    all_covered = all(all(pos) for pos in results)
    return results, all_covered

def simulation_parallel(simulation_repetitions: int|None = None):
    """Parallel version of the simulation using multiprocessing."""
    
    if not isinstance(simulation_repetitions, int) \
        and simulation_repetitions is not None:
        raise TypeError("simulation_repetitions must be of type int")
    
    if simulation_repetitions <= 0:
        raise ValueError("simulation_repetitions must be greater than 0")
    
    if simulation_repetitions is None:
        simulation_repetitions = 1000
    
    design_matrix = _EXAMPLE_DESIGN_MATRIX
    
    n = 128
    sample_sizes = [n,n,n,n]
    
    # Get the true beta weights
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
        args_list = [(i, design_matrix, working_matrix, 
                      sample_sizes, true_beta_weights) 
                    for i in range(simulation_repetitions)]

        # Use imap for progress tracking
        results = list(tqdm(
            pool.imap(_single_simulation_iteration, args_list),
            total=simulation_repetitions,
            desc=f"Parallel simulation ({n_processes} cores)"
        ))
    
    # Aggregate results
    for iteration_results, all_covered in results:
        for i, (boundary_covered, drift_covered, ndt_covered) in \
            enumerate(iteration_results):
            if boundary_covered: boundary_coverage[i] += 1
            if drift_covered: drift_coverage[i] += 1
            if ndt_covered: ndt_coverage[i] += 1
        
        if all_covered: total_coverage += 1
    
    f = 100 / simulation_repetitions
    # Close the progress bar
    print("Coverage by beta weight position:")
    for p, c in zip(['Boundary', 'Drift', 'NDT'], 
                    [boundary_coverage, drift_coverage, ndt_coverage]):
        print(f" > {p}:")
        for i in range(n_positions):
            print(f"   - Position {i}: {f * c[i]:7.1f}%  (should be ≈ 95%)")
    
    print(f"\nOverall coverage (all parameters in all positions):")
    print(f" > Total   : {f * total_coverage:7.1f}%  (should be ≈ 63%)") # .95^9


"""
Test suite
"""
class TestSuite(unittest.TestCase):
    def setUp(self):
        import io
        self.stdout = sys.stdout
        sys.stdout = io.StringIO()
        os.environ['TQDM_DISABLE'] = '1'

    def tearDown(self):
        sys.stdout = self.stdout
        os.environ['TQDM_DISABLE'] = '0'
    
    def test_demo_execution_1(self):
        demo()
        self.assertIn("## True beta weights:", sys.stdout.getvalue())
        self.assertIn("## True parameters:", sys.stdout.getvalue())
        self.assertIn("## Simulating data...", sys.stdout.getvalue())
        self.assertIn("## Observations (4 conditions):", sys.stdout.getvalue())
        self.assertIn("## Estimating parameters...", sys.stdout.getvalue())
        self.assertIn("## Checking parameter coverage...", sys.stdout.getvalue())
    
    def test_demo_execution_2(self):
        demo(design_matrix=_EXAMPLE_DESIGN_MATRIX,
             sample_sizes=[100, 200, 300, 400])
        self.assertIn("## True beta weights:", sys.stdout.getvalue())
        self.assertIn("## True parameters:", sys.stdout.getvalue())
        self.assertIn("## Simulating data...", sys.stdout.getvalue())
        self.assertIn("## Observations (4 conditions):", sys.stdout.getvalue())
        self.assertIn("## Estimating parameters...", sys.stdout.getvalue())
        self.assertIn("## Checking parameter coverage...", sys.stdout.getvalue())
    
    def test_demo_input_checks_1(self):
        with self.assertRaises(TypeError):
            demo(design_matrix="test")
        with self.assertRaises(TypeError):
            demo(sample_sizes="test")
        with self.assertRaises(ValueError):
            demo(design_matrix=_EXAMPLE_DESIGN_MATRIX,
                 sample_sizes=[100, 200, 300])
        with self.assertRaises(ValueError):
            demo(design_matrix=_EXAMPLE_DESIGN_MATRIX,
                 sample_sizes=[100, 200, 300, 400, 500])
        with self.assertRaises(TypeError):
            demo(design_matrix=_EXAMPLE_DESIGN_MATRIX,
                 sample_sizes=[100, 200, 300, 40.9])
        with self.assertRaises(ValueError):
            demo(design_matrix=_EXAMPLE_DESIGN_MATRIX,
                 sample_sizes=[100, 200, 300, 400, -500])
    
    def test_simulation_input_checks_1(self):
        with self.assertRaises(ValueError):
            simulation(simulation_repetitions=-1)
        with self.assertRaises(TypeError):
            simulation(simulation_repetitions=0.5)
        with self.assertRaises(ValueError):
            simulation(simulation_repetitions=0)
        with self.assertRaises(TypeError):
            simulation(simulation_repetitions="test")
    
    def test_simulation_parallel_input_checks(self):
        with self.assertRaises(ValueError):
            simulation_parallel(simulation_repetitions=-1)
        with self.assertRaises(TypeError):
            simulation_parallel(simulation_repetitions=0.5)
        with self.assertRaises(ValueError):
            simulation_parallel(simulation_repetitions=0)
        with self.assertRaises(TypeError):
            simulation_parallel(simulation_repetitions="test")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()  
    
    parser.add_argument("--test", action="store_true", 
                        help="Run the test suite")
    parser.add_argument("--demo", action="store_true", 
                        help="Run the demo")
    parser.add_argument("--simulation", action="store_true", 
                        help="Run the simulation")
    parser.add_argument("--parallel", action="store_true", 
                        help="Run the simulation in parallel")
    parser.add_argument("--repetitions", type=int, default=1000, 
                        help="Number of simulation repetitions (default: 1000)")
    
    args = parser.parse_args()
    
    if args.test:
        unittest.main(argv=[__file__], verbosity=0, failfast=True)
        
    if args.demo:
        demo()
        
    if args.simulation:
        if args.parallel:
            simulation_parallel(args.repetitions)
        else:
            simulation(args.repetitions)
    
    
        
