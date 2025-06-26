#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
from typing import Tuple, List
from vendor.ezas.base import ez_equations as ez
from vendor.ezas.classes.moments import Observations
from vendor.ezas.classes.parameters import Parameters
import unittest
import time
import argparse
from tqdm import tqdm
import vendor.ezas.utils.prettify as pretty

_DEMO_DEFAULT_PARAMETERS = Parameters(1.0, 0.5, 0.2)
_DEMO_DEFAULT_SAMPLE_SIZE = 10

"""
Quick and dirty single parameter estimation
"""
def qnd_single_estimation(
    observations: Observations, 
    n_repetitions: int = 1000
) -> Parameters:
    """Estimate EZ-diffusion parameters with uncertainty given observed statistics.
    
    Args:
        observations: Observed summary statistics (Observations)
        n_repetitions: Number of repetitions
    Returns:
        - Parameters: estimated parameters (as a Parameters object)
    """
    if not isinstance(observations, Observations):
        raise TypeError("observations must be of type Observations")
        
    # Initialize the parameter matrix (K x 1 list of Parameters)
    parameter_matrix = [
        ez.inverse(observations.resample().to_moments())
        for _ in range(n_repetitions)
    ]
    
    # Get descriptive statistics of distribution of Parameters using Parameters.summarize_list
    return Parameters.summarize(parameter_matrix)


"""
Demonstrate basic QND parameter estimation
"""
def demo(true_parameters: Parameters|None = None, sample_size: int|None = None):
    """Demonstrate basic QND parameter estimation.
    
    Args:
        true_parameters: True parameters (Parameters)
        sample_size: Number of trials used to generate the statistics (int)
    """

    if true_parameters is None:
        true_parameters = _DEMO_DEFAULT_PARAMETERS
    if sample_size is None:
        sample_size = _DEMO_DEFAULT_SAMPLE_SIZE
    
    if not isinstance(true_parameters, Parameters):
        raise TypeError("true_parameters must be of type Parameters")
    if not isinstance(sample_size, int):
        raise TypeError("sample_size must be of type int")
    
        
    # Generate some random parameters
    moments = ez.forward(true_parameters)
    observations = moments.sample(sample_size)
    
    # Estimate the parameters and time the operation
    start_time = time.time()
    estimated_parameters = qnd_single_estimation(observations, 
                                                 n_repetitions=1000)
    end_time = time.time()
    
    evaluation = true_parameters.is_within_bounds_of(estimated_parameters)
    
    # Write report to console
    report = f"""#### Sample size: {sample_size}
     > Moments               : {moments}
     > Observations          : {observations}
     > True parameters       : {true_parameters}
     > QND parameters        : {estimated_parameters}
     > Evaluation            : {pretty.b(evaluation)}
     > Time taken            : {(end_time*1000 - start_time*1000):.0f} ms
    """
    print(report)


"""
Simulation study for basic QND parameter estimation
"""
def simulation(simulation_repetitions: int|None = None):
    """Simulate the basic QND parameter estimation.
    
    Args:
        simulation_repetitions: Number of repetitions
    """
    
    if simulation_repetitions is None:
        simulation_repetitions = 1000

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
        observations = moments.sample(sample_size)
        estimate = qnd_single_estimation(observations, n_repetitions=1000)
        
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
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
    def tearDown(self):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        
    def test_demo_execution_11(self):
        demo(true_parameters=_DEMO_DEFAULT_PARAMETERS, 
             sample_size=_DEMO_DEFAULT_SAMPLE_SIZE)
        self.assertIn(f"Sample size: {_DEMO_DEFAULT_SAMPLE_SIZE}", 
                      sys.stdout.getvalue())
        
    def test_demo_execution_10(self):
        demo(true_parameters=_DEMO_DEFAULT_PARAMETERS)
        self.assertIn(f"Sample size: {_DEMO_DEFAULT_SAMPLE_SIZE}", 
                      sys.stdout.getvalue())
        demo(true_parameters=_DEMO_DEFAULT_PARAMETERS, 
             sample_size=None)
        self.assertIn(f"Sample size: {_DEMO_DEFAULT_SAMPLE_SIZE}", 
                      sys.stdout.getvalue())
    
    def test_demo_execution_01(self):
        demo(true_parameters=None, 
             sample_size=_DEMO_DEFAULT_SAMPLE_SIZE)
        self.assertIn(f"Sample size: {_DEMO_DEFAULT_SAMPLE_SIZE}", 
                      sys.stdout.getvalue())
        demo(sample_size=_DEMO_DEFAULT_SAMPLE_SIZE)
        self.assertIn(f"Sample size: {_DEMO_DEFAULT_SAMPLE_SIZE}", 
                      sys.stdout.getvalue())
        
    def test_demo_execution_00(self):
        demo()
        self.assertIn(f"Sample size: {_DEMO_DEFAULT_SAMPLE_SIZE}", 
                      sys.stdout.getvalue())
        demo(true_parameters=None, 
             sample_size=None)
        self.assertIn(f"Sample size: {_DEMO_DEFAULT_SAMPLE_SIZE}", 
                      sys.stdout.getvalue())
        
    def test_demo_input_checks_1(self):
        with self.assertRaises(TypeError): 
            demo(true_parameters=None, 
                 sample_size="test")
        with self.assertRaises(TypeError): 
            demo(true_parameters=None, 
                 sample_size=_DEMO_DEFAULT_PARAMETERS)
        
    def test_demo_input_checks_2(self):
        with self.assertRaises(TypeError): 
            demo(true_parameters=_DEMO_DEFAULT_PARAMETERS, 
                 sample_size="test")
        with self.assertRaises(TypeError): 
            demo(true_parameters=_DEMO_DEFAULT_PARAMETERS, 
                 sample_size=_DEMO_DEFAULT_PARAMETERS)
        
    def test_demo_input_checks_3(self):
        with self.assertRaises(TypeError): 
            demo(true_parameters="test", 
                 sample_size=None)
        with self.assertRaises(TypeError): 
            demo(true_parameters="test", 
                 sample_size=_DEMO_DEFAULT_SAMPLE_SIZE)
        with self.assertRaises(TypeError): 
            demo(true_parameters="test", 
                 sample_size="test")
        with self.assertRaises(TypeError): 
            demo(true_parameters="test", 
                 sample_size=_DEMO_DEFAULT_PARAMETERS)
        
    def test_demo_input_checks_4(self):
        with self.assertRaises(TypeError): 
            demo(true_parameters=_DEMO_DEFAULT_SAMPLE_SIZE, 
                 sample_size=_DEMO_DEFAULT_SAMPLE_SIZE)
        with self.assertRaises(TypeError): 
            demo(true_parameters=_DEMO_DEFAULT_SAMPLE_SIZE, 
                 sample_size=None)
        with self.assertRaises(TypeError): 
            demo(true_parameters=_DEMO_DEFAULT_SAMPLE_SIZE, 
                 sample_size="10")
        with self.assertRaises(TypeError): 
            demo(true_parameters=_DEMO_DEFAULT_SAMPLE_SIZE, 
                 sample_size=_DEMO_DEFAULT_PARAMETERS)

    def test_simulation_execution_1(self):
        simulation(simulation_repetitions=10)
        self.assertIn("Coverage:", sys.stdout.getvalue())
        
    def test_simulation_input_checks(self):
        with self.assertRaises(TypeError): 
            simulation(simulation_repetitions="10")
        with self.assertRaises(TypeError):
            simulation(simulation_repetitions=0.50)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()  
    
    parser.add_argument("--test", 
                        action="store_true", 
                        help="Run the test suite")
    parser.add_argument("--demo", 
                        action="store_true", 
                        help="Run the demo")
    parser.add_argument("--simulation", 
                        action="store_true", 
                        help="Run the simulation")
    
    args = parser.parse_args()
    
    if args.test:
        unittest.main(argv=[__file__], verbosity=0, failfast=True)
        
    if args.demo:
        demo()
        
    if args.simulation:
        simulation()

