#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
from typing import Tuple, List
from vendor.ezas.base import ez_equations as ez
from vendor.ezas.classes import Parameters
import unittest
import argparse

"""
Results class
"""
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

"""
Calculate error
"""
def calculate_error(true: Parameters, 
                    estimated: Parameters) -> Tuple[Parameters, Parameters]:
    """Calculate bias and squared error"""
    bias = Parameters(
        true.boundary() - estimated.boundary(),
        true.drift() - estimated.drift(),
        true.ndt() - estimated.ndt()
    )
    
    relative_bias = Parameters(
        (true.boundary() - estimated.boundary()) / true.boundary(),
        (true.drift() - estimated.drift()) / true.drift(),
        (true.ndt() - estimated.ndt()) / true.ndt()
    )
    
    squared_error = Parameters(
        bias.boundary()**2,
        bias.drift()**2,
        bias.ndt()**2
    )
    
    return bias, squared_error, relative_bias

"""
Mean parameters
"""
def mean_parameters(results: List[Results]) -> Results:
    """Calculate the mean parameters from a list of results"""
    return Results(
        Parameters(
            np.mean([result.true_params().boundary() for result in results]),
            np.mean([result.true_params().drift() for result in results]),
            np.mean([result.true_params().ndt() for result in results])
        ),
        Parameters(
            np.mean([result.est_params().boundary() for result in results]),
            np.mean([result.est_params().drift() for result in results]),
            np.mean([result.est_params().ndt() for result in results])
        ),
        Parameters(
            np.mean([result.bias().boundary() for result in results]),
            np.mean([result.bias().drift() for result in results]),
            np.mean([result.bias().ndt() for result in results])
        ),
        Parameters(
            np.mean([result.sq_error().boundary() for result in results]),
            np.mean([result.sq_error().drift() for result in results]),
            np.mean([result.sq_error().ndt() for result in results])
        ),
        Parameters(
            np.mean([result.relative_bias().boundary() for result in results]),
            np.mean([result.relative_bias().drift() for result in results]),
            np.mean([result.relative_bias().ndt() for result in results])
        )
    )

"""
Run simulation
"""
def run_simulation(N: int, 
                   lower_bound: Parameters, 
                   upper_bound: Parameters) -> Results:
    """Run a single simulation and return true, estimated, and bias parameters"""
    results = []
    for _ in range(10000):
        true_params = Parameters.random(lower_bound, upper_bound)
        pred_stats = ez.forward(true_params)
        obs_stats = [s.sample(N) for s in pred_stats]
        est_params = ez.inverse(obs_stats)
        bias, sq_error, relative_bias = calculate_error(true_params, est_params)
        results.append(Results(true_params, est_params, bias, sq_error, relative_bias))
    
    return mean_parameters(results)

"""
Test suite
"""
class TestSuite(unittest.TestCase):
    def test_results(self):
        """
        Test that the Results class works correctly.
        """
        results = Results(
            true_params=Parameters(1.0, 2.0, 3.0),
            est_params=Parameters(1.0, 2.0, 3.0),
            bias=Parameters(0.0, 0.0, 0.0),
            sq_error=Parameters(0.0, 0.0, 0.0),
            relative_bias=Parameters(0.0, 0.0, 0.0)
        )
        self.assertEqual(results.true_params(), Parameters(1.0, 2.0, 3.0))
        self.assertEqual(results.est_params(), Parameters(1.0, 2.0, 3.0))
        self.assertEqual(results.bias(), Parameters(0.0, 0.0, 0.0))
        self.assertEqual(results.sq_error(), Parameters(0.0, 0.0, 0.0))
        self.assertEqual(results.relative_bias(), Parameters(0.0, 0.0, 0.0))
        
    def test_calculate_error(self):
        """
        Test that the calculate_error function works correctly.
        """
        bias, sq_error, relative_bias = calculate_error(
            Parameters(1.0, 2.0, 3.0),
            Parameters(1.0, 2.0, 3.0)
        )
        self.assertEqual(bias, Parameters(0.0, 0.0, 0.0))
        self.assertEqual(sq_error, Parameters(0.0, 0.0, 0.0))
        self.assertEqual(relative_bias, Parameters(0.0, 0.0, 0.0))
        
    def test_mean_parameters(self):
        """
        Test that the mean_parameters function works correctly.
        """
        results = [
            Results(
                true_params=Parameters(1.0, 2.0, 3.0),
                est_params=Parameters(1.0, 2.0, 3.0),
                bias=Parameters(0.0, 0.0, 0.0),
                sq_error=Parameters(0.0, 0.0, 0.0),
                relative_bias=Parameters(0.0, 0.0, 0.0)
            ),
            Results(
                true_params=Parameters(2.0, 3.0, 4.0),
                est_params=Parameters(2.0, 3.0, 4.0),
                bias=Parameters(0.0, 0.0, 0.0),
                sq_error=Parameters(0.0, 0.0, 0.0),
                relative_bias=Parameters(0.0, 0.0, 0.0)
            )
        ]
        mean_results = mean_parameters(results) 
        self.assertEqual(mean_results.true_params(), Parameters(1.5, 2.5, 3.5))
        self.assertEqual(mean_results.est_params(), Parameters(1.5, 2.5, 3.5))
        self.assertEqual(mean_results.bias(), Parameters(0.0, 0.0, 0.0))
        self.assertEqual(mean_results.sq_error(), Parameters(0.0, 0.0, 0.0))
        self.assertEqual(mean_results.relative_bias(), Parameters(0.0, 0.0, 0.0))

"""
Demo
"""
def demo():
    print("Demo for simulation:")
    print("No demo available")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--test", action="store_true", help="Run the test suite")
    parser.add_argument("--demo", action="store_true", help="Run the demo")
    
    args = parser.parse_args()  
    
    if args.test:
        unittest.main(argv=[__file__], verbosity=0, failfast=True)

    if args.demo:
        demo()
    