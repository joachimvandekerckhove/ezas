#!/usr/bin/env python3

import numpy as np
import argparse
import unittest

from base import (Parameters, SummaryStats, 
                forward_equations, resample_summary_stats, inverse_equations)

"""
Design matrix class
"""
class DesignMatrix:
    def __init__(self,
                 boundary_design: np.ndarray, 
                 drift_design: np.ndarray, 
                 ndt_design: np.ndarray,
                 boundary_weights: np.ndarray = None,
                 drift_weights: np.ndarray = None,
                 ndt_weights: np.ndarray = None,
                 boundary: np.ndarray = None,
                 drift: np.ndarray = None,
                 ndt: np.ndarray = None):
        self._boundary_design  = boundary_design
        self._drift_design     = drift_design
        self._ndt_design       = ndt_design
        self._boundary_weights = boundary_weights
        self._drift_weights    = drift_weights
        self._ndt_weights      = ndt_weights
        self._boundary   = boundary
        self._drift      = drift
        self._ndt        = ndt
        
        self._is_fixed = False

        self.fix()
    
    def fix(self):
        if self._is_fixed:
            return
        
        if self._boundary is None and self._boundary_weights is not None:
            # print("Recomputing boundary")
            self._boundary = self._boundary_design @ self._boundary_weights

        if self._drift is None and self._drift_weights is not None:
            # print("Recomputing drift")
            self._drift = self._drift_design @ self._drift_weights

        if self._ndt is None and self._ndt_weights is not None:
            # print("Recomputing ndt")
            self._ndt = self._ndt_design @ self._ndt_weights
        
        if self._boundary_weights is None and self._boundary is not None:
            # print("Recomputing boundary weights")
            self._boundary_weights = np.linalg.pinv(self._boundary_design) @ self._boundary

        if self._drift_weights is None and self._drift is not None:
            # print("Recomputing drift weights")
            self._drift_weights = np.linalg.pinv(self._drift_design) @ self._drift

        if self._ndt_weights is None and self._ndt is not None:
            # print("Recomputing ndt weights")
            self._ndt_weights = np.linalg.pinv(self._ndt_design) @ self._ndt
            
        self._is_fixed = True
    
    def set_boundary(self, boundary: np.ndarray):
        self._boundary = boundary
        self._is_fixed = False
    
    def set_drift(self, drift: np.ndarray):
        self._drift = drift
        self._is_fixed = False
        
    def set_ndt(self, ndt: np.ndarray):
        self._ndt = ndt
        self._is_fixed = False
        
    def boundary(self) -> np.ndarray:
        self.fix()
        return self._boundary_design @ self._boundary_weights
    
    def drift(self) -> np.ndarray:
        self.fix()
        return self._drift_design @ self._drift_weights
    
    def ndt(self) -> np.ndarray:
        self.fix()
        return self._ndt_design @ self._ndt_weights
    
    def boundary_nd(self) -> np.ndarray:
        self.fix()
        return self._boundary_design
    
    def drift_nd(self) -> np.ndarray:
        self.fix()
        return self._drift_design
    
    def ndt_nd(self) -> np.ndarray:
        self.fix()
        return self._ndt_design
    
    def boundary(self) -> np.ndarray:
        self.fix()
        return np.array(self._boundary)
    
    def drift(self) -> np.ndarray:
        self.fix()
        return np.array(self._drift)
    
    def ndt(self) -> np.ndarray:
        self.fix()
        return np.array(self._ndt)
    
    def boundary_weights(self) -> np.ndarray:
        self.fix()
        return self._boundary_weights
    
    def drift_weights(self) -> np.ndarray:
        self.fix()
        return self._drift_weights
    
    def ndt_weights(self) -> np.ndarray:
        self.fix()
        return self._ndt_weights
    
    def boundary_design(self) -> np.ndarray:
        return self._boundary_design
    
    def drift_design(self) -> np.ndarray:
        return self._drift_design
    
    def ndt_design(self) -> np.ndarray:
        return self._ndt_design
    
    def as_parameter_list(self) -> list[Parameters]:
        return [
            Parameters(boundary=a, drift=d, ndt=t) 
            for a, d, t in zip(self.boundary(), self.drift(), self.ndt())
            ]
    
    def resample(self, sample_sizes: list[int]):
        
        if len(sample_sizes) != len(self.as_parameter_list()):
            raise ValueError("sample_sizes must be the same length as the number of parameters")
        
        self.fix()
        
        # Generate observed summary statistics
        print("Parameters (bound):")
        print([p.boundary for p in self.as_parameter_list()])
        print("Parameters (drift):")
        print([p.drift for p in self.as_parameter_list()])
        print("Parameters (ndt):")
        print([p.ndt for p in self.as_parameter_list()])
        
        observed_summary_statistics = forward_equations(self.as_parameter_list())
        
        # Resample the summary statistics
        resampled_summary_statistics = resample_summary_stats(observed_summary_statistics, sample_sizes)
        
        # Compute the parameters
        estimated_parameters = inverse_equations(resampled_summary_statistics)
        
        # Set the parameters
        self.set_boundary([p.boundary for p in estimated_parameters])
        self.set_drift([p.drift for p in estimated_parameters])
        self.set_ndt([p.ndt for p in estimated_parameters])
        
    
    def __str__(self):
        self.fix()
        return f"DesignMatrix(boundary_design={self._boundary_design}, \
                           drift_design={self._drift_design}, \
                           ndt_design={self._ndt_design}, \
                           boundary_weights={self._boundary_weights}, \
                           drift_weights={self._drift_weights}, \
                           ndt_weights={self._ndt_weights})"

def demonstrate_design_matrix_parameter_estimation():
    design_matrix = DesignMatrix(
        boundary_design = np.array([[1, 0, 0], 
                                    [0, 1, 0], 
                                    [0, 0, 1], 
                                    [0, 0, 1]]),
        drift_design = np.array([[1, 0, 0], 
                                 [0, 1, 0], 
                                 [0, 0, 1], 
                                 [0, 1, 0]]),
        ndt_design = np.array([[1, 0, 0], 
                               [0, 1, 0], 
                               [0, 0, 1], 
                               [1, 0, 0]]),
        boundary_weights = np.array([1.0, 1.5, 2.0]),
        drift_weights    = np.array([0.4, 0.8, 1.2]),
        ndt_weights      = np.array([0.2, 0.3, 0.4])
    )
    
    design_matrix.resample([10, 10, 10, 10])
    
    print(design_matrix)
    


"""
Test suite
"""
class TestSuite(unittest.TestCase):
    
    def test_design_matrix(self):
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
        np.testing.assert_array_equal(design_matrix.boundary_weights(), np.array([1.0, 1.5, 2.0]))
        np.testing.assert_array_equal(design_matrix.drift_weights(), np.array([0.4, 0.8, 1.2]))
        np.testing.assert_array_equal(design_matrix.ndt_weights(), np.array([0.2, 0.3, 0.4]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    
    parser.add_argument("--test", action="store_true", help="Run the test suite")
    parser.add_argument("--demo", action="store_true", help="Run the demo")
    
    args = parser.parse_args()
    
    if args.test:
        unittest.main(argv=[__file__], verbosity=0, failfast=True)
    if args.demo:
        demonstrate_design_matrix_parameter_estimation()
