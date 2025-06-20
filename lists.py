#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import argparse
import unittest
from dataclasses import dataclass
import arviz as az
import pandas as pd

from vendor.ezas.base import ez_equations as ez
from vendor.ezas.classes.parameters import Parameters
    
"""
Design list
A list of design matrices, one for each parameter of the EZ diffusion model.
"""
@dataclass
class DesignList:
    boundary: np.ndarray
    drift: np.ndarray
    ndt: np.ndarray

"""
Weight list
A list of weights, one for each parameter of the EZ diffusion model.
"""
@dataclass
class WeightList:
    boundary: np.ndarray
    drift: np.ndarray
    ndt: np.ndarray
    boundary_sd: np.ndarray = None
    drift_sd: np.ndarray = None
    ndt_sd: np.ndarray = None
    boundary_q025: np.ndarray = None
    drift_q025: np.ndarray = None
    ndt_q025: np.ndarray = None
    boundary_q975: np.ndarray = None
    drift_q975: np.ndarray = None
    ndt_q975: np.ndarray = None
    
    def has_weights(self) -> bool:
        return self.boundary is not None and self.drift is not None and self.ndt is not None
    
    def has_sd(self) -> bool:
        return self.boundary_sd is not None and self.drift_sd is not None and self.ndt_sd is not None
    
    def has_quantiles(self) -> bool:
        return self.boundary_q025 is not None and self.drift_q025 is not None and self.ndt_q025 is not None and self.boundary_q975 is not None and self.drift_q975 is not None and self.ndt_q975 is not None
    
    def __str__(self):
        """
        Print as table with conditional formatting
        """
        if self.has_quantiles():
            return f"{'Boundary:':10s}{self.boundary:5.2f} in [{self.boundary_q025:5.2f}, {self.boundary_q975:5.2f}], " + \
                   f"{'Drift:':10s}{self.drift:5.2f} in [{self.drift_q025:5.2f}, {self.drift_q975:5.2f}], " + \
                   f"{'NDT:':10s}{self.ndt:5.2f} in [{self.ndt_q025:5.2f}, {self.ndt_q975:5.2f}]"
        elif self.has_sd():
            return f"{'Boundary:':10s}{self.boundary:5.2f} +/- {self.boundary_sd:5.2f}, " + \
                   f"{'Drift:':10s}{self.drift:5.2f} +/- {self.drift_sd:5.2f}, " + \
                   f"{'NDT:':10s}{self.ndt:5.2f} +/- {self.ndt_sd:5.2f}"
        else:
            return f"{'Boundary:':10s}{self.boundary:5.2f}, " + \
                   f"{'Drift:':10s}{self.drift:5.2f}, " + \
                   f"{'NDT:':10s}{self.ndt:5.2f}"
    
    @staticmethod
    def from_trace(trace: az.InferenceData, designList: DesignList) -> 'WeightList':
        
        # Print summary table for only those nodes with weight in their name
        summary = az.summary(trace)
        mask = summary.index.str.contains('weights')
        print(summary.loc[mask])
        
        post_mn = summary['mean']
        post_sd = summary['sd']
        quantiles = trace.posterior.quantile([0.025, 0.975])
        
        weightList = WeightList(
            # means
            boundary      = np.array([ post_mn[ f'boundary_weights[{i}]' ] for i in range(designList.boundary.shape[1]) ]),
            drift         = np.array([ post_mn[ f'drift_weights[{i}]'    ] for i in range(designList.drift.shape[1])    ]),
            ndt           = np.array([ post_mn[ f'ndt_weights[{i}]'      ] for i in range(designList.ndt.shape[1])      ]),
            # standard deviations
            boundary_sd   = np.array([ post_sd[ f'boundary_weights[{i}]' ] for i in range(designList.boundary.shape[1]) ]),
            drift_sd      = np.array([ post_sd[ f'drift_weights[{i}]'    ] for i in range(designList.drift.shape[1])    ]),
            ndt_sd        = np.array([ post_sd[ f'ndt_weights[{i}]'      ] for i in range(designList.ndt.shape[1])      ]),
            # quantiles
            boundary_q025 = quantiles[ 'boundary_weights' ].sel(quantile=0.025).values,
            drift_q025    = quantiles[ 'drift_weights'    ].sel(quantile=0.025).values,
            ndt_q025      = quantiles[ 'ndt_weights'      ].sel(quantile=0.025).values,
            boundary_q975 = quantiles[ 'boundary_weights' ].sel(quantile=0.975).values,
            drift_q975    = quantiles[ 'drift_weights'    ].sel(quantile=0.975).values,
            ndt_q975      = quantiles[ 'ndt_weights'      ].sel(quantile=0.975).values)
        
        return weightList

"""
Bigger
Given a design list and a weight list, return a list of parameters.
"""
def bigger(design_list: DesignList, weight_list: WeightList) -> list[Parameters]:
    boundary = design_list.boundary @ weight_list.boundary
    drift    = design_list.drift    @ weight_list.drift
    ndt      = design_list.ndt      @ weight_list.ndt
    return [Parameters(boundary=b, drift=d, ndt=n) for b, d, n in zip(boundary, drift, ndt)]

"""
Smaller
Given a design list and a list of parameters, return a weight list.
"""
def smaller(design_list: DesignList, parameters: list[Parameters]) -> WeightList:
    boundary = [p.boundary for p in parameters]
    drift    = [p.drift    for p in parameters]
    ndt      = [p.ndt      for p in parameters]
    return WeightList(boundary=np.linalg.pinv(design_list.boundary) @ boundary,
                      drift=np.linalg.pinv(design_list.drift) @ drift,
                      ndt=np.linalg.pinv(design_list.ndt) @ ndt)

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
        print([p.boundary() for p in self.as_parameter_list()])
        print("Parameters (drift):")
        print([p.drift() for p in self.as_parameter_list()])
        print("Parameters (ndt):")
        print([p.ndt() for p in self.as_parameter_list()])
        
        moments = ez.forward(self.as_parameter_list())
        
        # Resample the summary statistics
        observations = [
            s.sample(sample_size) 
            for s, sample_size in zip(moments, sample_sizes)
        ]
        
        # Compute the parameters
        estimated_parameters = ez.inverse(observations)
        
        # Set the parameters
        self.set_boundary([p.boundary() for p in estimated_parameters])
        self.set_drift([p.drift() for p in estimated_parameters])
        self.set_ndt([p.ndt() for p in estimated_parameters])
        
    
    def __str__(self):
        self.fix()
        return f"DesignMatrix(boundary_design={self._boundary_design}, \
                           drift_design={self._drift_design}, \
                           ndt_design={self._ndt_design}, \
                           boundary_weights={self._boundary_weights}, \
                           drift_weights={self._drift_weights}, \
                           ndt_weights={self._ndt_weights})"

"""
Demo
"""
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
