#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import argparse
import unittest

from vendor.ezas.base import ez_equations as ez
from vendor.ezas.classes.parameters import Parameters
from vendor.ezas.utils.linear_algebra import linear_prediction, linear_regression

_LOWER_QUANTILE = 0.025
_UPPER_QUANTILE = 0.975

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
        
        self._has_fixed_parameters = False
        self._has_fixed_weights = False

        self._fix_parameters()
        self._fix_weights()
    
    def _fix_parameters(self):
        if self._has_fixed_parameters:
            return

        # print(f"Fixing parameters")
        
        if self._boundary_weights is None or self._drift_weights is None or self._ndt_weights is None:
            return
        
        try:
            self._boundary = linear_prediction(self._boundary_design, self._boundary_weights)
        except Exception as e:
            print(f"Error in boundary prediction:")
            print(f"  Design matrix: {self._boundary_design}")
            print(f"  Weights: {self._boundary_weights}")
            raise e

        try:
            self._drift = linear_prediction(self._drift_design, self._drift_weights)
        except Exception as e:
            print(f"Error in drift prediction:")
            print(f"  Design matrix: {self._drift_design}")
            print(f"  Weights: {self._drift_weights}")
            raise e

        try:
            self._ndt = linear_prediction(self._ndt_design, self._ndt_weights)
        except Exception as e:
            print(f"Error in ndt prediction:")
            print(f"  Design matrix: {self._ndt_design}")
            print(f"  Weights: {self._ndt_weights}")
            raise e

        self._has_fixed_parameters = True
        
    def _fix_weights(self):
        if self._has_fixed_weights:
            return
        
        # print(f"Fixing weights")
        
        if self._boundary is None or self._drift is None or self._ndt is None:
            return
        
        try:
            self._boundary_weights = linear_regression(self._boundary_design, self._boundary)
        except Exception as e:
            print(f"Error in boundary regression:")
            print(f"  Design matrix: {self._boundary_design}")
            print(f"  Parameters: {self._boundary}")
            raise e

        try:
            self._drift_weights = linear_regression(self._drift_design, self._drift)
        except Exception as e:
            print(f"Error in drift regression:")
            print(f"  Design matrix: {self._drift_design}")
            print(f"  Parameters: {self._drift}")
            raise e

        try:
            self._ndt_weights = linear_regression(self._ndt_design, self._ndt)
        except Exception as e:
            print(f"Error in ndt regression:")
            print(f"  Design matrix: {self._ndt_design}")
            print(f"  Parameters: {self._ndt}")
            raise e
            
        self._has_fixed_weights = True
            
    def fix(self):
        self._fix_parameters()
        self._fix_weights()
    
    def set_boundary(self, boundary: np.ndarray):
        # print(f"Setting boundary to {boundary}")
        # print(f"Boundary was {self._boundary}")
        self._boundary = boundary
        # print(f"Boundary is now {self._boundary}")
        self._has_fixed_weights = False
    
    def set_drift(self, drift: np.ndarray):
        self._drift = drift
        self._has_fixed_weights = False
        
    def set_ndt(self, ndt: np.ndarray):
        self._ndt = ndt
        self._has_fixed_weights = False
    
    def set_parameters(self, parameters: list[Parameters]):
        # print(f"Setting parameters to")
        # [print(p) for p in parameters]
        self.set_boundary([p.boundary() for p in parameters])
        self.set_drift([p.drift() for p in parameters])
        self.set_ndt([p.ndt() for p in parameters])
        # print(f"Parameters are now")
        # [print(p) for p in self.as_parameter_list()]
    
    def boundary(self) -> np.ndarray:
        self._fix_weights()
        try:
            return linear_prediction(self._boundary_design, self._boundary_weights)
        except Exception as e:
            print(f"Error in boundary prediction:")
            print(f"  Design matrix: {self._boundary_design}")
            print(f"  Weights: {self._boundary_weights}")
            raise e
    
    def drift(self) -> np.ndarray:
        self._fix_weights()
        try:
            return linear_prediction(self._drift_design, self._drift_weights)
        except Exception as e:
            print(f"Error in drift prediction:")
            print(f"  Design matrix: {self._drift_design}")
            print(f"  Weights: {self._drift_weights}")
            raise e
    
    def ndt(self) -> np.ndarray:
        self._fix_weights()
        try:
            return linear_prediction(self._ndt_design, self._ndt_weights)
        except Exception as e:
            print(f"Error in ndt prediction:")
            print(f"  Design matrix: {self._ndt_design}")
            print(f"  Weights: {self._ndt_weights}")
            raise e
    
    def boundary_nd(self) -> np.ndarray:
        return self._boundary_design
    
    def drift_nd(self) -> np.ndarray:
        return self._drift_design
    
    def ndt_nd(self) -> np.ndarray:
        return self._ndt_design
    
    def boundary_weights(self) -> np.ndarray:
        self._fix_weights()
        return self._boundary_weights
    
    def drift_weights(self) -> np.ndarray:
        self._fix_weights()
        return self._drift_weights
    
    def ndt_weights(self) -> np.ndarray:
        self._fix_weights()
        return self._ndt_weights
    
    def boundary_design(self) -> np.ndarray:
        return self._boundary_design
    
    def drift_design(self) -> np.ndarray:
        return self._drift_design
    
    def ndt_design(self) -> np.ndarray:
        return self._ndt_design
    
    def as_parameter_list(self) -> list[Parameters]:
        self._fix_parameters()
        return [
            Parameters(boundary=a, drift=d, ndt=t) 
            for a, d, t in zip(self.boundary(), self.drift(), self.ndt())
            ]
    
    def get_parameters(self) -> list[Parameters]:
        return self.as_parameter_list()
    
    def sample(self, sample_sizes: list[int]):
        if len(sample_sizes) != self._boundary_design.shape[0]:
            raise ValueError("sample_sizes must be the same length as the number of parameters")
        
        self._fix_parameters()
        self._fix_weights()
        
        moments = ez.forward(self.as_parameter_list())
        return [ s.sample(n) for s, n in zip(moments, sample_sizes) ]
        
    def get_beta_weights(self) -> 'BetaWeights':
        self._fix_weights()
        return BetaWeights(
            beta_boundary_mean=self._boundary_weights,
            beta_drift_mean=self._drift_weights,
            beta_ndt_mean=self._ndt_weights
        )
    
    def __str__(self):
        self._fix_parameters()
        self._fix_weights()
        return f"DesignMatrix(boundary_design={self._boundary_design}, \
                           drift_design={self._drift_design}, \
                           ndt_design={self._ndt_design}, \
                           boundary_weights={self._boundary_weights}, \
                           drift_weights={self._drift_weights}, \
                           ndt_weights={self._ndt_weights})"

class BetaWeights:
    def __init__(self,
                 beta_boundary_mean: np.ndarray,
                 beta_drift_mean: np.ndarray,
                 beta_ndt_mean: np.ndarray,
                 beta_boundary_sd: np.ndarray = None,
                 beta_drift_sd: np.ndarray = None,
                 beta_ndt_sd: np.ndarray = None,
                 beta_boundary_lower: np.ndarray = None,
                 beta_drift_lower: np.ndarray = None,
                 beta_ndt_lower: np.ndarray = None,
                 beta_boundary_upper: np.ndarray = None,
                 beta_drift_upper: np.ndarray = None,
                 beta_ndt_upper: np.ndarray = None):
        self._beta_boundary_mean = beta_boundary_mean
        self._beta_drift_mean = beta_drift_mean
        self._beta_ndt_mean = beta_ndt_mean
        self._beta_boundary_sd = beta_boundary_sd
        self._beta_drift_sd = beta_drift_sd
        self._beta_ndt_sd = beta_ndt_sd
        self._beta_boundary_lower = beta_boundary_lower
        self._beta_drift_lower = beta_drift_lower
        self._beta_ndt_lower = beta_ndt_lower
        self._beta_boundary_upper = beta_boundary_upper
        self._beta_drift_upper = beta_drift_upper
        self._beta_ndt_upper = beta_ndt_upper
    
    def len(self) -> int:
        return len(self._beta_boundary_mean)
    
    def beta_boundary_mean(self) -> np.ndarray:
        return self._beta_boundary_mean
    
    def beta_drift_mean(self) -> np.ndarray:  
        return self._beta_drift_mean
    
    def beta_ndt_mean(self) -> np.ndarray:
        return self._beta_ndt_mean
    
    def beta_boundary_sd(self) -> np.ndarray:
        return self._beta_boundary_sd
    
    def beta_drift_sd(self) -> np.ndarray:
        return self._beta_drift_sd
    
    def beta_ndt_sd(self) -> np.ndarray:
        return self._beta_ndt_sd
    
    def beta_boundary_lower(self) -> np.ndarray:
        return self._beta_boundary_lower
    
    def beta_boundary_upper(self) -> np.ndarray:
        return self._beta_boundary_upper
    
    def beta_drift_lower(self) -> np.ndarray:
        return self._beta_drift_lower
    
    def beta_drift_upper(self) -> np.ndarray:
        return self._beta_drift_upper
    
    def beta_ndt_lower(self) -> np.ndarray:
        return self._beta_ndt_lower
    
    def beta_ndt_upper(self) -> np.ndarray:
        return self._beta_ndt_upper
    
    def has_sd(self) -> bool:
        return self._beta_boundary_sd is not None and \
               self._beta_drift_sd is not None and \
               self._beta_ndt_sd is not None
    
    def has_lower(self) -> bool:
        return self._beta_boundary_lower is not None and \
               self._beta_drift_lower is not None and \
               self._beta_ndt_lower is not None
    
    def has_upper(self) -> bool:
        return self._beta_boundary_upper is not None and \
               self._beta_drift_upper is not None and \
               self._beta_ndt_upper is not None
    
    def has_all(self) -> bool:
        return self.has_sd() and self.has_lower() and self.has_upper()
    
    def __str__(self):
        def pr(arr): 
            return np.array2string(arr, precision=3, separator=', ', formatter={'float_kind':lambda x: f"{x:6.3f}".rjust(6)})
        
        if self.has_all(): # All uncertainty
            return f"   BetaWeights(\n" + \
                   f"     {'beta_boundary_mean':<20} = {pr(self._beta_boundary_mean)} in [{pr(self._beta_boundary_lower)}, {pr(self._beta_boundary_upper)}], \n" + \
                   f"     {'beta_drift_mean'   :<20} = {pr(self._beta_drift_mean   )} in [{pr(self._beta_drift_lower   )}, {pr(self._beta_drift_upper   )}], \n" + \
                   f"     {'beta_ndt_mean'     :<20} = {pr(self._beta_ndt_mean     )} in [{pr(self._beta_ndt_lower     )}, {pr(self._beta_ndt_upper     )}]\n" + \
                   f"   )"
        elif self.has_sd(): # Only sd
            return f"   BetaWeights(\n" + \
                   f"     {'beta_boundary_mean':<20} = {pr(self._beta_boundary_mean)} ± {pr(self._beta_boundary_sd)}, \n" + \
                   f"     {'beta_drift_mean'   :<20} = {pr(self._beta_drift_mean   )} ± {pr(self._beta_drift_sd   )}, \n" + \
                   f"     {'beta_ndt_mean'     :<20} = {pr(self._beta_ndt_mean     )} ± {pr(self._beta_ndt_sd     )}\n" + \
                   f"   )"
        else:   # No uncertainty
            return f"   BetaWeights(\n" +  \
                   f"     {'beta_boundary_mean':<20} = {pr(self._beta_boundary_mean)}, \n" + \
                   f"     {'beta_drift_mean'   :<20} = {pr(self._beta_drift_mean   )}, \n" + \
                   f"     {'beta_ndt_mean'     :<20} = {pr(self._beta_ndt_mean     )}\n" + \
                   f"   )"
    
    @staticmethod
    def summarize(beta_weights: list['BetaWeights']) -> 'BetaWeights':
        return BetaWeights(
            beta_boundary_mean=np.mean([b.beta_boundary_mean() for b in beta_weights]),
            beta_drift_mean=np.mean([b.beta_drift_mean() for b in beta_weights]),
            beta_ndt_mean=np.mean([b.beta_ndt_mean() for b in beta_weights]),
            beta_boundary_sd=np.std([b.beta_boundary_mean() for b in beta_weights]),
            beta_drift_sd=np.std([b.beta_drift_mean() for b in beta_weights]),
            beta_ndt_sd=np.std([b.beta_ndt_mean() for b in beta_weights]),
            beta_boundary_lower=np.quantile([b.beta_boundary_mean() for b in beta_weights], _LOWER_QUANTILE),
            beta_drift_lower=np.quantile([b.beta_drift_mean() for b in beta_weights], _LOWER_QUANTILE),
            beta_ndt_lower=np.quantile([b.beta_ndt_mean() for b in beta_weights], _LOWER_QUANTILE),
            beta_boundary_upper=np.quantile([b.beta_boundary_mean() for b in beta_weights], _UPPER_QUANTILE),
            beta_drift_upper=np.quantile([b.beta_drift_mean() for b in beta_weights], _UPPER_QUANTILE),
            beta_ndt_upper=np.quantile([b.beta_ndt_mean() for b in beta_weights], _UPPER_QUANTILE),
        )
    
    @staticmethod
    def summarize_matrix(beta_weights_list: list['BetaWeights']) -> 'BetaWeights':
        """Summarize a list of BetaWeights by averaging each parameter position separately.
        
        Args:
            beta_weights_list: List of BetaWeights from different replications
        Returns:
            BetaWeights with mean, std, and quantiles for each parameter position
        """
        if not isinstance(beta_weights_list, list):
            raise TypeError("beta_weights_list must be a list")
        if not all(isinstance(b, BetaWeights) for b in beta_weights_list):
            raise TypeError("All elements must be BetaWeights objects")
        
        # Extract arrays for each parameter type across all replications
        boundary_means = [b.beta_boundary_mean() for b in beta_weights_list]
        drift_means = [b.beta_drift_mean() for b in beta_weights_list]
        ndt_means = [b.beta_ndt_mean() for b in beta_weights_list]
        
        # Average each parameter position separately
        avg_boundary = np.mean(boundary_means, axis=0)
        avg_drift = np.mean(drift_means, axis=0)
        avg_ndt = np.mean(ndt_means, axis=0)
        
        # Compute std and quantiles for each parameter position
        std_boundary = np.std(boundary_means, axis=0)
        std_drift = np.std(drift_means, axis=0)
        std_ndt = np.std(ndt_means, axis=0)
        
        lower_boundary = np.quantile(boundary_means, _LOWER_QUANTILE, axis=0)
        lower_drift = np.quantile(drift_means, _LOWER_QUANTILE, axis=0)
        lower_ndt = np.quantile(ndt_means, _LOWER_QUANTILE, axis=0)
        
        upper_boundary = np.quantile(boundary_means, _UPPER_QUANTILE, axis=0)
        upper_drift = np.quantile(drift_means, _UPPER_QUANTILE, axis=0)
        upper_ndt = np.quantile(ndt_means, _UPPER_QUANTILE, axis=0)
        
        return BetaWeights(
            beta_boundary_mean=avg_boundary,
            beta_drift_mean=avg_drift,
            beta_ndt_mean=avg_ndt,
            beta_boundary_sd=std_boundary,
            beta_drift_sd=std_drift,
            beta_ndt_sd=std_ndt,
            beta_boundary_lower=lower_boundary,
            beta_drift_lower=lower_drift,
            beta_ndt_lower=lower_ndt,
            beta_boundary_upper=upper_boundary,
            beta_drift_upper=upper_drift,
            beta_ndt_upper=upper_ndt
        )

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
        np.testing.assert_array_almost_equal(design_matrix.boundary_weights(), np.array([1.0, 1.5, 2.0]))
        np.testing.assert_array_almost_equal(design_matrix.drift_weights(), np.array([0.4, 0.8, 1.2]))
        np.testing.assert_array_almost_equal(design_matrix.ndt_weights(), np.array([0.2, 0.3, 0.4]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    
    parser.add_argument("--test", action="store_true", help="Run the test suite")
    parser.add_argument("--demo", action="store_true", help="Run the demo")
    
    args = parser.parse_args()
    
    if args.test:
        unittest.main(argv=[__file__], verbosity=0, failfast=True)
    if args.demo:
        demonstrate_design_matrix_parameter_estimation()
