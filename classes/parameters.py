import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import unittest
import argparse

_LOWER_QUANTILE = 0.025
_UPPER_QUANTILE = 0.975

class Parameters:
    """Model parameters for EZ diffusion"""
    def __init__(self, 
                 boundary: float, 
                 drift: float, 
                 ndt: float,
                 boundary_sd: float = None,
                 drift_sd: float = None,
                 ndt_sd: float = None,
                 boundary_lower_bound: float = None,
                 boundary_upper_bound: float = None,
                 drift_lower_bound: float = None,
                 drift_upper_bound: float = None,
                 ndt_lower_bound: float = None,
                 ndt_upper_bound: float = None):
        if not isinstance(boundary, (int, float)):
            raise TypeError("Boundary must be a number.")
        if not isinstance(drift, (int, float)):
            raise TypeError("Drift must be a number.")
        if not isinstance(ndt, (int, float)):
            raise TypeError("NDT must be a number.")
        
        if boundary_sd is not None:
            if not isinstance(boundary_sd, (int, float)):
                raise TypeError("Boundary SD must be a number.")
        if drift_sd is not None:
            if not isinstance(drift_sd, (int, float)):
                raise TypeError("Drift SD must be a number.")
        if ndt_sd is not None:
            if not isinstance(ndt_sd, (int, float)):
                raise TypeError("NDT SD must be a number.")
        
        if boundary_lower_bound is not None:
            if not isinstance(boundary_lower_bound, (int, float)):
                raise TypeError("Boundary lower bound must be a number.")
        if boundary_upper_bound is not None:
            if not isinstance(boundary_upper_bound, (int, float)):
                raise TypeError("Boundary upper bound must be a number.")
        if drift_lower_bound is not None:
            if not isinstance(drift_lower_bound, (int, float)):
                raise TypeError("Drift lower bound must be a number.")
        if drift_upper_bound is not None:
            if not isinstance(drift_upper_bound, (int, float)):
                raise TypeError("Drift upper bound must be a number.")
        if ndt_lower_bound is not None:
            if not isinstance(ndt_lower_bound, (int, float)):
                raise TypeError("NDT lower bound must be a number.")
        if ndt_upper_bound is not None:
            if not isinstance(ndt_upper_bound, (int, float)):
                raise TypeError("NDT upper bound must be a number.")
        
        self._boundary = boundary
        self._drift = drift
        self._ndt = ndt
        
        self._boundary_sd = boundary_sd
        self._drift_sd = drift_sd
        self._ndt_sd = ndt_sd
        
        self._boundary_lower_bound = boundary_lower_bound
        self._boundary_upper_bound = boundary_upper_bound
        self._drift_lower_bound = drift_lower_bound
        self._drift_upper_bound = drift_upper_bound
        self._ndt_lower_bound = ndt_lower_bound
        self._ndt_upper_bound = ndt_upper_bound
        
    def boundary(self):
        return self._boundary
    
    def drift(self):
        return self._drift
    
    def ndt(self):
        return self._ndt
    
    def boundary_sd(self):
        return self._boundary_sd
    
    def drift_sd(self):
        return self._drift_sd
    
    def ndt_sd(self):
        return self._ndt_sd
    
    def boundary_lower_bound(self):
        return self._boundary_lower_bound
    
    def boundary_upper_bound(self):
        return self._boundary_upper_bound
    
    def drift_lower_bound(self):
        return self._drift_lower_bound
    
    def drift_upper_bound(self):
        return self._drift_upper_bound
    
    def ndt_lower_bound(self):
        return self._ndt_lower_bound
    
    def ndt_upper_bound(self):
        return self._ndt_upper_bound
    
    def has_sd(self):
        return self._boundary_sd is not None or \
               self._drift_sd is not None or \
               self._ndt_sd is not None
    
    def has_bounds(self):
        return self._boundary_lower_bound is not None or \
               self._boundary_upper_bound is not None or \
               self._drift_lower_bound is not None or \
               self._drift_upper_bound is not None or \
               self._ndt_lower_bound is not None or \
               self._ndt_upper_bound is not None
    
    def is_within_bounds_of(self, other) -> tuple[bool, bool, bool]:
        if not isinstance(other, Parameters):
            raise TypeError("other must be an instance of Parameters")
        if not other.has_bounds():
            raise ValueError("other must have bounds")
        
        boundary_in_bounds = self.boundary() >= other.boundary_lower_bound() and \
                             self.boundary() <= other.boundary_upper_bound()
        drift_in_bounds = self.drift() >= other.drift_lower_bound() and \
                          self.drift() <= other.drift_upper_bound()
        ndt_in_bounds = self.ndt() >= other.ndt_lower_bound() and \
                        self.ndt() <= other.ndt_upper_bound()
        
        return (boundary_in_bounds, drift_in_bounds, ndt_in_bounds)
    
    def array(self):
        return np.array([self._boundary, self._drift, self._ndt])
        
    def __sub__(self, other):
        if not isinstance(other, Parameters):
            raise TypeError("other must be an instance of Parameters")
        if self.has_sd() or other.has_sd() or self.has_bounds() or other.has_bounds():
            raise ValueError("Subtraction of Parameters with SDs is not supported")
        return Parameters(self._boundary - other._boundary,
                          self._drift - other._drift, 
                          self._ndt - other._ndt)

    def __eq__(self, other):
        if not isinstance(other, Parameters):
            return False
        return self.boundary() == other.boundary() and \
               self.drift() == other.drift() and \
               self.ndt() == other.ndt()
    
    def __str__(self):
        if self.has_bounds():
            return f"{'Boundary:':10s}{self._boundary:5.2f} in [{self._boundary_lower_bound:5.2f}, {self._boundary_upper_bound:5.2f}], " + \
                   f"{'Drift:':10s}{self._drift:5.2f} in [{self._drift_lower_bound:5.2f}, {self._drift_upper_bound:5.2f}], " + \
                   f"{'NDT:':10s}{self._ndt:5.2f} in [{self._ndt_lower_bound:5.2f}, {self._ndt_upper_bound:5.2f}]"
        elif self.has_sd():
            return f"{'Boundary:':10s}{self._boundary:5.2f} +/- {self._boundary_sd:5.2f}, " + \
                   f"{'Drift:':10s}{self._drift:5.2f} +/- {self._drift_sd:5.2f}, " + \
                   f"{'NDT:':10s}{self._ndt:5.2f} +/- {self._ndt_sd:5.2f}"
        else:
            return f"{'Boundary:':10s}{self._boundary:5.2f}, " + \
                   f"{'Drift:':10s}{self._drift:5.2f}, " + \
                   f"{'NDT:':10s}{self._ndt:5.2f}"

    # Mean of an array of Parameter objects
    @staticmethod
    def mean(params_array: list['Parameters']) -> 'Parameters':
        if not isinstance(params_array, list):
            raise TypeError("params_array must be a list of Parameters")
        return Parameters(np.mean([p.boundary() for p in params_array]),
                          np.mean([p.drift() for p in params_array]),
                          np.mean([p.ndt() for p in params_array]))
        
    # Standard deviation of an array of Parameter objects
    @staticmethod
    def sd(params_array: list['Parameters']) -> 'Parameters':
        if not isinstance(params_array, list):
            raise TypeError("params_array must be a list of Parameters")
        return Parameters(np.std([p.boundary() for p in params_array]),
                          np.std([p.drift() for p in params_array]),
                          np.std([p.ndt() for p in params_array]))
        
    # Quantiles of an array of Parameter objects
    @staticmethod
    def quantile(params_array: list['Parameters'], quantile: float) -> 'Parameters':
        if not isinstance(params_array, list):
            raise TypeError("params_array must be a list of Parameters")
        return Parameters(np.quantile([p.boundary() for p in params_array], quantile),
                          np.quantile([p.drift() for p in params_array], quantile),
                          np.quantile([p.ndt() for p in params_array], quantile))
        
    # Randomly generate a Parameter object within a given range
    @staticmethod
    def random(lower_bound = None, upper_bound = None, rng = np.random) -> 'Parameters':
        if lower_bound is None:
            lower_bound = Parameters(0, 0, 0)
        if upper_bound is None:
            upper_bound = Parameters(1, 1, 1)
        if not isinstance(lower_bound, Parameters):
            raise TypeError("lower_bound must be an instance of Parameters")
        if not isinstance(upper_bound, Parameters):
            raise TypeError("upper_bound must be an instance of Parameters")
        
        return Parameters(
            rng.uniform(lower_bound.boundary(), upper_bound.boundary()),
            rng.uniform(lower_bound.drift(), upper_bound.drift()),
            rng.uniform(lower_bound.ndt(), upper_bound.ndt())
        )
        
    @staticmethod
    def summarize(parameter_list: list['Parameters']) -> 'Parameters':
        if not isinstance(parameter_list, list) or not all(isinstance(p, Parameters) for p in parameter_list):
            raise TypeError("parameter_list must be a list of Parameters")

        mean_params = Parameters.mean(parameter_list)
        sd_params = Parameters.sd(parameter_list)
        lower_quantile_params = Parameters.quantile(parameter_list, _LOWER_QUANTILE)
        upper_quantile_params = Parameters.quantile(parameter_list, _UPPER_QUANTILE)
        
        return Parameters(
            boundary = mean_params.boundary(),
            drift = mean_params.drift(),
            ndt = mean_params.ndt(),
            boundary_sd = sd_params.boundary(),
            drift_sd = sd_params.drift(),
            ndt_sd = sd_params.ndt(),
            boundary_lower_bound = lower_quantile_params.boundary(),
            drift_lower_bound = lower_quantile_params.drift(),
            ndt_lower_bound = lower_quantile_params.ndt(),
            boundary_upper_bound = upper_quantile_params.boundary(),
            drift_upper_bound = upper_quantile_params.drift(),
            ndt_upper_bound = upper_quantile_params.ndt()
        )
    
    @staticmethod
    def summarize_matrix(params_matrix: list[list['Parameters']]) -> list['Parameters']:
        """Summarize a matrix of Parameters (list of lists) into a list of summarized Parameters.
        
        Args:
            params_matrix: List of lists of Parameters, where each inner list 
                          contains parameters from one replication
        Returns:
            List of Parameters with mean, std, and quantiles for each parameter position
        """
        if not isinstance(params_matrix, list):
            raise TypeError("params_matrix must be a list")
        if not all(isinstance(row, list) for row in params_matrix):
            raise TypeError("params_matrix must be a list of lists")
        if not all(all(isinstance(p, Parameters) for p in row) for row in params_matrix):
            raise TypeError("All elements must be Parameters objects")
        
        # Get the number of parameters per replication
        n_params = len(params_matrix[0])
        if not all(len(row) == n_params for row in params_matrix):
            raise ValueError("All rows must have the same number of parameters")
        
        # Summarize each parameter position across replications
        summarized_params = []
        for i in range(n_params):
            param_list = [row[i] for row in params_matrix]
            summarized_params.append(Parameters.summarize(param_list))
        
        return summarized_params

    # Mean of a matrix of Parameter objects (list of lists)
    @staticmethod
    def mean_matrix(params_matrix: list[list['Parameters']]) -> list['Parameters']:
        """Compute the mean of parameters across replications.
        
        Args:
            params_matrix: List of lists of Parameters, where each inner list 
                          contains parameters from one replication
        Returns:
            List of Parameters representing the mean across replications
        """
        if not isinstance(params_matrix, list):
            raise TypeError("params_matrix must be a list")
        if not all(isinstance(row, list) for row in params_matrix):
            raise TypeError("params_matrix must be a list of lists")
        if not all(all(isinstance(p, Parameters) for p in row) for row in params_matrix):
            raise TypeError("All elements must be Parameters objects")
        
        # Get the number of parameters per replication
        n_params = len(params_matrix[0])
        if not all(len(row) == n_params for row in params_matrix):
            raise ValueError("All rows must have the same number of parameters")
        
        # Compute mean for each parameter position across replications
        mean_params = []
        for i in range(n_params):
            param_list = [row[i] for row in params_matrix]
            mean_params.append(Parameters.mean(param_list))
        
        return mean_params
    
    # Standard deviation of a matrix of Parameter objects (list of lists)
    @staticmethod
    def std_matrix(params_matrix: list[list['Parameters']]) -> list['Parameters']:
        """Compute the standard deviation of parameters across replications.
        
        Args:
            params_matrix: List of lists of Parameters, where each inner list 
                          contains parameters from one replication
        Returns:
            List of Parameters representing the standard deviation across replications
        """
        if not isinstance(params_matrix, list):
            raise TypeError("params_matrix must be a list")
        if not all(isinstance(row, list) for row in params_matrix):
            raise TypeError("params_matrix must be a list of lists")
        if not all(all(isinstance(p, Parameters) for p in row) for row in params_matrix):
            raise TypeError("All elements must be Parameters objects")
        
        # Get the number of parameters per replication
        n_params = len(params_matrix[0])
        if not all(len(row) == n_params for row in params_matrix):
            raise ValueError("All rows must have the same number of parameters")
        
        # Compute standard deviation for each parameter position across replications
        std_params = []
        for i in range(n_params):
            param_list = [row[i] for row in params_matrix]
            std_params.append(Parameters.sd(param_list))
        
        return std_params
    
    # Quantile of a matrix of Parameter objects (list of lists)
    @staticmethod
    def quantile_matrix(params_matrix: list[list['Parameters']], quantile: float) -> list['Parameters']:
        """Compute the quantile of parameters across replications.
        
        Args:
            params_matrix: List of lists of Parameters, where each inner list 
                          contains parameters from one replication
            quantile: Quantile value (0.0 to 1.0)
        Returns:
            List of Parameters representing the quantile across replications
        """
        if not isinstance(params_matrix, list):
            raise TypeError("params_matrix must be a list")
        if not all(isinstance(row, list) for row in params_matrix):
            raise TypeError("params_matrix must be a list of lists")
        if not all(all(isinstance(p, Parameters) for p in row) for row in params_matrix):
            raise TypeError("All elements must be Parameters objects")
        
        # Get the number of parameters per replication
        n_params = len(params_matrix[0])
        if not all(len(row) == n_params for row in params_matrix):
            raise ValueError("All rows must have the same number of parameters")
        
        # Compute quantile for each parameter position across replications
        quantile_params = []
        for i in range(n_params):
            param_list = [row[i] for row in params_matrix]
            quantile_params.append(Parameters.quantile(param_list, quantile))
        
        return quantile_params

# --- TESTS AND DEMO ---
class TestSuite(unittest.TestCase):
    def test_basic_init(self):
        """
        Test that the Parameters class works correctly.
        """
        p = Parameters(1.0, 0.5, 0.2)
        self.assertEqual(p.boundary(), 1.0)
        self.assertEqual(p.drift(), 0.5)
        self.assertEqual(p.ndt(), 0.2)
    def test_bounds(self):
        """
        Test that the Parameters class works correctly with bounds.
        """
        p = Parameters(1.0, 0.5, 0.2, boundary_lower_bound=0.5, boundary_upper_bound=2.0)
        self.assertTrue(p.has_bounds())
    def test_sd(self):
        """
        Test that the Parameters class works correctly with standard deviations.
        """
        p = Parameters(1.0, 0.5, 0.2, boundary_sd=0.1)
        self.assertTrue(p.has_sd())
    def test_sub(self):
        """
        Test that the Parameters class works correctly with subtraction.
        """
        p1 = Parameters(1.0, 0.5, 0.2)
        p2 = Parameters(0.5, 0.2, 0.1)
        diff = p1 - p2
        self.assertAlmostEqual(diff.boundary(), 0.5)
        self.assertAlmostEqual(diff.drift(), 0.3)
        self.assertAlmostEqual(diff.ndt(), 0.1)
    def test_eq(self):
        """
        Test that the Parameters class works correctly with equality.
        """
        p1 = Parameters(1.0, 0.5, 0.2)
        p2 = Parameters(1.0, 0.5, 0.2)
        self.assertTrue(p1 == p2)
        p3 = Parameters(1.0, 0.5, 0.3)
        self.assertFalse(p1 == p3)
        p4 = Parameters(1.0, 0.6, 0.2)
        self.assertFalse(p1 == p4)
        p5 = Parameters(1.1, 0.5, 0.2)
        self.assertFalse(p1 == p5)
        self.assertFalse(p1 == None)
        self.assertFalse(p1 == "not a Parameters object")
        
    def test_random(self):
        """
        Test that the Parameters class works correctly with random parameters.
        """
        rng = np.random.default_rng(42)
        lb = Parameters(0, 0, 0)
        ub = Parameters(1, 1, 1)
        p = Parameters.random(lb, ub, rng)
        self.assertTrue(lb.boundary() <= p.boundary() <= ub.boundary())
        self.assertTrue(lb.drift() <= p.drift() <= ub.drift())
        self.assertTrue(lb.ndt() <= p.ndt() <= ub.ndt())

def demo():
    print("Demo for Parameters class:")
    p = Parameters(1.0, 0.5, 0.2, boundary_sd=0.1, drift_sd=0.05, ndt_sd=0.01)
    print("Created:", p)
    print("Boundary:", p.boundary())
    print("Drift:", p.drift())
    print("NDT:", p.ndt())
    print("Has SD:", p.has_sd())
    print("Has bounds:", p.has_bounds())
    lb = Parameters(0, 0, 0)
    ub = Parameters(2, 2, 2)
    rand_p = Parameters.random(lb, ub)
    print("Random parameters in [0,2]:", rand_p)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run the test suite")
    parser.add_argument("--demo", action="store_true", help="Run the demo")
    args = parser.parse_args()
    if args.test:
        unittest.main(argv=[__file__], verbosity=2, exit=False)
    if args.demo:
        demo() 