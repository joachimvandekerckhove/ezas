#!/usr/bin/env python3

import numpy as np
import unittest
import argparse

"""
This module contains the core EZ diffusion model and related functions.
"""

"""
Parameters class
"""
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
        
    def array(self):
        return np.array([self.boundary, self.drift, self.ndt])
        
    def __sub__(self, other):
        if not isinstance(other, Parameters):
            raise TypeError("other must be an instance of Parameters")
        return Parameters(self.boundary - other.boundary,
                          self.drift - other.drift, 
                          self.ndt - other.ndt)

    def __str__(self):
        return f"{'Boundary:':10s}{self.boundary:5.2f}, " + \
               f"{'Drift:':10s}{self.drift:5.2f}, " + \
               f"{'NDT:':10s}{self.ndt:5.2f}"

    # Mean of an array of Parameter objects
    @staticmethod
    def mean(params_array: list['Parameters']) -> 'Parameters':
        if not isinstance(params_array, list):
            raise TypeError("params_array must be a list of Parameters")
        return Parameters(np.mean([p.boundary for p in params_array]),
                          np.mean([p.drift for p in params_array]),
                          np.mean([p.ndt for p in params_array]))
        
    # Standard deviation of an array of Parameter objects
    @staticmethod
    def std(params_array: list['Parameters']) -> 'Parameters':
        if not isinstance(params_array, list):
            raise TypeError("params_array must be a list of Parameters")
        return Parameters(np.std([p.boundary for p in params_array]),
                          np.std([p.drift for p in params_array]),
                          np.std([p.ndt for p in params_array]))
        
    # Quantiles of an array of Parameter objects
    @staticmethod
    def quantile(params_array: list['Parameters'], quantile: float) -> 'Parameters':
        if not isinstance(params_array, list):
            raise TypeError("params_array must be a list of Parameters")
        return Parameters(np.quantile([p.boundary for p in params_array], quantile),
                          np.quantile([p.drift for p in params_array], quantile),
                          np.quantile([p.ndt for p in params_array], quantile))
        
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
            rng.uniform(lower_bound.boundary, upper_bound.boundary),
            rng.uniform(lower_bound.drift, upper_bound.drift),
            rng.uniform(lower_bound.ndt, upper_bound.ndt)
        )

"""
SummaryStats class
"""
class SummaryStats:
    """Summary statistics from behavioral data"""
    def __init__(self, 
                 accuracy: float,
                 mean_rt: float, 
                 var_rt: float):
        if not isinstance(accuracy, float):
            raise TypeError("Accuracy must be a number.")
        if not isinstance(mean_rt, (int, float)):
            raise TypeError("Mean RT must be a number.")
        if not isinstance(var_rt, (int, float)):
            raise TypeError("Var RT must be a number.")

        self.accuracy = accuracy
        self.mean_rt = mean_rt
        self.var_rt = var_rt
    
    def to_observations(self, sample_size: int) -> 'Observations':
        return Observations(self.accuracy, self.mean_rt, self.var_rt, sample_size)
    
    def sample(self, sample_size: int, rng = np.random) -> 'Observations':
        return Observations(
            accuracy = rng.binomial(sample_size, self.accuracy),
            mean_rt = rng.normal(self.mean_rt, np.sqrt(self.var_rt/sample_size)),
            var_rt = rng.gamma((sample_size-1)/2, 2*self.var_rt/(sample_size-1)),
            sample_size = sample_size
        )
    
    def __sub__(self, other):
        if not isinstance(other, SummaryStats):
            raise TypeError("other must be an instance of SummaryStats")
        return SummaryStats(self.accuracy - other.accuracy, 
                            self.mean_rt - other.mean_rt, 
                            self.var_rt - other.var_rt)

    def __str__(self):
        return f"{'Accuracy:':10s}{self.accuracy:5.2f}, " + \
               f"{'Mean RT:':10s}{self.mean_rt:5.2f}, " + \
               f"{'Var RT:':10s}{self.var_rt:5.2f}"
    
    
"""
Observations class
"""
class Observations:
    """Observations from behavioral data"""
    def __init__(self, 
                 accuracy: int,
                 mean_rt: float, 
                 var_rt: float,
                 sample_size: int):
        if not isinstance(accuracy, int):
            raise TypeError("Accuracy must be an integer.")
        if not isinstance(mean_rt, (int, float)):
            raise TypeError("Mean RT must be a number.")
        if not isinstance(var_rt, (int, float)):
            raise TypeError("Var RT must be a number.")
        if not isinstance(sample_size, int):
            raise TypeError("Sample size must be an integer.")

        self._summary_stats = SummaryStats(accuracy/sample_size, mean_rt, var_rt)
        self._sample_size = sample_size
        
    def accuracy(self) -> int:
        return round(self._summary_stats.accuracy * self._sample_size)
    
    def mean_rt(self) -> float:
        return self._summary_stats.mean_rt
    
    def var_rt(self) -> float:
        return self._summary_stats.var_rt
    
    def sample_size(self) -> int:
        return self._sample_size
    
    def to_summary_stats(self) -> SummaryStats:
        return self._summary_stats
    
    def __str__(self):
        return f"{'Accuracy:':10s}{self.accuracy()}, " + \
               f"{'Mean RT:':10s}{self.mean_rt():5.2f}, " + \
               f"{'Var RT:':10s}{self.var_rt():5.2f}, " + \
               f"{'Sample size:':10s}{self.sample_size()}"
    
"""
Resample summary statistics
"""
def resample_summary_stats(
    summary_stats: SummaryStats | list[SummaryStats], 
    sample_sizes: int | list[int], 
    rng = np.random) -> Observations | list[Observations]:
    
    if isinstance(summary_stats, SummaryStats) and \
        isinstance(sample_sizes, int):
        return summary_stats.sample(sample_sizes, rng)
                
    if isinstance(summary_stats, list) and \
        all(isinstance(x, SummaryStats) for x in summary_stats) and \
        isinstance(sample_sizes, list) and \
        all(isinstance(x, int) for x in sample_sizes):
        return [s.sample(N, rng) for s, N in zip(summary_stats, sample_sizes)]
    
    raise TypeError("summary_stats must be a SummaryStats instance or a list of SummaryStats and sample_sizes must be an integer or a list of integers")
    


"""
Forward equations
"""
def forward_equations(params: Parameters | list[Parameters]) -> SummaryStats | list[SummaryStats]:
    if not (isinstance(params, Parameters) or (isinstance(params, list) and all(isinstance(x, Parameters) for x in params))):
        raise TypeError("params must be a Parameters instance or a list of Parameters")
    if isinstance(params, Parameters):
        return _forward_equations_single(params)
    if isinstance(params, list):
        return [_forward_equations_single(x) for x in params]
    
def _forward_equations_single(params: Parameters) -> SummaryStats:
    """Calculate predicted summary statistics from parameters"""
    if params.boundary <= 0:
        raise ValueError("Boundary must be strictly positive.")
    
    if params.drift == 0:
        # Special case for drift = 0
        return SummaryStats(0.5, params.ndt + params.boundary, params.boundary)
    
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
    


"""
Inverse equations
"""
def inverse_equations(obs_stats: SummaryStats | list[SummaryStats]) -> Parameters | list[Parameters]:
    """Estimate parameters from observed summary statistics"""
    if isinstance(obs_stats, Observations):
        obs_stats = obs_stats.to_summary_stats()
    if isinstance(obs_stats, list) and all(isinstance(x, Observations) for x in obs_stats):
        obs_stats = [s.to_summary_stats() for s in obs_stats]
        
    if not (isinstance(obs_stats, SummaryStats) or (isinstance(obs_stats, list) and all(isinstance(x, SummaryStats) for x in obs_stats))):
        raise TypeError("obs_stats must be a SummaryStats instance or a list of SummaryStats")
    
    if isinstance(obs_stats, SummaryStats):
        return _inverse_equations_single(obs_stats)
    if isinstance(obs_stats, list):
        return [_inverse_equations_single(x) for x in obs_stats]
    
def _inverse_equations_single(obs_stats: SummaryStats) -> Parameters:
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
        print(f"WARNING: Numerical failure for N = {obs_stats.N}")
        return Parameters(1.0, 0.0, obs_stats.mean_rt)
    
    
"""
Generate random parameters
"""
def generate_random_parameters(lower_bound: Parameters, 
                               upper_bound: Parameters) -> Parameters:
    """Generate random parameters within given bounds"""
    return Parameters.random(lower_bound, upper_bound)

# """
# Resample statistics
# """
# def sample_statistics(pred_stats: SummaryStats, N: int) -> SummaryStats:
#     """Resample summary statistics from given summary statistics"""
#     if not isinstance(pred_stats, SummaryStats):
#         raise TypeError("pred_stats must be an instance of SummaryStats")
#     if not isinstance(N, int):
#         raise TypeError("N must be an integer")
#     if N <= 1:
#         raise ValueError("N must be greater than 1")
    
#     return resample_summary_stats(pred_stats, N)


"""
Test suite
"""
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
        self.assertRaises(ValueError, forward_equations, Parameters(0.0, 1.0, 0.0))

    def test_forward_equations_special_case(self):
        drift = 0.0
        boundary = 0.5
        ndt = 0.1
        true_params = Parameters(boundary, drift, ndt)
        pred_stats = forward_equations(true_params)
        self.assertIsInstance(pred_stats, SummaryStats)
        self.assertEqual(pred_stats.accuracy, 0.5)
        self.assertEqual(pred_stats.mean_rt, ndt + boundary)
        self.assertEqual(pred_stats.var_rt, boundary)
        
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

    def test_resample_summary_stats(self):
        accuracy = 8
        mean_rt = 0.33
        var_rt = 0.05
        sample_size = 10
        pred_stats = SummaryStats(accuracy/sample_size, mean_rt, var_rt)
        obs_stats = resample_summary_stats(pred_stats, sample_size)
        self.assertIsInstance(obs_stats, Observations)
        self.assertEqual(obs_stats.sample_size(), sample_size)
                
    def test_parameter_array_mean(self):
        boundary_array = [1.0, 2.0, 3.0]
        drift_array = [0.5, -0.5, 0.0]
        ndt_array = [0.2, 0.3, 0.4]
        true_mean_params = Parameters(np.mean(boundary_array), np.mean(drift_array), np.mean(ndt_array))
        params_array = [Parameters(boundary, drift, ndt) for boundary, drift, ndt in zip(boundary_array, drift_array, ndt_array)]
        mean_params = Parameters.mean(params_array)
        self.assertIsInstance(mean_params, Parameters)
        self.assertLess(abs(mean_params.boundary - true_mean_params.boundary), 1e-6)
        self.assertLess(abs(mean_params.drift - true_mean_params.drift), 1e-6)
        self.assertLess(abs(mean_params.ndt - true_mean_params.ndt), 1e-6)
        
    def test_parameter_array_std(self):
        boundary_array = [1.0, 2.0, 3.0]
        drift_array = [0.5, -0.5, 0.0]
        ndt_array = [0.2, 0.3, 0.4]
        true_std_params = Parameters(np.std(boundary_array), np.std(drift_array), np.std(ndt_array))
        params_array = [Parameters(boundary, drift, ndt) for boundary, drift, ndt in zip(boundary_array, drift_array, ndt_array)]
        std_params = Parameters.std(params_array)
        self.assertIsInstance(std_params, Parameters)
        self.assertLess(abs(std_params.boundary - true_std_params.boundary), 1e-6)
        self.assertLess(abs(std_params.drift - true_std_params.drift), 1e-6)
        self.assertLess(abs(std_params.ndt - true_std_params.ndt), 1e-6)
        
    def test_parameter_array_quantile(self):
        rng = np.random.default_rng(42)
        boundary_array = rng.uniform(0, 1, 100)
        drift_array = rng.uniform(-1, 1, 100)
        ndt_array = rng.uniform(0, 1, 100)
        quantile = 0.95
        true_quantile_params = Parameters(np.quantile(boundary_array, quantile), np.quantile(drift_array, quantile), np.quantile(ndt_array, quantile))
        params_array = [Parameters(boundary, drift, ndt) for boundary, drift, ndt in zip(boundary_array, drift_array, ndt_array)]
        quantile_params = Parameters.quantile(params_array, quantile)
        self.assertIsInstance(quantile_params, Parameters)
        self.assertLess(abs(quantile_params.boundary - true_quantile_params.boundary), 1e-6)
        self.assertLess(abs(quantile_params.drift - true_quantile_params.drift), 1e-6)
        self.assertLess(abs(quantile_params.ndt - true_quantile_params.ndt), 1e-6)
        
    def test_parameter_array_random(self):
        rng = np.random.default_rng(42)
        lower_bound = Parameters(0, 0, 0)
        upper_bound = Parameters(1, 1, 1)
        random_params = Parameters.random(lower_bound, upper_bound, rng = rng)
        self.assertIsInstance(random_params, Parameters)
        self.assertGreaterEqual(random_params.boundary, lower_bound.boundary)
        self.assertLessEqual(random_params.boundary, upper_bound.boundary)
        self.assertGreaterEqual(random_params.drift, lower_bound.drift)
        self.assertLessEqual(random_params.drift, upper_bound.drift)
        self.assertGreaterEqual(random_params.ndt, lower_bound.ndt)
        self.assertLessEqual(random_params.ndt, upper_bound.ndt)
        
    def test_parameter_array_random_without_bounds(self):
        rng = np.random.default_rng(42)
        random_params = Parameters.random(rng = rng)
        self.assertIsInstance(random_params, Parameters)
        self.assertGreaterEqual(random_params.boundary, 0)
        self.assertGreaterEqual(random_params.ndt, 0)
    
    # def test_bayesian_parameter_estimation(self):
    #     # Generate some test data
    #     true_params = Parameters(1.0, 0.5, 0.2)
    #     pred_stats = forward_equations(true_params)
    #     obs_stats = sample_statistics(pred_stats, 100)  # Use more trials for better estimates
        
    #     # Run Bayesian estimation
    #     est_params = bayesian_parameter_estimation(obs_stats, 100, n_samples=100, n_tune=50)
        
    #     # Check that estimates are reasonable
    #     self.assertIsInstance(est_params, Parameters)
    #     self.assertGreater(est_params.boundary, 0)
    #     self.assertGreater(est_params.ndt, 0)
        
    #     # Check that estimates are within reasonable range of true values
    #     self.assertLess(abs(est_params.boundary - true_params.boundary), 1.0)
    #     self.assertLess(abs(est_params.drift - true_params.drift), 1.0)
    #     self.assertLess(abs(est_params.ndt - true_params.ndt), 0.5)

def demo():
    # Demonstrate the use of some of the functions in this module
    print(" > Create parameters", end="\t> ")
    true_params = Parameters(1.0, 0.5, 0.2)
    print(true_params)
    
    print(" > Forward equations", end="\t> ")
    pred_stats = forward_equations(true_params)
    print(pred_stats)
    
    print(" > Resample statistics", end="\t> ")
    obs_stats = resample_summary_stats(pred_stats, 100)
    print(obs_stats)
    
    print(" > Inverse equations", end="\t> ")
    est_params = inverse_equations(obs_stats)
    print(est_params)
    
    print(" > Recovery difference", end="\t> ")
    difference = true_params - est_params
    print(difference)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--test", action="store_true", help="Run the test suite")
    parser.add_argument("--demo", action="store_true", help="Run the demo")
    
    args = parser.parse_args()
    
    if args.test:
        print("Running test suite...")
        unittest.main(argv=[__file__], verbosity=0, failfast=True)
    if args.demo:
        print("Running demo...")
        demo()
