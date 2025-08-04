#!/usr/bin/env python3

import numpy as np
import argparse
from bayes import bayesian_design_matrix_parameter_estimation
from base import ez_equations as ez
from classes.parameters import Parameters
from utils import run_simulation
from classes.design_matrix import DesignMatrix
from vendor.ezas.qnd.qnd import qnd_design_matrix_parameter_estimation
import unittest

"""
Run EZ diffusion simulation study
"""
def run_ezdiffusion_simulation_study():
    # Define bounds before running simulation
    lower_bound = Parameters(0.5, 1.0, 0.2)
    upper_bound = Parameters(2.0, 2.0, 0.5)
    
    for N in [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240]:
        print(f"N = {N}")
        print(run_simulation(N, lower_bound, upper_bound))

"""
Demonstrate QND parameter estimation
"""
def demonstrate_qnd_parameter_estimation():
    N = 56

    design_matrix = DesignMatrix(
        boundary_design  = np.array([[1, 0, 0], 
                                     [0, 1, 0], 
                                     [0, 0, 1], 
                                     [0, 0, 1]]),
        drift_design     = np.array([[1, 0, 0], 
                                     [0, 1, 0], 
                                     [0, 0, 1], 
                                     [0, 1, 0]]),
        ndt_design       = np.array([[1, 0, 0], 
                                     [0, 1, 0], 
                                     [0, 0, 1], 
                                     [1, 0, 0]]),
        boundary_weights = np.array([1.0, 1.5, 2.0]),
        drift_weights    = np.array([0.4, 0.8, 1.2]),
        ndt_weights      = np.array([0.3, 0.4, 0.5]))
    
    true_bounds = design_matrix.boundary()
    true_drifts = design_matrix.drift()
    true_ndts   = design_matrix.ndt()

    true_params_list = [Parameters(bound, drift, ndt) for bound, drift, ndt in zip(true_bounds, true_drifts, true_ndts)]
    pred_stats_list = [ez.forward(true_params) for true_params in true_params_list]
    obs_stats_list = [pred_stats.sample(N) for pred_stats in pred_stats_list]

    mean_boundary_weights, mean_drift_weights, mean_ndt_weights, \
        std_boundary_weights, std_drift_weights, std_ndt_weights = \
            qnd_design_matrix_parameter_estimation(obs_stats_list, 
                                                    [N, N, N, N], 
                                                    design_matrix.boundary_nd(), 
                                                    design_matrix.drift_nd(),
                                                    design_matrix.ndt_nd())

    print("\n# True parameters:\n")
    [print(t) for t in true_params_list]
    print("\n# Weights:\n")
    print(f"Boundary weights : {mean_boundary_weights} ({std_boundary_weights})")
    print(f"Drift weights    : {mean_drift_weights} ({std_drift_weights})")
    print(f"NDT weights      : {mean_ndt_weights} ({std_ndt_weights})")

    print(f"\n--------------------------------------------\n")

"""
Run tests
"""
class TestSuite(unittest.TestCase):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--test", action="store_true", help="Run the test suite")
    parser.add_argument("--demo", action="store_true", help="Run the demo")
    parser.add_argument("--simstudy", action="store_true", help="Run the simulation study")
    parser.add_argument("--bayes", action="store_true", help="Run the Bayesian parameter estimation")
    parser.add_argument("--qnd", action="store_true", help="Run the QND parameter estimation")
    
    args = parser.parse_args()
    
    if args.simstudy:
        run_ezdiffusion_simulation_study()
    # if args.demo:
    #     demonstrate_design_matrix_parameter_estimation()
    # if args.bayes:
    #     demonstrate_design_matrix_parameter_estimation()
    if args.qnd:
        demonstrate_qnd_parameter_estimation()
    if args.test:
        unittest.main(argv=[__file__], verbosity=0, failfast=True)
    