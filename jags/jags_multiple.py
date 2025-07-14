#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


import numpy as np
from typing import Tuple, List
import unittest
import argparse
from vendor.ezas.base import ez_equations as ez
from vendor.ezas.classes.moments import Observations
from vendor.ezas.classes.parameters import Parameters
from tqdm import tqdm
from vendor.ezas.utils import b as pretty

import vendor.py2jags.src as p2
from vendor.py2jags.src.mcmc_samples import MCMCSamples

_MAX_R_HAT = 1.1

_DEMO_DEFAULT_PARAMETERS = [
    Parameters(
        boundary=2.0,
        drift=d,
        ndt=0.2
    ) for d in [0.5, 1.5, 2.5]
]

_DEMO_DEFAULT_SAMPLE_SIZE = 1000


def observations_list_to_jags_data(observations: List[Observations]) -> dict:
    """Convert list of Observations objects to JAGS data format."""
    n_conditions = len(observations)
    data_dict = {'n_conditions': n_conditions}
    
    # Create separate variables for each condition
    for i, obs in enumerate(observations):
        idx = i + 1
        data_dict[f'N_{idx}'] = int(obs.sample_size())
        data_dict[f'acc_obs_{idx}'] = int(obs.accuracy())
        data_dict[f'rt_obs_{idx}'] = float(obs.mean_rt())
        data_dict[f'var_obs_{idx}'] = float(obs.var_rt())
    
    return data_dict


def mcmc_samples_to_parameters_list(mcmc_samples: MCMCSamples, n_conditions: int) -> List[Parameters]:
    """Convert MCMC samples to list of Parameters objects."""
    parameters_list = []
    
    for i in range(n_conditions):
        # Parameter names use underscores in the unrolled model
        idx = i + 1
        boundary_name = f'boundary_{idx}'
        drift_name = f'drift_{idx}'
        ndt_name = f'ndt_{idx}'
        
        # Extract statistics for this condition
        boundary_stats = mcmc_samples.stats[boundary_name]
        drift_stats = mcmc_samples.stats[drift_name]
        ndt_stats = mcmc_samples.stats[ndt_name]
        
        # Create Parameters object for this condition
        params = Parameters(
            boundary=float(boundary_stats['mean']),
            drift=float(drift_stats['mean']),
            ndt=float(ndt_stats['mean']),
            boundary_sd=float(boundary_stats['std']),
            drift_sd=float(drift_stats['std']),
            ndt_sd=float(ndt_stats['std']),
            boundary_lower_bound=float(boundary_stats['q025']),
            boundary_upper_bound=float(boundary_stats['q975']),
            drift_lower_bound=float(drift_stats['q025']),
            drift_upper_bound=float(drift_stats['q975']),
            ndt_lower_bound=float(ndt_stats['q025']),
            ndt_upper_bound=float(ndt_stats['q975'])
        )
        
        parameters_list.append(params)
    
    return parameters_list


def bayesian_multiple_parameter_estimation(
    observations: List[Observations], 
    n_samples: int = 2000, 
    n_tune: int = 1000,
    n_chains: int = 4,
    verbosity: int = 2
) -> List[Parameters]:
    """Estimate EZ-diffusion parameters using JAGS given observed statistics.
    
    Args:
        observations: Observed summary statistics (list of Observations)
        n_samples: Number of MCMC samples to draw
        n_tune: Number of tuning steps for MCMC
        n_chains: Number of MCMC chains
        verbosity: Verbosity level
        
    Returns:
        List of estimated Parameters objects (one per condition)
    """
    if not isinstance(observations, list):
        raise TypeError("observations must be a list")
    if not all(isinstance(x, Observations) for x in observations):
        raise TypeError("All elements in observations must be Observations instances")
    
    n_conditions = len(observations)
    
    # Create JAGS model string for multiple conditions (unrolled)
    model_blocks = []
    for i in range(n_conditions):
        idx = i + 1  # JAGS uses 1-based indexing
        model_blocks.append(f"""
        # Condition {idx}
        boundary_{idx} ~ dunif(0.1, 5.0)
        drift_{idx} ~ dnorm(0, 0.25) T(0.01, )
        ndt_{idx} ~ dunif(0.0, 1.0)
        
        # Helper calculations for condition {idx}
        y_{idx} <- exp(-boundary_{idx} * drift_{idx})
        pred_acc_{idx} <- 1 / (y_{idx} + 1)
        
        # Likelihoods for condition {idx}
        acc_obs_{idx} ~ dbin(pred_acc_{idx}, N_{idx})
        
        pred_mean_{idx} <- ndt_{idx} + (boundary_{idx} / (2 * drift_{idx})) * ((1 - y_{idx}) / (1 + y_{idx}))
        pred_var_{idx} <- (boundary_{idx} / (2 * pow(drift_{idx}, 3))) * ((1 - 2*boundary_{idx}*drift_{idx}*y_{idx} - pow(y_{idx}, 2)) / pow((y_{idx} + 1), 2))
        
        rt_obs_{idx} ~ dnorm(pred_mean_{idx}, N_{idx} / pred_var_{idx})
        var_obs_{idx} ~ dgamma((N_{idx}-1)/2, (N_{idx}-1)/(2*pred_var_{idx}))""")
    
    jags_model = f"""
    model {{{''.join(model_blocks)}
    }}
    """
    
    # Data for JAGS
    jags_data = observations_list_to_jags_data(observations)
    
    # Initial values for multiple chains
    jags_inits = []
    for chain in range(n_chains):
        init_dict = {}
        for i in range(n_conditions):
            idx = i + 1
            init_dict[f'boundary_{idx}'] = 1.5 + 0.2 * chain + 0.1 * i
            init_dict[f'drift_{idx}'] = 0.5 + 0.2 * chain + 0.1 * i  
            init_dict[f'ndt_{idx}'] = 0.25 + 0.05 * chain + 0.02 * i
        jags_inits.append(init_dict)
    
    # Monitor parameters for all conditions
    monitor_params = []
    for i in range(n_conditions):
        idx = i + 1
        monitor_params.extend([f'boundary_{idx}', f'drift_{idx}', f'ndt_{idx}'])
    
    # Initialize JAGS model
    mcmc_samples = p2.run_jags(
        model_string=jags_model,
        data_dict=jags_data,
        nchains=n_chains,
        init=jags_inits,
        nadapt=n_tune,
        nburnin=n_tune,
        nsamples=n_samples,
        monitorparams=monitor_params,
        verbosity=verbosity
    )
    
    # Print summary if verbose
    if verbosity >= 2:
        mcmc_samples.summary()
    
    # Check convergence
    if verbosity >= 1:
        mcmc_samples.show_diagnostics()
    
    # Convert to list of Parameters objects using the helper function
    return mcmc_samples_to_parameters_list(mcmc_samples, n_conditions)


"""
Simulation
"""
def simulation(repetitions: int|None = None):
    """
    Run a comprehensive demonstration of Bayesian parameter recovery accuracy.
    """
    
    if repetitions is None:
        repetitions = 1000

    if not isinstance(repetitions, int):
        raise TypeError("repetitions must be of type int")

    true_parameters = _DEMO_DEFAULT_PARAMETERS
    sample_size = _DEMO_DEFAULT_SAMPLE_SIZE
    n_conditions = len(true_parameters)

    # Initialize a progress bar
    progress_bar = tqdm(total=repetitions, 
                        desc="Simulation progress")

    boundary_coverage = 0
    drift_coverage = 0
    ndt_coverage = 0
    total_coverage = 0

    for _ in range(repetitions):
        moments = ez.forward(true_parameters)
        observations = [m.sample(sample_size=sample_size) for m in moments]
        estimated_parameters = bayesian_multiple_parameter_estimation(
            observations,
            n_samples=1000,
            n_tune=1000,
            n_chains=4,
            verbosity=0
        )

        for i, p in enumerate(true_parameters):
            boundary_in_bounds, drift_in_bounds, ndt_in_bounds = p.is_within_bounds_of(estimated_parameters[i])
            boundary_coverage += boundary_in_bounds
            drift_coverage    += drift_in_bounds
            ndt_coverage      += ndt_in_bounds
            total_coverage    += boundary_in_bounds and drift_in_bounds and ndt_in_bounds

        progress_bar.update(1)

    progress_bar.close()

    boundary_coverage *= (100 / (repetitions * n_conditions))
    drift_coverage    *= (100 / (repetitions * n_conditions))
    ndt_coverage      *= (100 / (repetitions * n_conditions))
    total_coverage    *= (100 / (repetitions * n_conditions))

    print("Coverage:")
    print(f" > Boundary : {boundary_coverage:5.1f}%  (should be ≈ 95%)")
    print(f" > Drift    : {drift_coverage:5.1f}%  (should be ≈ 95%)")
    print(f" > NDT      : {ndt_coverage:5.1f}%  (should be ≈ 95%)")
    print(f" > Total    : {total_coverage:5.1f}%  (should be ≈ {100*(.95**3):.1f}%)")

    # Print brief report to file, with simulation settings and results
    this_path = os.path.dirname(os.path.abspath(__file__))
    report_file = os.path.join(this_path, "jags_multiple_simulation_report.txt")
    with open(report_file, "w") as f:
        f.write(f"Simulation settings:\n")
        f.write(f" > Repetitions :  {repetitions}\n")
        f.write(f" > Sample size :  {sample_size}\n")
        f.write(f" > Conditions  :  {n_conditions}\n")
        f.write(f" > True parameters:\n")
        [f.write(f"  * {str(p)}\n") for p in true_parameters]
        f.write(f"Results:\n")
        f.write(f" > Boundary coverage :  {boundary_coverage:5.1f}%\n")
        f.write(f" > Drift coverage    :  {drift_coverage:5.1f}%\n")
        f.write(f" > NDT coverage      :  {ndt_coverage:5.1f}%\n")
        f.write(f" > Total coverage    :  {total_coverage:5.1f}%\n")


"""
Demo
"""
def demo():

    # Set some parameters
    parameters = [
        Parameters(
            boundary=2.0,
            drift=d,
            ndt=0.2
        ) for d in [0.5, 1.5, 2.5]
    ]
    
    sample_size = 1000
    
    print("True parameters:")
    [print(p) for p in parameters]
    
    # Compute the moments
    moments = [ez.forward(p) for p in parameters]
    print("True moments:")
    [print(m) for m in moments]
    
    # Sample the moments
    observations = [m.sample(sample_size=sample_size) for m in moments]
    print("Observations:")
    [print(o) for o in observations]
    
    # Estimate the parameters
    estimated_parameters = bayesian_multiple_parameter_estimation(observations)
    
    print("Estimated parameters:")
    [print(p) for p in estimated_parameters]
    
    # Check if the true parameters are in the bounds of the estimated parameters and print colorful emoji checkmark or X
    for i, p in enumerate(parameters):
        boundary_in_bounds, drift_in_bounds, ndt_in_bounds = p.is_within_bounds_of(estimated_parameters[i])
        print(f"Boundary {i} in bounds : {pretty(boundary_in_bounds)}")
        print(f"Drift {i} in bounds    : {pretty(drift_in_bounds)}")
        print(f"NDT {i} in bounds      : {pretty(ndt_in_bounds)}")

"""
Test suite
"""
class TestSuite(unittest.TestCase):    
    def test_bayesian_multiple_parameter_estimation(self):
        demo()

"""
Main
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian Multiple Parameter Estimation")
    parser.add_argument("--test", action="store_true", help="Run the test suite")
    parser.add_argument("--demo", action="store_true", help="Run the demo")
    parser.add_argument("--simulation", action="store_true", help="Run the simulation study")
    parser.add_argument("--repetitions", type=int, default=1000, help="Number of repetitions for the simulation study")
    args = parser.parse_args()
    
    if args.test:
        unittest.main(argv=['first-arg-is-ignored'], exit=False)
        sys.exit(0)
    
    if args.demo:
        demo()
    elif args.simulation:
        simulation(repetitions=args.repetitions)
    else:
        print("Use --demo for basic demonstration or --simulation for simulation study") 