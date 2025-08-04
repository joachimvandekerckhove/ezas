#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import argparse
import pandas as pd
import unittest
import itertools

from vendor.ezas.classes import Observations, DesignMatrix
from vendor.ezas.jags.jags_beta_weights import bayesian_design_matrix_parameter_estimation
from vendor.ezas.utils import b as pretty
import vendor.py2jags.src as p2
from vendor.py2jags.src.mcmc_samples import MCMCSamples

_EPSILON = 0.1
_ALPHA = 0.05

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def create_two_condition_design(
    drift_difference: float = 0.0, 
    base_drift: float = 0.5,
    boundary: float = 1.0,
    ndt: float = 0.2
) -> DesignMatrix:
    """
    Create a simple two-condition design matrix for testing drift differences.
    There are two cells, one for each condition.
    
    Args:
        drift_difference: Difference in drift rate between conditions (condition 2 - condition 1)
        base_drift: Base drift rate for condition 1
        boundary: Boundary parameter (same for both conditions)
        ndt: Non-decision time (same for both conditions)
        
    Returns:
        DesignMatrix with two conditions
    """
    
    # Design matrices: [intercept, condition_difference] 
    boundary_design = np.ones((2, 1))
    ndt_design      = np.ones((2, 1))
    drift_design    = np.array([[1, 0],   # Condition 1: base_drift + 0*drift_difference
                                [1, 1]])  # Condition 2: base_drift + 1*drift_difference
    
    # Beta weights: [intercept, condition_difference]
    boundary_weights = np.array([boundary])
    drift_weights    = np.array([base_drift, drift_difference])
    ndt_weights      = np.array([ndt])
    
    return DesignMatrix(
        boundary_design  = boundary_design,
        drift_design     = drift_design,
        ndt_design       = ndt_design,
        boundary_weights = boundary_weights,
        drift_weights    = drift_weights,
        ndt_weights      = ndt_weights
    )
    

def run_drift_difference_simulation(
    true_drift_differences: List[float],
    sample_sizes: List[int] = [100, 200],
    n_simulations: int = 100,
    n_samples: int = 2000,
    n_tune: int = 1000,
    n_chains: int = 4,
    epsilon: float = _EPSILON) -> dict:
    """
    Run simulation study for JAGS drift difference inference.
    
    Args:
        true_drift_differences: List of true drift differences to test
        sample_sizes: List of sample sizes per condition
        n_simulations: Number of simulation repetitions
        n_samples: Number of MCMC samples per inference
        n_tune: Number of tuning iterations
        n_chains: Number of MCMC chains
        epsilon: Threshold for "close to zero"
        
    Returns:
        Dictionary with simulation results
    """
    # Make the factorial design of true drift differences and sample sizes
    design = list(itertools.product(true_drift_differences, sample_sizes))
    
    # Initialize results dictionary
    results = {
        'true_drift_differences'   : [d[0] for d in design],
        'sample_sizes'             : [d[1] for d in design],
        'epsilon'                  : epsilon,
        'null_rejected'            : [],
        'null_retained'            : []
    }
    
    for i, (drift_diff, N) in enumerate(design):
        print(f"# Testing drift difference: {drift_diff} and sample size: {N}")
                
        # Create design matrix with this drift difference
        design_matrix = create_two_condition_design(
            drift_difference = drift_diff
        )
        
        working_matrix = DesignMatrix(
            boundary_design = design_matrix.boundary_design(),
            drift_design    = design_matrix.drift_design(),
            ndt_design      = design_matrix.ndt_design()
        )
        
        # Run simulations
        null_rejected_count      = 0
        null_retained_count      = 0
            
        for _ in tqdm(range(n_simulations), desc=f"N={N}", leave=False):
            # Generate data
            observations = design_matrix.sample([N, N])
                
            # Perform inference
            est_beta, est_params, mcmc_samples = bayesian_design_matrix_parameter_estimation(
                observations   = observations,
                working_matrix = working_matrix,
                n_samples      = n_samples,
                n_tune         = n_tune,
                n_chains       = n_chains,
                verbosity      = 0
            )
                
            # Record results
            if est_beta.beta_drift_lower()[1] * est_beta.beta_drift_upper()[1] > 0:
                null_rejected_count += 1
            else:
                null_retained_count += 1
            
        # Calculate summary statistics
        null_rejected = null_rejected_count / n_simulations
        null_retained = null_retained_count / n_simulations
        
        results['null_rejected'].append(null_rejected)
        results['null_retained'].append(null_retained)
    
    return results

def create_inference_table(
    simulation_results: dict, 
    display: bool = True
) -> pd.DataFrame:
    """Create a table of inference results."""
    print(simulation_results)
    table = pd.DataFrame(simulation_results)
    if display:
        print(table)
    return table


def create_inference_figure(
    simulation_results: dict
) -> plt.Figure:
    """Create comprehensive figure for inference simulation results with separate panels for each sample size."""
    # Get unique drift differences and sample sizes
    unique_diffs = sorted(list(set(simulation_results['true_drift_differences'])))
    unique_sizes = sorted(list(set(simulation_results['sample_sizes'])))
    
    # Create subplots - one panel for each sample size
    n_panels = len(unique_sizes)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6), sharey=True)
    
    # Handle case where there's only one panel
    if n_panels == 1:
        axes = [axes]
    
    # Define colors for different probability types
    colors = {
        'rejected': '#e74c3c',      # Red
        'retained': '#3498db',      # Blue  
        'approx_zero': '#2ecc71',   # Green
        'less_than_zero': '#f39c12' # Orange
    }
    
    # Plot each sample size in a separate panel
    for panel_idx, N in enumerate(unique_sizes):
        ax = axes[panel_idx]
        
        # Get data for this sample size
        indices = [i for i, size in enumerate(simulation_results['sample_sizes']) if size == N]
        
        if len(indices) > 0:
            drift_vals = [simulation_results['true_drift_differences'][i] for i in indices]
            rejected_vals = [simulation_results['null_rejected'][i] for i in indices]
            retained_vals = [simulation_results['null_retained'][i] for i in indices]
            
            # Sort by drift difference for clean lines
            sorted_data = sorted(zip(drift_vals, rejected_vals, retained_vals, 
                                   ))
            drift_vals, rejected_vals, retained_vals = zip(*sorted_data)
            
            # Plot lines with markers for each probability type
            ax.plot(drift_vals, rejected_vals, 'o-', color=colors['rejected'], 
                   linewidth=2, markersize=6, alpha=0.8, label='Rejected')
            ax.plot(drift_vals, retained_vals, 's-', color=colors['retained'], 
                   linewidth=2, markersize=6, alpha=0.8, label='Retained')
        
        # Customize each panel
        ax.set_xlabel('True Drift Difference')
        ax.set_title(f'N = {N}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add legend to the first panel only
        if panel_idx == 0:
            ax.legend(loc='upper right')
    
    # Set y-label only for the leftmost panel
    axes[0].set_ylabel('Probability')
    
    # Add overall title
    fig.suptitle('JAGS Drift Difference Inference Results', fontsize=14, y=0.98)
    
    plt.tight_layout()
    return fig

def demo(
    true_drift_diff: float = 0.20
) -> None:
    """Run a demonstration of the JAGS drift difference inference method."""
    print("=== JAGS Drift Difference Inference Demo ===\n")
    
    # Test with a small drift difference
    sample_size = 80
    epsilon = 0.10
    n_samples = 2000
    n_tune = 1000
    n_chains = 4
    
    print(f"True drift difference: {true_drift_diff}")
    print(f"Sample size per condition: {sample_size}")
    print(f"MCMC samples: {n_samples}")
    print(f"MCMC tuning: {n_tune}")
    print(f"MCMC chains: {n_chains}")
    
    # Create design matrix
    design_matrix = create_two_condition_design(drift_difference=true_drift_diff)
    n_cells = design_matrix.boundary_design().shape[0]
    print(f"\nDesign matrix created with {n_cells} cells")
    
    # Generate data
    print("\nGenerating data...")
    observations = design_matrix.sample(n_cells * [sample_size])
    
    # Perform inference
    print("Performing JAGS inference...")
    start_time = time.time()
    est_beta, est_params, mcmc_samples = bayesian_design_matrix_parameter_estimation(
        observations = observations, 
        working_matrix = design_matrix,
        n_samples = n_samples,
        n_tune = n_tune,
        n_chains = n_chains,
        verbosity = 2
    )
    elapsed_time = time.time() - start_time
    
    print("# True parameter values:")
    print(f"  Drift difference: {true_drift_diff}")
    print(f"  Boundary: {design_matrix.boundary_weights()}")
    print(f"  Drift: {design_matrix.drift_weights()}")
    print(f"  NDT: {design_matrix.ndt_weights()}")
    
    print(f"\nInference completed in {elapsed_time:.2f} seconds")
    print(f"\nResults:")
    print(f"  Mean beta weight: {est_beta.beta_drift_mean()}")
    print(f"  Standard error: {est_beta.beta_drift_sd()}")
    print(f"  95% CI: [{est_beta.beta_drift_lower()}, {est_beta.beta_drift_upper()}]")
    print(f"  Max R-hat: {mcmc_samples.max_rhat()}")
    print(f"  Converged: {pretty(mcmc_samples.converged())}")
    
    # Inference results
    beta_drift_lower = est_beta.beta_drift_lower()[1]
    beta_drift_upper = est_beta.beta_drift_upper()[1]
    
    print(f"\nInference:")
    print(f"  P(|drift_diff| < {epsilon}): {beta_drift_lower * beta_drift_upper > 0}")
    print(f"  P(drift_diff < 0): {beta_drift_lower < 0}")
    
    # Check if true value is in CI
    ci_contains_true = (beta_drift_lower <= true_drift_diff <= beta_drift_upper)
    print(f"  True value in CI: {pretty(ci_contains_true)}")
    
    # Decision based on alpha threshold
    decision = "Reject H0" if beta_drift_lower * beta_drift_upper > 0 else "Retain H0"
    print(f"  Decision (α = {_ALPHA}): {decision}")

def simulation(repetitions: int = 100) -> None:
    """Run a simulation study of the JAGS drift difference inference method."""
    print("=== JAGS Drift Difference Inference Simulation ===\n")
    
    # Test different true drift differences
    true_drift_differences = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    sample_sizes = [100, 200]
    n_simulations = repetitions
    n_samples = 2000
    n_tune = 1000
    n_chains = 4
    
    print(f"True drift differences: {true_drift_differences}")
    print(f"Sample sizes: {sample_sizes}")
    print(f"Simulations per condition: {n_simulations}")
    print(f"MCMC samples per inference: {n_samples}")
    print(f"MCMC tuning per inference: {n_tune}")
    print(f"MCMC chains per inference: {n_chains}")
    
    # Run simulation
    print("\nRunning simulation...")
    start_time = time.time()
    results = run_drift_difference_simulation(
        true_drift_differences = true_drift_differences,
        sample_sizes           = sample_sizes,
        n_simulations          = n_simulations,
        n_samples              = n_samples,
        n_tune                 = n_tune,
        n_chains               = n_chains,
        epsilon                = _EPSILON
    )
    elapsed_time = time.time() - start_time
    
    print(f"\nSimulation completed in {elapsed_time/60:.1f} minutes")
        
    df = pd.DataFrame(results)
    print(df)
    
    # Create and save figure
    fig = create_inference_figure(results)
    fig.savefig("jags_drift_difference_inference_demo.png", dpi=300, bbox_inches='tight')
    print(f"\nFigure saved as jags_drift_difference_inference_demo.png")


# Test suite

class TestSuite(unittest.TestCase):
    def test_jags_drift_difference_inference(self):
        """Test the JAGS drift difference inference method."""
        # Create simple two-condition design
        design_matrix = create_two_condition_design(drift_difference=0.3)
        observations = design_matrix.sample([50, 50])
        
        # Run inference
        est_beta, est_params, mcmc_samples = bayesian_design_matrix_parameter_estimation(
            observations = observations,
            n_samples = 500,
            n_tune = 500,
            n_chains = 2,
            verbosity = 0
        )
        
        # Check that all required keys are present
        required_keys = ['beta_drift_mean', 'beta_drift_sd', 'beta_drift_lower', 'beta_drift_upper', 'max_rhat', 'converged']
        for key in required_keys:
            self.assertIn(key, est_beta)
        
        # Check that CI bounds are ordered correctly
        self.assertLess(est_beta.beta_drift_lower(), est_beta.beta_drift_upper())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JAGS Drift Difference Inference")
    parser.add_argument("--test", action="store_true", help="Run the test suite")
    parser.add_argument("--demo", action="store_true", help="Run the demo")
    parser.add_argument("--simulation", action="store_true", help="Run the simulation study")
    parser.add_argument("--repetitions", type=int, default=100, help="Number of repetitions for the simulation study")
    parser.add_argument("--drift-diff", type=float, default=0.20, help="Set the true drift difference for the demo")
    args = parser.parse_args()
    
    if args.test:
        unittest.main(argv=['first-arg-is-ignored'], exit=False)
        sys.exit(0)
    
    if args.demo:
        demo(true_drift_diff=args.drift_diff)
    elif args.simulation:
        simulation(repetitions=args.repetitions)
    else:
        print("Use --demo for basic demonstration or --simulation for simulation study") 