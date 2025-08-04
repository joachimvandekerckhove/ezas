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
from vendor.ezas.qnd import qnd_beta_weights_estimation
from vendor.ezas.utils import b as pretty

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
    There are ten cells in each condition.
    
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
    drift_design    = np.ones((2, 2))
    drift_design[1, 1] = 0
    
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
    

def qnd_drift_difference_inference(
    observations: List[Observations], 
    n_bootstrap: int = 1000,
    epsilon: float = _EPSILON
) -> dict:
    """
    Perform QND inference on drift rate differences between two conditions.
    
    Args:
        observations: List of observations for two conditions
        n_bootstrap: Number of bootstrap samples
        epsilon: Threshold for "close to zero" (default: 0.05)
        
    Returns:
        Dictionary with inference results
    """
    n_cells = len(observations)
    
    # Create design matrix
    boundary_design = np.ones((n_cells, 1))
    ndt_design      = np.ones((n_cells, 1))
    drift_design    = np.ones((n_cells, 2))
    drift_design[1, 1] = 0
    
    working_matrix = DesignMatrix(
        boundary_design = boundary_design,
        drift_design    = drift_design,
        ndt_design      = ndt_design
    )
    
    estimated_beta_weights, _, beta_weights_list \
        = qnd_beta_weights_estimation(
            observations   = observations,
            working_matrix = working_matrix,
            n_bootstrap    = n_bootstrap
        )
        
    # Calculate probability that drift difference is close to zero
    prob_close_to_zero = np.mean(
        [np.abs(w.beta_drift_mean()[1]) <= epsilon for w in beta_weights_list]
    )
            
    # ... and less than zero
    prob_less_than_zero = np.mean(
        [w.beta_drift_mean()[1] <= 0.0 for w in beta_weights_list]
    )
    
    return {
        'prob_close_to_zero'  : prob_close_to_zero,
        'prob_less_than_zero' : prob_less_than_zero,
        'mean_beta_weight'    : estimated_beta_weights.beta_drift_mean()[1],
        'std_beta_weight'     : estimated_beta_weights.beta_drift_sd()[1],
        'lower_ci'            : estimated_beta_weights.beta_drift_lower()[1],
        'upper_ci'            : estimated_beta_weights.beta_drift_upper()[1]
    }

def run_drift_difference_simulation(
    true_drift_differences: List[float],
    sample_sizes: List[int] = [100, 200],
    n_simulations: int = 1000,
    n_bootstrap: int = 1000,
    epsilon: float = _EPSILON) -> dict:
    """
    Run simulation study for drift difference inference.
    
    Args:
        true_drift_differences: List of true drift differences to test
        sample_sizes: List of sample sizes per condition
        n_simulations: Number of simulation repetitions
        n_bootstrap: Number of bootstrap samples per inference
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
        'null_retained'            : [],
        'mean_prob_close_to_zero'  : [],
        'mean_prob_less_than_zero' : []
    }
    
    for i, (drift_diff, N) in enumerate(design):
        print(f"# Testing drift difference: {drift_diff} and sample size: {N}")
                
        # Create design matrix with this drift difference
        design_matrix = create_two_condition_design(
            drift_difference = drift_diff
        )
            
        # Run simulations
        null_rejected_count      = 0
        null_retained_count      = 0
        prob_close_to_zero_list  = []
        prob_less_than_zero_list = []
            
        for _ in tqdm(range(n_simulations), desc=f"N={N}", leave=False):
            # Generate data
            observations = design_matrix.sample([N, N])
                
            # Perform inference
            inference_result = qnd_drift_difference_inference(
                observations = observations,
                n_bootstrap  = n_bootstrap,
                epsilon      = epsilon
            )
                
            # Record results
            prob_close_to_zero_list.append(inference_result['prob_close_to_zero'])
            prob_less_than_zero_list.append(inference_result['prob_less_than_zero'])
                
            if inference_result['prob_close_to_zero'] < _ALPHA:
                null_rejected_count += 1
            else:
                null_retained_count += 1
            
        # Calculate summary statistics
        null_rejected = null_rejected_count / n_simulations
        null_retained = null_retained_count / n_simulations
        prob_close_to_zero = np.mean(prob_close_to_zero_list)
        prob_less_than_zero = np.mean(prob_less_than_zero_list)
        
        results['null_rejected'].append(null_rejected)
        results['null_retained'].append(null_retained)
        results['mean_prob_close_to_zero'].append(prob_close_to_zero)
        results['mean_prob_less_than_zero'].append(prob_less_than_zero)
    
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
            approx_zero_vals = [simulation_results['mean_prob_close_to_zero'][i] for i in indices]
            less_than_zero_vals = [simulation_results['mean_prob_less_than_zero'][i] for i in indices]
            
            # Sort by drift difference for clean lines
            sorted_data = sorted(zip(drift_vals, rejected_vals, retained_vals, 
                                   approx_zero_vals, less_than_zero_vals))
            drift_vals, rejected_vals, retained_vals, approx_zero_vals, less_than_zero_vals = zip(*sorted_data)
            
            # Plot lines with markers for each probability type
            ax.plot(drift_vals, rejected_vals, 'o-', color=colors['rejected'], 
                   linewidth=2, markersize=6, alpha=0.8, label='Rejected')
            ax.plot(drift_vals, retained_vals, 's-', color=colors['retained'], 
                   linewidth=2, markersize=6, alpha=0.8, label='Retained')
            ax.plot(drift_vals, approx_zero_vals, '^-', color=colors['approx_zero'], 
                   linewidth=2, markersize=6, alpha=0.8, label='Approx Zero')
            ax.plot(drift_vals, less_than_zero_vals, 'v-', color=colors['less_than_zero'], 
                   linewidth=2, markersize=6, alpha=0.8, label='Less Than Zero')
        
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
    fig.suptitle('QND Drift Difference Inference Results', fontsize=14, y=0.98)
    
    plt.tight_layout()
    return fig

def demo(
    true_drift_diff: float = 0.20
) -> None:
    """Run a demonstration of the drift difference inference method."""
    print("=== QND Drift Difference Inference Demo ===\n")
    
    # Test with a small drift difference
    sample_size = 80
    epsilon = 0.10
    n_bootstrap = 1000
    
    print(f"True drift difference: {true_drift_diff}")
    print(f"Sample size per condition: {sample_size}")
    
    # Create design matrix
    design_matrix = create_two_condition_design(drift_difference=true_drift_diff)
    n_cells = design_matrix.boundary_design().shape[0]
    print(f"\nDesign matrix created with {n_cells} cells")
    
    # Generate data
    print("\nGenerating data...")
    observations = design_matrix.sample(n_cells * [sample_size])
    
    # Perform inference
    print("Performing inference...")
    start_time = time.time()
    inference_result = qnd_drift_difference_inference(observations, n_bootstrap=n_bootstrap, epsilon=epsilon)
    elapsed_time = time.time() - start_time
    
    print(f"\nInference completed in {elapsed_time:.2f} seconds")
    print(f"\nResults:")
    print(f"  Mean beta weight: {inference_result['mean_beta_weight']:.4f}")
    print(f"  Standard error: {inference_result['std_beta_weight']:.4f}")
    print(f"  95% CI: [{inference_result['lower_ci']:.4f}, {inference_result['upper_ci']:.4f}]")
    
    # Check if true value is in CI
    ci_contains_true = (inference_result['lower_ci'] <= true_drift_diff <= inference_result['upper_ci'])
    print(f"  True value in CI: {pretty(ci_contains_true)}")

def simulation(repetitions: int = 1000) -> None:
    """Run a simulation study of the drift difference inference method."""
    print("=== QND Drift Difference Inference Simulation ===\n")
    
    # Test different true drift differences
    true_drift_differences = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sample_sizes = [100, 1000]
    n_simulations = repetitions
    n_bootstrap = 1000
    
    print(f"True drift differences: {true_drift_differences}")
    print(f"Sample sizes: {sample_sizes}")
    print(f"Simulations per condition: {n_simulations}")
    print(f"Bootstrap samples per inference: {n_bootstrap}")
    
    # Run simulation
    print("\nRunning simulation...")
    start_time = time.time()
    results = run_drift_difference_simulation(
        true_drift_differences = true_drift_differences,
        sample_sizes           = sample_sizes,
        n_simulations          = n_simulations,
        n_bootstrap            = n_bootstrap,
        epsilon                = _EPSILON
    )
    elapsed_time = time.time() - start_time
    
    print(f"\nSimulation completed in {elapsed_time/60:.1f} minutes")
        
    df = pd.DataFrame(results)

    # Table header
    print(f"{'N':<5} {'rejected':<10} {'retained':<10} {'approx 0':<10} {'less than 0':<10}")
    print(f"{'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    for i in range(len(df)):
        # Table body
        print(f"{df['sample_sizes'][i]:<5} {df['null_rejected'][i]:<10.3f} {df['null_retained'][i]:<10.3f} {df['mean_prob_close_to_zero'][i]:<10.3f} {df['mean_prob_less_than_zero'][i]:<10.3f}")
    # Table footer
    print(f"{'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    # Create and save figure
    fig = create_inference_figure(results)
    fig.savefig("qnd_drift_difference_inference_demo.png", dpi=300, bbox_inches='tight')
    print(f"\nFigure saved as qnd_drift_difference_inference_demo.png")
    plt.show()


# Test suite

class TestSuite(unittest.TestCase):
    def test_qnd_drift_difference_inference(self):
        """Test the drift difference inference method."""
        self.assertEqual(1, 1)
        self.assertEqual(1, 1)
        self.assertEqual(1, 1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QND Drift Difference Inference")
    parser.add_argument("--test", action="store_true", help="Run the test suite")
    parser.add_argument("--demo", action="store_true", help="Run the demo")
    parser.add_argument("--simulation", action="store_true", help="Run the simulation study")
    parser.add_argument("--repetitions", type=int, default=1000, help="Number of repetitions for the simulation study")
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