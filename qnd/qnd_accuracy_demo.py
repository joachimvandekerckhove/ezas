#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import argparse
from tqdm import tqdm, trange
import time
import pickle

from vendor.ezas.base import ez_equations as ez
from vendor.ezas.classes.moments import Observations
from vendor.ezas.classes.parameters import Parameters
from vendor.ezas.qnd.qnd_single import qnd_single_estimation
from vendor.ezas.qnd.qnd_beta_weights import qnd_beta_weights_estimation, _EXAMPLE_DESIGN_MATRIX
from vendor.ezas.classes.design_matrix import DesignMatrix

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def run_single_parameter_simulation(
    true_params: Parameters,
    sample_sizes: List[int],
    n_simulations: int = 1000,
    n_bootstrap: int = 1000
) -> dict:
    """
    Run simulation study for single parameter estimation.
    
    Args:
        true_params: True parameters to recover
        sample_sizes: List of sample sizes to test
        n_simulations: Number of simulation repetitions
        n_bootstrap: Number of bootstrap repetitions for QND
        
    Returns:
        Dictionary with simulation results
    """
    results = {
        'sample_sizes': sample_sizes,
        'boundary_coverage': [],
        'drift_coverage': [],
        'ndt_coverage': [],
        'total_coverage': [],
        'mean_elapsed_time': [],
    }
    d = 3  # number of parameters
    for N in sample_sizes:
        boundary_covered = 0
        drift_covered = 0
        ndt_covered = 0
        total_covered = 0
        elapsed_times = []
        for _ in trange(n_simulations, desc=f"(N = {N:3d})"):
            moments = ez.forward(true_params)
            observations = moments.sample(N)
            start = time.time()
            estimated_params = qnd_single_estimation(observations, n_repetitions=n_bootstrap)
            elapsed_times.append(time.time() - start)
            boundary_in_bounds, drift_in_bounds, ndt_in_bounds = true_params.is_within_bounds_of(estimated_params)
            if boundary_in_bounds: boundary_covered += 1
            if drift_in_bounds: drift_covered += 1
            if ndt_in_bounds: ndt_covered += 1
            if boundary_in_bounds and drift_in_bounds and ndt_in_bounds: total_covered += 1
        results['boundary_coverage'].append(boundary_covered / n_simulations)
        results['drift_coverage'].append(drift_covered / n_simulations)
        results['ndt_coverage'].append(ndt_covered / n_simulations)
        results['total_coverage'].append(total_covered / n_simulations)
        results['mean_elapsed_time'].append(np.mean(elapsed_times))
    results['theoretical_joint_coverage'] = 0.95 ** d
    return results

def run_design_matrix_simulation(
    design_matrix: DesignMatrix,
    sample_sizes: List[int],
    n_simulations: int = 500
) -> dict:
    """
    Run simulation study for design matrix estimation.
    
    Args:
        design_matrix: Design matrix to test
        sample_sizes: List of sample sizes per condition
        n_simulations: Number of simulation repetitions
        
    Returns:
        Dictionary with simulation results
    """
    results = {
        'sample_sizes': sample_sizes,
        'boundary_weight_coverage': [],
        'drift_weight_coverage': [],
        'ndt_weight_coverage': [],
        'boundary_weight_bias': [],
        'drift_weight_bias': [],
        'ndt_weight_bias': []
    }
    
    true_beta_weights = design_matrix.get_beta_weights()
    true_boundary_weights = true_beta_weights.beta_boundary_mean()
    true_drift_weights = true_beta_weights.beta_drift_mean()
    true_ndt_weights = true_beta_weights.beta_ndt_mean()
    
    for N in tqdm(sample_sizes, desc="Testing design matrix"):
        working_matrix = DesignMatrix(
            boundary_design=design_matrix.boundary_design(),
            drift_design=design_matrix.drift_design(),
            ndt_design=design_matrix.ndt_design()
        )
        
        boundary_covered = np.zeros(len(true_boundary_weights))
        drift_covered = np.zeros(len(true_drift_weights))
        ndt_covered = np.zeros(len(true_ndt_weights))
        
        boundary_biases = []
        drift_biases = []
        ndt_biases = []
        
        for _ in range(n_simulations):
            # Generate data
            observations = design_matrix.sample([N] * design_matrix.boundary_design().shape[0])
            
            # Estimate parameters
            estimated_beta_weights, _ = qnd_beta_weights_estimation(observations, working_matrix)
            
            # Check coverage for each weight
            est_boundary_weights = estimated_beta_weights.beta_boundary_mean()
            est_drift_weights = estimated_beta_weights.beta_drift_mean()
            est_ndt_weights = estimated_beta_weights.beta_ndt_mean()
            
            est_boundary_lower = estimated_beta_weights.beta_boundary_lower()
            est_boundary_upper = estimated_beta_weights.beta_boundary_upper()
            est_drift_lower = estimated_beta_weights.beta_drift_lower()
            est_drift_upper = estimated_beta_weights.beta_drift_upper()
            est_ndt_lower = estimated_beta_weights.beta_ndt_lower()
            est_ndt_upper = estimated_beta_weights.beta_ndt_upper()
            
            for i in range(len(true_boundary_weights)):
                if est_boundary_lower[i] <= true_boundary_weights[i] <= est_boundary_upper[i]:
                    boundary_covered[i] += 1
                if est_drift_lower[i] <= true_drift_weights[i] <= est_drift_upper[i]:
                    drift_covered[i] += 1
                if est_ndt_lower[i] <= true_ndt_weights[i] <= est_ndt_upper[i]:
                    ndt_covered[i] += 1
            
            boundary_biases.append(est_boundary_weights - true_boundary_weights)
            drift_biases.append(est_drift_weights - true_drift_weights)
            ndt_biases.append(est_ndt_weights - true_ndt_weights)
        
        results['boundary_weight_coverage'].append(boundary_covered / n_simulations)
        results['drift_weight_coverage'].append(drift_covered / n_simulations)
        results['ndt_weight_coverage'].append(ndt_covered / n_simulations)
        
        results['boundary_weight_bias'].append(np.mean(boundary_biases, axis=0))
        results['drift_weight_bias'].append(np.mean(drift_biases, axis=0))
        results['ndt_weight_bias'].append(np.mean(ndt_biases, axis=0))
    
    return results

def create_calibration_figure(single_results: dict):
    base_fontsize = 13
    sample_sizes = single_results['sample_sizes']
    theoretical_joint = single_results['theoretical_joint_coverage']
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax = axes[0]
    markersize = 14
    linewidth = 4
    ax.plot(sample_sizes, single_results['boundary_coverage'], 'o-', label='Boundary', linewidth=linewidth,
            markersize=markersize, markerfacecolor='white', markeredgewidth=2, markeredgecolor='red', color='red')
    ax.plot(sample_sizes, single_results['drift_coverage'], 's-', label='Drift', linewidth=linewidth,
            markersize=markersize, markerfacecolor='white', markeredgewidth=2, markeredgecolor='green', color='green')
    ax.plot(sample_sizes, single_results['ndt_coverage'], '^-', label='NDT', linewidth=linewidth,
            markersize=markersize, markerfacecolor='white', markeredgewidth=2, markeredgecolor='blue', color='blue')
    ax.plot(sample_sizes, single_results['total_coverage'], 'd-', label='All Parameters (joint)', linewidth=linewidth,
            markersize=markersize, markerfacecolor='white', markeredgewidth=2, markeredgecolor='black', color='black')
    ax.axhline(y=0.95, color='red', linestyle='--', linewidth=2)
    ax.axhline(y=theoretical_joint, color='purple', linestyle=':', linewidth=2)
    ax.set_xticks(sample_sizes, labels=sample_sizes, fontsize=base_fontsize*1.2)
    yrange = np.arange(0.85, 1.04, 0.05)
    ax.set_ylim(yrange[0], 1.01)
    ax.set_yticks(yrange, labels=[f"{x:.2f}" for x in yrange], fontsize=base_fontsize*1.2)
    ax.set_xlabel('Sample Size', fontsize=base_fontsize*1.3, fontweight='bold')
    ax.set_ylabel('Coverage Rate', fontsize=base_fontsize*1.3, fontweight='bold')
    ax.set_title('Calibration of Bootstrap Credible Intervals', fontsize=base_fontsize*1.3, fontweight='bold')
    ax.legend(fontsize=base_fontsize*1.3, loc=(.3, .25))
    ax.grid(True, alpha=0.3)

    # Elapsed time subplot
    ax2 = axes[1]
    ax2.plot(sample_sizes, 1000 * np.array(single_results['mean_elapsed_time']), 'o-', color='black', linewidth=2)
    ax2.set_ylim(0, 1000 * np.max(single_results['mean_elapsed_time']) * 1.1)
    ax2.set_xticks(sample_sizes, labels=sample_sizes, fontsize=base_fontsize)
    yrange = np.arange(0, 1000 * np.max(single_results['mean_elapsed_time']) * 1.1, 5)
    ax2.set_yticks(yrange, labels=[f"{x:.0f}" for x in yrange], fontsize=base_fontsize)
    ax2.set_xlabel('Sample Size', fontsize=base_fontsize*1.2)
    ax2.set_ylabel('Mean Elapsed Time (ms)', fontsize=base_fontsize*1.2)
    ax2.set_title('Mean Bootstrap Estimation Time', fontsize=base_fontsize*1.3)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def demo():
    """
    Run a comprehensive demonstration of QND estimation accuracy.
    """
    print("=== QND Calibration Demo ===\n")
    true_params = Parameters(1.0, 0.5, 0.2)
    sample_sizes = [10, 20, 40, 80, 160, 320, 640]
    print(f"True parameters: {true_params}")
    print(f"Sample sizes to test: {sample_sizes}")
    print(f"Number of simulations per condition: 1000")
    print(f"Number of bootstrap repetitions: 1000")

    # Time a single estimation
    moments = ez.forward(true_params)
    observations = moments.sample(100)
    start = time.time()
    _ = qnd_single_estimation(observations, n_repetitions=1000)
    elapsed = time.time() - start
    print(f"\nElapsed time for a single QND estimation (N=100, 1000 bootstraps): {elapsed:.2f} seconds\n")

    print("Running calibration simulation...")
    start_time = time.time()
    single_results = run_single_parameter_simulation(
        true_params=true_params,
        sample_sizes=sample_sizes,
        n_simulations=1000,
        n_bootstrap=1000
    )
    elapsed_sim = time.time() - start_time
    print(f"Calibration simulation completed in {elapsed_sim/60:.1f} minutes\n")

    # Print summary
    print("=== Calibration Results ===")
    print(f"Marginal coverage (should be ≈ 0.95):")
    for name, cov in zip(['Boundary', 'Drift', 'NDT'], [single_results['boundary_coverage'], single_results['drift_coverage'], single_results['ndt_coverage']]):
        print(f"  {name:8s}: {np.mean(cov):.3f}")
    print(f"Joint coverage (should be ≈ {single_results['theoretical_joint_coverage']:.2f}): {np.mean(single_results['total_coverage']):.3f}")
    print(f"\nMean elapsed time per estimation (seconds):")
    for N, t in zip(sample_sizes, single_results['mean_elapsed_time']):
        print(f"  N={N:4d}: {t:.3f} s")

    # Make and save figure
    fig = create_calibration_figure(single_results)
    fig.savefig("qnd_calibration_demo.png", dpi=300, bbox_inches='tight')
    print("Calibration figure saved as qnd_calibration_demo.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QND Calibration Demo")
    parser.add_argument("--demo", action="store_true", help="Run the full demo")
    parser.add_argument("--quick", action="store_true", help="Run a quick demo with fewer simulations")
    args = parser.parse_args()
    if args.quick:
        n_simulations = 2000
        n_bootstrap = 2000
        cache_dir = "cache" # TODO: make a command line argument to ignore the cache
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        cache_file = os.path.join(cache_dir, f"qnd_calibration_quick_demo_{n_simulations}_{n_bootstrap}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                single_results = pickle.load(f)
        else:
            true_params = Parameters(1.0, 0.5, 0.2)
            sample_sizes = [32, 64, 128, 256, 512]
            print("Running quick calibration demo...")
            moments = ez.forward(true_params)
            observations = moments.sample(40)
            start = time.time()
            _ = qnd_single_estimation(observations, n_repetitions=1000)
            elapsed = time.time() - start
            print(f"Elapsed time for a single QND estimation (N=40, 1000 bootstraps): {elapsed:.2f} seconds\n")
            single_results = run_single_parameter_simulation(
                true_params=true_params,
                sample_sizes=sample_sizes,
                n_simulations=n_simulations,
                n_bootstrap=n_bootstrap
            )
            with open(cache_file, "wb") as f:
                pickle.dump(single_results, f)
        fig = create_calibration_figure(single_results)
        fig.savefig("qnd_calibration_quick_demo.png", dpi=300, bbox_inches='tight')
        fig.savefig("qnd_calibration_quick_demo.pdf", dpi=300, bbox_inches='tight')
        print("Quick calibration demo completed. Figure saved as qnd_calibration_quick_demo.png")
        plt.show()
    elif args.demo:
        demo()
    else:
        print("Use --demo for full demonstration or --quick for a quick demo")
