"""
Fallback MCMC implementation for EZ-diffusion parameter estimation.

This module provides a Metropolis-Hastings MCMC implementation that can be used
when JAGS is not available, maintaining the same interface as the JAGS version.
"""

import unittest
import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
from scipy import stats
from copy import deepcopy
from typing import Callable, Tuple, List
from tqdm import tqdm
from vendor.ezas.classes import Parameters, Moments, Observations
from vendor.ezas.base.ez_equations import forward

# Configuration constants
MAX_RHAT = 1.1
DEFAULT_STEP_SIZE = 0.1
ADAPTATION_INTERVAL = 100
MIN_ACCEPTANCE_RATE = 0.2
MAX_ACCEPTANCE_RATE = 0.5
ADAPTATION_FACTOR = 0.8
EXPANSION_FACTOR = 1.2

# Parameter bounds
BOUNDARY_MIN, BOUNDARY_MAX = 0.1, 5.0
NDT_MIN, NDT_MAX = 0.0, 1.0
DRIFT_MIN_ABS = 1e-6

# Prior parameters
DRIFT_PRIOR_MEAN, DRIFT_PRIOR_STD = 0.0, 2.0


def _compute_log_likelihood(params: Parameters, observations: Observations) -> float:
    """Compute log-likelihood for EZ-diffusion model."""
    boundary, drift, ndt = params.boundary(), params.drift(), params.ndt()
    
    # Check parameter bounds
    if not (BOUNDARY_MIN < boundary < BOUNDARY_MAX and 
            NDT_MIN < ndt < NDT_MAX and 
            abs(drift) > DRIFT_MIN_ABS):
        return -np.inf
    
    try:
        # Get predicted moments from forward model
        pred_moments = forward(params)
        pred_acc  = pred_moments.accuracy
        pred_mean = pred_moments.mean_rt
        pred_var  = pred_moments.var_rt
        
        if not all(np.isfinite([pred_acc, pred_mean, pred_var])) or pred_var <= 0:
            return -np.inf
        
        # Compute likelihood components
        acc_ll = stats.binom.logpmf(
            observations.accuracy(), 
            observations.sample_size(), 
            pred_acc
        )
        rt_ll = stats.norm.logpdf(
            observations.mean_rt(), 
            pred_mean, 
            np.sqrt(pred_var / observations.sample_size())
        )
        var_ll = stats.gamma.logpdf(
            observations.var_rt(),
            (observations.sample_size() - 1) / 2, 
            scale=2 * pred_var / (observations.sample_size() - 1)
        )
        
        # Compute prior components
        boundary_prior = stats.uniform.logpdf(
            boundary, BOUNDARY_MIN, BOUNDARY_MAX - BOUNDARY_MIN
        )
        drift_prior = stats.norm.logpdf(
            drift, DRIFT_PRIOR_MEAN, DRIFT_PRIOR_STD
        )
        ndt_prior = stats.uniform.logpdf(
            ndt, NDT_MIN, NDT_MAX - NDT_MIN
        )
        
        total_ll = acc_ll + rt_ll + var_ll + boundary_prior + drift_prior + ndt_prior
        return total_ll if np.isfinite(total_ll) else -np.inf
        
    except (OverflowError, ValueError, ZeroDivisionError):
        return -np.inf


def _propose_parameters(
    current: Parameters, 
    cov: np.ndarray, 
    rng: np.random.Generator
) -> Parameters:
    """Propose new parameters using multivariate normal distribution."""
    current_array = np.array([current.boundary(), current.drift(), current.ndt()])
    proposal = rng.multivariate_normal(current_array, cov)
    return Parameters(proposal[0], proposal[1], proposal[2])


def _adapt_proposal_covariance(
    cov: np.ndarray, 
    acceptance_rate: float
) -> np.ndarray:
    """Adapt proposal covariance based on acceptance rate."""
    if acceptance_rate < MIN_ACCEPTANCE_RATE:
        return cov * ADAPTATION_FACTOR
    elif acceptance_rate > MAX_ACCEPTANCE_RATE:
        return cov * EXPANSION_FACTOR
    return cov


def _mcmc_chain(
    n_samples: int, 
    n_tune: int, 
    initial_params: Parameters, 
    log_likelihood_fn: Callable[[Parameters], float], 
    step_size: float = DEFAULT_STEP_SIZE, 
    rng: np.random.Generator = None
) -> Tuple[np.ndarray, float]:
    """Run single MCMC chain with adaptive proposal."""
    if rng is None:
        rng = np.random.default_rng()
    
    # Initialize chain
    current_params = initial_params
    current_ll = log_likelihood_fn(current_params)
    
    samples = []
    n_accepted = 0
    proposal_cov = np.eye(3) * step_size**2
    
    total_iterations = n_tune + n_samples
    
    for i in range(total_iterations):
        # Propose new parameters
        proposal_params = _propose_parameters(current_params, proposal_cov, rng)
        proposal_ll = log_likelihood_fn(proposal_params)
        
        # Accept/reject step
        if _should_accept(current_ll, proposal_ll, rng):
            current_params = proposal_params
            current_ll = proposal_ll
            n_accepted += 1
        
        # Store samples after burn-in (convert to array for compatibility)
        if i >= n_tune:
            param_array = np.array([current_params.boundary(), current_params.drift(), current_params.ndt()])
            samples.append(param_array)
            
        # Adapt proposal covariance during burn-in
        if i < n_tune and i > 0 and i % ADAPTATION_INTERVAL == 0:
            acceptance_rate = n_accepted / (i + 1)
            proposal_cov = _adapt_proposal_covariance(proposal_cov, acceptance_rate)
    
    return np.array(samples), n_accepted / total_iterations


def _should_accept(current_ll: float, proposal_ll: float, rng: np.random.Generator) -> bool:
    """Determine whether to accept proposal using Metropolis-Hastings criterion."""
    if proposal_ll > current_ll:
        return True
    elif np.isfinite(proposal_ll) and np.isfinite(current_ll):
        log_alpha = min(0, proposal_ll - current_ll)
        return rng.random() < np.exp(log_alpha)
    return False


def _compute_rhat(chain_samples: List[np.ndarray]) -> float:
    """Compute R-hat convergence diagnostic for multiple chains."""
    try:
        n_chains = len(chain_samples)
        if n_chains < 2:
            return np.nan
            
        chain_length = len(chain_samples[0])
        if chain_length < 2:
            return np.nan
        
        # Compute between and within chain variances
        chain_means = np.array([np.mean(chain) for chain in chain_samples])
        B = chain_length * np.var(chain_means, ddof=1)
        W = np.mean([np.var(chain, ddof=1) for chain in chain_samples])
        
        if W <= 0:
            return np.nan
            
        # Compute R-hat
        var_plus = ((chain_length - 1) / chain_length) * W + (1 / chain_length) * B
        rhat = np.sqrt(var_plus / W)
        
        return rhat if np.isfinite(rhat) else np.nan
        
    except (ValueError, ZeroDivisionError):
        return np.nan


def _get_initial_values() -> List[Parameters]:
    """Get initial parameter values for multiple chains."""
    return [
        Parameters(1.5, 0.5, 0.3),
        Parameters(2.0, 1.0, 0.2),
        Parameters(1.0, -0.5, 0.4)
    ]


def _create_parameters_from_stats(boundary_stats: dict, drift_stats: dict, ndt_stats: dict) -> Parameters:
    """Create Parameters object from summary statistics."""
    return Parameters(
        boundary             = float(boundary_stats['mean']),
        drift                = float(drift_stats['mean']),
        ndt                  = float(ndt_stats['mean']),
        boundary_sd          = float(boundary_stats['sd']),
        drift_sd             = float(drift_stats['sd']),
        ndt_sd               = float(ndt_stats['sd']),
        boundary_lower_bound = float(boundary_stats['q025']),
        boundary_upper_bound = float(boundary_stats['q975']),
        drift_lower_bound    = float(drift_stats['q025']),
        drift_upper_bound    = float(drift_stats['q975']),
        ndt_lower_bound      = float(ndt_stats['q025']),
        ndt_upper_bound      = float(ndt_stats['q975'])
    )


def _compute_summary_stats(samples: np.ndarray) -> dict:
    """Compute summary statistics for parameter samples."""
    return {
        'mean': np.mean(samples),
        'sd'  : np.std(samples, ddof=1),
        'q025': np.percentile(samples, 2.5),
        'q975': np.percentile(samples, 97.5)
    }


def _print_diagnostics(boundary_stats: dict, drift_stats: dict, ndt_stats: dict, 
                      boundary_rhat: float, drift_rhat: float, ndt_rhat: float, 
                      acceptance_rates: List[float], verbosity: int) -> None:
    """Print MCMC diagnostics and convergence information."""
    if verbosity >= 2:
        print(f"Boundary: mean={boundary_stats['mean']:.3f}, sd={boundary_stats['sd']:.3f}, rhat={boundary_rhat:.3f}")
        print(f"Drift:    mean={drift_stats['mean']:.3f}, sd={drift_stats['sd']:.3f}, rhat={drift_rhat:.3f}")
        print(f"NDT:      mean={ndt_stats['mean']:.3f}, sd={ndt_stats['sd']:.3f}, rhat={ndt_rhat:.3f}")
        print(f"Acceptance rates: {[f'{r:.3f}' for r in acceptance_rates]}")
    
    if verbosity >= 1:
        rhat_values = [r for r in [boundary_rhat, drift_rhat, ndt_rhat] if np.isfinite(r)]
        max_rhat = max(rhat_values) if rhat_values else np.nan
        
        if np.isnan(max_rhat):
            print("Warning: Could not calculate Rhat (numerical issues)")
        elif max_rhat > MAX_RHAT:
            print(f"Warning: Maximum Rhat is {max_rhat:.4f}, which is greater than {MAX_RHAT}.")
        else:
            print(f"Maximum Rhat is {max_rhat:.4f}, which is less than {MAX_RHAT}.")


def mcmc(observations: Observations, n_samples: int, n_tune: int, verbosity: int = 0) -> Parameters:
    """
    Run MCMC sampling for EZ-diffusion parameter estimation.
    
    Args:
        observations: Observed behavioral data
        n_samples: Number of samples to collect (post burn-in)
        n_tune: Number of burn-in samples
        verbosity: Verbosity level (0=silent, 1=convergence, 2=detailed)
    
    Returns:
        Parameters object with estimated values and uncertainty
    """
    rng = np.random.default_rng(42)
    
    # Create likelihood function with observations bound
    log_likelihood_fn = lambda params: _compute_log_likelihood(params, observations)
    
    # Run multiple chains
    chains = []
    acceptance_rates = []
    
    for initial_params in _get_initial_values():
        samples, acceptance_rate = _mcmc_chain(
            n_samples, n_tune, initial_params, log_likelihood_fn, rng=rng
        )
        chains.append(samples)
        acceptance_rates.append(acceptance_rate)
    
    # Combine chains and compute statistics
    all_samples = np.vstack(chains)
    
    boundary_stats = _compute_summary_stats(all_samples[:, 0])
    drift_stats    = _compute_summary_stats(all_samples[:, 1])
    ndt_stats      = _compute_summary_stats(all_samples[:, 2])
    
    # Compute convergence diagnostics
    boundary_rhat = _compute_rhat([chain[:, 0] for chain in chains])
    drift_rhat    = _compute_rhat([chain[:, 1] for chain in chains])
    ndt_rhat      = _compute_rhat([chain[:, 2] for chain in chains])
    
    # Print diagnostics
    _print_diagnostics(
        boundary_stats   = boundary_stats, 
        drift_stats      = drift_stats, 
        ndt_stats        = ndt_stats, 
        boundary_rhat    = boundary_rhat, 
        drift_rhat       = drift_rhat, 
        ndt_rhat         = ndt_rhat, 
        acceptance_rates = acceptance_rates, 
        verbosity        = verbosity
    )
    
    # Return Parameters object
    return _create_parameters_from_stats(
        boundary_stats = boundary_stats, 
        drift_stats    = drift_stats, 
        ndt_stats      = ndt_stats
    )


def generate_synthetic_data(
    parameters: Parameters, 
    N: int, 
    rng: np.random.Generator|None = None
) -> Observations:
    """Generate synthetic behavioral data from EZ-diffusion parameters."""
    if rng is None:
        rng = np.random.default_rng()
    
    return forward(parameters).sample(N, rng=rng)


def demo():
    """Demonstrate fallback MCMC implementation."""
    print("=== Fallback MCMC Demo ===")
    
    # Generate synthetic data
    true_parameters = Parameters(1.5, 1.0, 0.3)
    observations = generate_synthetic_data(
        parameters = true_parameters, 
        N          = 100, 
        rng        = np.random.default_rng(42)
    )
    
    print(f"True parameters:")
    print(true_parameters)
    print(f"Observed data:")
    print(observations)
    print()
    
    # Run MCMC estimation
    estimated_parameters = mcmc(
        observations = observations, 
        n_samples    = 2000, 
        n_tune       = 1000, 
        verbosity    = 2
    )
    print()
    print(estimated_parameters)
    
    # Show recovery assessment
    print("\n=== Recovery Assessment ===")
    print(f"True parameters:")
    print(true_parameters)
    print(f"Estimated parameters:")
    print(estimated_parameters)


def _compute_parameter_errors(
    estimated: Parameters, 
    true_params: Parameters
) -> dict:
    """Compute parameter estimation errors."""
    return {
        'boundary' : estimated.boundary() - true_params.boundary(),
        'drift'    : estimated.drift() - true_params.drift(),
        'ndt'      : estimated.ndt() - true_params.ndt()
    }


def _compute_simulation_stats(errors: dict) -> dict:
    """Compute bias and RMSE statistics for simulation results."""
    stats = {}
    for param_name in ['boundary', 'drift', 'ndt']:
        bias = np.mean(errors[param_name])
        rmse = np.sqrt(np.mean(np.array(errors[param_name])**2))
        stats[param_name] = {'bias': bias, 'rmse': rmse}
    return stats


def simulation(repetitions: int = 100):
    """Run simulation study to evaluate MCMC performance."""
    print(f"=== Simulation Study ({repetitions} repetitions) ===")
    
    test_parameters = [
        Parameters(1.5, 1.0, 0.3),
        Parameters(2.0, 0.5, 0.2),
        Parameters(1.0, -0.5, 0.4),
        Parameters(2.5, 1.5, 0.1),
        Parameters(1.2, -1.0, 0.35)
    ]
    
    results = []
    rng = np.random.default_rng(42)
    
    # Initialize storage for all parameter sets
    all_param_errors = {i: {'boundary': [], 'drift': [], 'ndt': []} for i in range(len(test_parameters))}
    all_param_failures = {i: 0 for i in range(len(test_parameters))}
    
    # Single loop over all combinations
    total_iterations = len(test_parameters) * repetitions
    
    print(f"\nRunning {total_iterations} total iterations across {len(test_parameters)} parameter sets...")
    
    with tqdm(total=total_iterations, desc="MCMC Simulation", unit="iter") as pbar:
        for iteration in range(total_iterations):
            # Determine which parameter set and repetition we're on
            param_idx = iteration // repetitions
            rep_idx = iteration % repetitions
            true_params = test_parameters[param_idx]
            
            # Update progress bar description with current parameter set
            if rep_idx == 0:
                pbar.set_description(f"Parameter set {param_idx + 1}/{len(test_parameters)}")
            
            try:
                # Generate synthetic data
                observations = generate_synthetic_data(
                    parameters = true_params, 
                    N          = 100, 
                    rng        = rng
                )
                
                # Run MCMC estimation
                estimated = mcmc(
                    observations = observations, 
                    n_samples    = 750, 
                    n_tune       = 250, 
                    verbosity    = 0
                )
                
                # Compute errors using ezas objects
                errors = _compute_parameter_errors(estimated, true_params)
                
                # Store errors for this parameter set
                for param_name in ['boundary', 'drift', 'ndt']:
                    all_param_errors[param_idx][param_name].append(errors[param_name])
                
            except Exception as e:
                all_param_failures[param_idx] += 1
                if rep_idx % 20 == 0:  # Only print some failures to avoid spam
                    tqdm.write(f"Failed iteration {iteration + 1}: {e}")
            
            # Update progress bar
            pbar.update(1)
    
    print()  # Add newline after progress bar
    
    # Process results for each parameter set
    for param_idx in range(len(test_parameters)):
        true_params = test_parameters[param_idx]
        all_errors = all_param_errors[param_idx]
        failures = all_param_failures[param_idx]
        
        # Compute and display results
        if all_errors['boundary']:
            success_rate = (repetitions - failures) / repetitions
            stats = _compute_simulation_stats(all_errors)
            
            print(f"\nParameter set {param_idx + 1} results:")
            print(f"  Success rate: {success_rate:.2%}")
            
            for param_name in ['boundary', 'drift', 'ndt']:
                bias = stats[param_name]['bias']
                rmse = stats[param_name]['rmse']
                print(f"  {param_name.capitalize()} - Bias: {bias:.4f}, RMSE: {rmse:.4f}")
            
            results.append({
                'params': true_params,
                'success_rate': success_rate,
                'stats': stats
            })
    
    # Overall summary
    print("\n=== Overall Summary ===")
    if results:
        overall_success = np.mean([r['success_rate'] for r in results])
        print(f"Overall success rate: {overall_success:.2%}")
        
        for param_name in ['boundary', 'drift', 'ndt']:
            all_biases = [r['stats'][param_name]['bias'] for r in results]
            all_rmses = [r['stats'][param_name]['rmse'] for r in results]
            print(f"{param_name.capitalize()} - Mean bias: {np.mean(all_biases):.4f}, Mean RMSE: {np.mean(all_rmses):.4f}")


class TestSuite(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        self.test_params = Parameters(1.5, 1.0, 0.3)
        self.test_observations = generate_synthetic_data(
            parameters = self.test_params, 
            N          = 100, 
            rng        = self.rng
        )
    
    def _create_test_observations(self, params: Parameters, n: int) -> Observations:
        """Helper to create test observations."""
        return generate_synthetic_data(
            parameters = params, 
            N          = n, 
            rng        = self.rng
        )
    
    def _run_mcmc_test(self, observations: Observations, n_samples: int = 100, n_tune: int = 100) -> Parameters:
        """Helper to run MCMC for testing."""
        return mcmc(observations, n_samples=n_samples, n_tune=n_tune, verbosity=0)
    
    def test_mcmc_basic_functionality(self):
        """Test basic MCMC functionality."""
        result = self._run_mcmc_test(self.test_observations)
        
        self.assertIsInstance(result, Parameters)
        for attr in ['boundary', 'drift', 'ndt', 'boundary_sd', 'drift_sd', 'ndt_sd']:
            self.assertIsInstance(getattr(result, attr)(), float)
            self.assertTrue(np.isfinite(getattr(result, attr)()))
    
    def test_parameter_bounds(self):
        """Test that estimated parameters stay within bounds."""
        result = self._run_mcmc_test(self.test_observations, n_samples=500, n_tune=200)
        
        self.assertGreater(result.boundary(), BOUNDARY_MIN)
        self.assertLess(result.boundary(), BOUNDARY_MAX)
        self.assertGreater(result.ndt(), NDT_MIN)
        self.assertLess(result.ndt(), NDT_MAX)
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        obs = self._create_test_observations(self.test_params, 100)
        
        self.assertIsInstance(obs, Observations)
        self.assertGreaterEqual(obs.accuracy(), 0)
        self.assertLessEqual(obs.accuracy(), 100)
        self.assertGreater(obs.mean_rt(), 0)
        self.assertGreater(obs.var_rt(), 0)
    
    def test_parameter_recovery(self):
        """Test parameter recovery with synthetic data."""
        rng = np.random.default_rng(123)
        true_params = Parameters(1.5, 1.0, 0.3)
        obs = generate_synthetic_data(
            parameters = true_params, 
            N          = 200, 
            rng        = rng
        )
        
        estimated = mcmc(obs, n_samples=2000, n_tune=1000, verbosity=0)
        
        # Check recovery within 3 standard deviations using ezas objects
        errors = _compute_parameter_errors(estimated, true_params)
        
        self.assertLess(abs(errors['boundary']), 3 * estimated.boundary_sd())
        self.assertLess(abs(errors['drift']), 3 * estimated.drift_sd())
        self.assertLess(abs(errors['ndt']), 3 * estimated.ndt_sd())
    
    def test_ezas_integration(self):
        """Test integration with ezas framework."""
        # Test forward function
        moments = forward(self.test_params)
        self.assertIsInstance(moments, Moments)
        
        # Test sampling
        obs = moments.sample(100, rng=self.rng)
        self.assertIsInstance(obs, Observations)
        
        # Test moment conversion
        moments2 = obs.to_moments()
        self.assertIsInstance(moments2, Moments)
        
        # Check that accuracy is reasonable (within sampling error)
        self.assertGreater(moments2.accuracy, 0.0)
        self.assertLess(moments2.accuracy, 1.0)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        test_cases = [
            (10, 8, 0.4, 0.05),     # Small sample
            (100, 95, 0.3, 0.02),   # High accuracy
            (100, 15, 0.8, 0.3),    # Low accuracy
            (1000, 999, 0.1, 0.001) # Extreme values
        ]
        
        for n, acc, rt, var in test_cases:
            with self.subTest(n=n, acc=acc, rt=rt, var=var):
                obs = Observations(
                    accuracy    = acc, 
                    mean_rt     = rt, 
                    var_rt      = var, 
                    sample_size = n
                )
                result = mcmc(
                    observations = obs, 
                    n_samples    = 100, 
                    n_tune       = 50, 
                    verbosity    = 0
                )
                self.assertIsInstance(result, Parameters)
                self.assertTrue(np.isfinite(result.boundary()))
                self.assertTrue(np.isfinite(result.drift()))
                self.assertTrue(np.isfinite(result.ndt()))
    
    def test_convergence_diagnostics(self):
        """Test convergence diagnostic output."""
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            mcmc(
                observations = self.test_observations, 
                n_samples    = 500, 
                n_tune       = 200, 
                verbosity    = 1
            )
        
        output = f.getvalue()
        self.assertIn("Rhat", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fallback MCMC Implementation")
    parser.add_argument("--test", action="store_true", help="Run test suite")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("--simulation", action="store_true", help="Run simulation study")
    parser.add_argument("--repetitions", type=int, default=100, help="Simulation repetitions")
    
    args = parser.parse_args()
    
    if args.test:
        unittest.main(argv=[''], exit=False)
    elif args.demo:
        demo()
    elif args.simulation:
        simulation(repetitions=args.repetitions)
    else:
        print("Use --demo, --test, or --simulation to run the implementation") 