"""
MCMC submodule for EZ-diffusion parameter estimation.

This module provides a fallback MCMC implementation when JAGS is not available.
"""

from .fallback_mcmc import fallback_mcmc_implementation

__all__ = ['fallback_mcmc_implementation'] 