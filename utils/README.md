# Utils Module

This module provides utility functions that support the core functionality of the EZAS library. These are helper functions that make other parts of the library work more smoothly.

## What it contains

The utils module contains five main functions and classes:

- **`announce(message)`** - Prints timestamped messages to track what's happening during long computations
- **`linear_prediction(design_matrix, weights)`** - Does linear algebra calculations needed for design matrix analysis
- **`b(value)`** - Formats boolean variables for pretty printing with emoji cross/check boxes
- **`PosteriorSummary`** - Analyzes results from Bayesian estimation, computing statistics and credible intervals
- **`Results`** - Handles simulation results and organizes output data

These utilities help with debugging, mathematical calculations, output formatting, and result analysis throughout the library. 