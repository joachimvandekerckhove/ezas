#!/usr/bin/env python3

import sys
import argparse
import unittest

import numpy as np

def linear_prediction(design_matrix: np.ndarray, weights) -> np.ndarray:
    """
    Perform linear prediction: parameters = design_matrix @ weights
    
    Args:
        design_matrix: Design matrix (n_conditions x n_weights)
        weights: Weight vector (n_weights,) - can be list or numpy array
    
    Returns:
        Predicted parameters (n_conditions,)
    """
    # Convert weights to numpy array if it's a list
    if isinstance(weights, list):
        weights = np.array(weights)
    
    if design_matrix.shape[1] != weights.shape[0]:
        raise ValueError(f"Design matrix columns ({design_matrix.shape[1]}) must match weights length ({weights.shape[0]})")
    
    return design_matrix @ weights

def linear_regression_pinv(design_matrix: np.ndarray, parameters) -> np.ndarray:
    """
    Perform linear regression: weights = pinv(design_matrix) @ parameters
    
    Args:
        design_matrix: Design matrix (n_conditions x n_weights)
        parameters: Parameter vector (n_conditions,) - can be list or numpy array
    
    Returns:
        Estimated weights (n_weights,)
    """
    # Convert parameters to numpy array if it's a list
    if isinstance(parameters, list):
        parameters = np.array(parameters)
    
    if design_matrix.shape[0] != parameters.shape[0]:
        raise ValueError(f"Design matrix rows ({design_matrix.shape[0]}) must match parameters length ({parameters.shape[0]})")
    
    return np.linalg.pinv(design_matrix) @ parameters


def linear_regression_lstsq(design_matrix: np.ndarray, parameters) -> np.ndarray:
    """
    Perform linear regression: weights = pinv(design_matrix) @ parameters
    
    Args:
        design_matrix: Design matrix (n_conditions x n_weights)
        parameters: Parameter vector (n_conditions,) - can be list or numpy array
    
    Returns:
        Estimated weights (n_weights,)
    """
    # Convert parameters to numpy array if it's a list
    if isinstance(parameters, list):
        parameters = np.array(parameters)
    
    if design_matrix.shape[0] != parameters.shape[0]:
        raise ValueError(f"Design matrix rows ({design_matrix.shape[0]}) must match parameters length ({parameters.shape[0]})")
    
    return np.linalg.lstsq(design_matrix, parameters, rcond=None)[0]


def linear_prediction_batch(design_matrix: np.ndarray, weights) -> np.ndarray:
    """
    Perform linear prediction for multiple weight vectors.
    
    Args:
        design_matrix: Design matrix (n_conditions x n_weights)
        weights: Weight matrix (n_weights x n_batch) or weight vector (n_weights,) - can be list or numpy array
    
    Returns:
        Predicted parameters (n_conditions x n_batch) or (n_conditions,)
    """
    # Convert weights to numpy array if it's a list
    if isinstance(weights, list):
        weights = np.array(weights)
    
    if weights.ndim == 1:
        return linear_prediction(design_matrix, weights)
    else:
        return design_matrix @ weights

def linear_regression_batch(design_matrix: np.ndarray, parameters) -> np.ndarray:
    """
    Perform linear regression for multiple parameter vectors.
    
    Args:
        design_matrix: Design matrix (n_conditions x n_weights)
        parameters: Parameter matrix (n_conditions x n_batch) or parameter vector (n_conditions,) - can be list or numpy array
    
    Returns:
        Estimated weights (n_weights x n_batch) or (n_weights,)
    """
    # Convert parameters to numpy array if it's a list
    if isinstance(parameters, list):
        parameters = np.array(parameters)
    
    if parameters.ndim == 1:
        return linear_regression_pinv(design_matrix, parameters)
    else:
        return np.linalg.pinv(design_matrix) @ parameters

## --- CHECKS ---

def check_linear_algebra_consistency(design_matrix: np.ndarray, 
                                   weights: np.ndarray, 
                                   parameters: np.ndarray,
                                   tolerance: float = 1e-10) -> bool:
    """
    Check if linear prediction and regression are consistent.
    """
    predicted_params = linear_prediction(design_matrix, weights)
    estimated_weights = linear_regression_pinv(design_matrix, parameters)
    reconstructed_params = linear_prediction(design_matrix, estimated_weights)
    
    # Check if prediction matches parameters
    prediction_match = np.allclose(predicted_params, parameters, atol=tolerance)
    
    # Check if regression reconstruction matches original parameters
    regression_match = np.allclose(reconstructed_params, parameters, atol=tolerance)

    return prediction_match and regression_match


## --- DEMO ---
def demo():
    """
    Demo for linear_algebra.py
    """
    design_matrix = np.array([[1, 0, 0], 
                              [0, 1, 0], 
                              [0, 0, 1], 
                              [0, 0, 1]])
    weights = np.array([1.0, 1.5, 2.0])
    parameters = np.array([1.0, 1.5, 2.0, 2.0])
    
    predicted_params = linear_prediction(design_matrix, weights)
    estimated_weights = linear_regression_pinv(design_matrix, parameters)
    reconstructed_params = linear_prediction(design_matrix, estimated_weights)
    
    tolerance = 1e-10
    
    # Check if prediction matches parameters
    prediction_match = np.allclose(predicted_params, parameters, atol=tolerance)
    
    # Check if regression reconstruction matches original parameters
    regression_match = np.allclose(reconstructed_params, parameters, atol=tolerance)

    print(f"Prediction match: {prediction_match}")
    print(f"Regression match: {regression_match}")


## --- TESTS ---

class TestSuite(unittest.TestCase):
    def setUp(self):
        # Create a simple test case
        self.design_matrix = np.array([[1, 0, 0], 
                                       [0, 1, 0], 
                                       [0, 0, 1], 
                                       [0, 0, 1]])
        self.weights = np.array([1.0, 1.5, 2.0])
        self.parameters = np.array([1.0, 1.5, 2.0, 2.0])

    def test_linear_prediction(self):
        """
        Test that linear prediction works correctly.
        """
        result = linear_prediction(self.design_matrix, self.weights)
        expected = self.design_matrix @ self.weights
        np.testing.assert_array_almost_equal(result, expected)

    def test_linear_regression(self):
        """
        Test that linear regression works correctly.
        """
        result = linear_regression_pinv(self.design_matrix, self.parameters)
        expected = np.linalg.pinv(self.design_matrix) @ self.parameters
        np.testing.assert_array_almost_equal(result, expected)

    def test_linear_prediction_batch(self):
        """
        Test that linear prediction works correctly for multiple weight vectors.
        """
        result_single = linear_prediction_batch(self.design_matrix, self.weights)
        expected_single = linear_prediction(self.design_matrix, self.weights)
        np.testing.assert_array_almost_equal(result_single, expected_single)
        weights_batch = np.column_stack([self.weights, self.weights * 2])
        result_batch = linear_prediction_batch(self.design_matrix, weights_batch)
        expected_batch = self.design_matrix @ weights_batch
        np.testing.assert_array_almost_equal(result_batch, expected_batch)

    def test_linear_regression_batch(self):
        """
        Test that linear regression works correctly for multiple parameter vectors.
        """
        result_single = linear_regression_batch(self.design_matrix, self.parameters)
        expected_single = linear_regression_pinv(self.design_matrix, self.parameters)
        np.testing.assert_array_almost_equal(result_single, expected_single)
        params_batch = np.column_stack([self.parameters, self.parameters * 2])
        result_batch = linear_regression_batch(self.design_matrix, params_batch)
        expected_batch = np.linalg.pinv(self.design_matrix) @ params_batch
        np.testing.assert_array_almost_equal(result_batch, expected_batch)

    def test_consistency_check(self):
        """
        Test that linear prediction and regression are consistent.
        """
        is_consistent = check_linear_algebra_consistency(
            self.design_matrix, self.weights, self.parameters
        )
        self.assertTrue(is_consistent)
        inconsistent_weights = np.array([0.0, 0.0, 0.0])
        is_consistent = check_linear_algebra_consistency(
            self.design_matrix, inconsistent_weights, self.parameters
        )
        self.assertFalse(is_consistent)

    def test_dimension_validation(self):
        """
        Test that functions raise ValueError when dimensions are incorrect.
        """
        wrong_weights = np.array([1.0, 1.5])  # Wrong length
        with self.assertRaises(ValueError):
            linear_prediction(self.design_matrix, wrong_weights)
        wrong_params = np.array([1.0, 1.5])  # Wrong length
        with self.assertRaises(ValueError):
            linear_regression_pinv(self.design_matrix, wrong_params)

    def test_list_inputs(self):
        """
        Test that functions work correctly with list inputs.
        """
        # Test with list inputs
        weights_list = [1.0, 1.5, 2.0]
        params_list = [1.0, 1.5, 2.0, 2.0]
        
        # Test linear prediction with list
        result_pred = linear_prediction(self.design_matrix, weights_list)
        expected_pred = linear_prediction(self.design_matrix, self.weights)
        np.testing.assert_array_almost_equal(result_pred, expected_pred)
        
        # Test linear regression with list
        result_reg = linear_regression_pinv(self.design_matrix, params_list)
        expected_reg = linear_regression_pinv(self.design_matrix, self.parameters)
        np.testing.assert_array_almost_equal(result_reg, expected_reg)
        
        # Test consistency check with lists
        is_consistent = check_linear_algebra_consistency(
            self.design_matrix, weights_list, params_list
        )
        self.assertTrue(is_consistent)
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run unit tests for linear_algebra.py')
    parser.add_argument('--demo', action='store_true', help='Run demo for linear_algebra.py')
    args = parser.parse_args()

    if args.test:
        unittest.main(argv=[sys.argv[0]]) 
    if args.demo:
        demo()