import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import unittest
import argparse

class Moments:
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
    
    def sample(self, sample_size: int, rng = np.random.default_rng()) -> 'Observations':
        if not isinstance(sample_size, int):
            raise TypeError("sample_size must be an integer")
        if sample_size <= 0:
            raise ValueError("sample_size must be greater than 0")  
        if not isinstance(rng, np.random.Generator):
            raise TypeError(f"rng must be an instance of np.random.Generator, was {type(rng)} with value {rng}")
        return Observations(
            accuracy = rng.binomial(sample_size, self.accuracy),
            mean_rt = rng.normal(self.mean_rt, np.sqrt(self.var_rt/sample_size)),
            var_rt = rng.gamma((sample_size-1)/2, 2*self.var_rt/(sample_size-1)),
            sample_size = sample_size
        )
    
    def __sub__(self, other):
        if not isinstance(other, Moments):
            raise TypeError("other must be an instance of Moments")
        return Moments(self.accuracy - other.accuracy, 
                            self.mean_rt - other.mean_rt, 
                            self.var_rt - other.var_rt)

    def __str__(self):
        return f"{'Accuracy:':10s}{self.accuracy:5.2f}, " + \
               f"{'Mean RT:':10s}{self.mean_rt:5.2f}, " + \
               f"{'Var RT:':10s}{self.var_rt:5.2f}"

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

        self._sample_size = sample_size
        self._accuracy = accuracy
        self._mean_rt = mean_rt
        self._var_rt = var_rt
        
    def accuracy(self) -> int:
        return self._accuracy
    
    def mean_rt(self) -> float:
        return self._mean_rt
    
    def var_rt(self) -> float:
        return self._var_rt
    
    def sample_size(self) -> int:
        return self._sample_size
    
    def to_moments(self) -> Moments:
        return Moments(self._accuracy/self._sample_size, self._mean_rt, self._var_rt)

    def resample(self, sample_size: int | None = None, rng = np.random.default_rng()) -> 'Observations':
        if sample_size is not None:
            if not isinstance(sample_size, int):
                raise TypeError("sample_size must be an integer")
            if sample_size <= 0:
                raise ValueError("sample_size must be greater than 0")
        else:
            sample_size = self.sample_size()

        return self.to_moments().sample(sample_size, rng)
    
    def __str__(self):
        return f"{'Accuracy:':10s}{self.accuracy():5d}, " + \
               f"{'Mean RT:':10s}{self.mean_rt():5.2f}, " + \
               f"{'Var RT:':10s}{self.var_rt():5.2f}, " + \
               f"{'Sample size:':10s}{self.sample_size():5d}"

# --- TESTS AND DEMO ---
class TestSuite(unittest.TestCase):
    def test_moments_init(self):
        """
        Test that the Moments class works correctly.
        """
        m = Moments(0.8, 0.5, 0.1)
        self.assertEqual(m.accuracy, 0.8)
        self.assertEqual(m.mean_rt, 0.5)
        self.assertEqual(m.var_rt, 0.1)
    def test_observations_init(self):
        """
        Test that the Observations class works correctly.
        """
        o = Observations(8, 0.5, 0.1, 10)
        self.assertEqual(o.accuracy(), 8)
        self.assertEqual(o.mean_rt(), 0.5)
        self.assertEqual(o.var_rt(), 0.1)
        self.assertEqual(o.sample_size(), 10)
    def test_sample(self):
        """
        Test that the sample method works correctly.
        """
        m = Moments(0.7, 0.4, 0.2)
        o = m.sample(10)
        self.assertIsInstance(o, Observations)
        self.assertEqual(o.sample_size(), 10)
    def test_resample(self):
        """
        Test that the resample method works correctly.
        """
        m = Moments(0.7, 0.4, 0.2)
        o = m.sample(10)
        o2 = o.resample(5)
        self.assertIsInstance(o2, Observations)
        self.assertEqual(o2.sample_size(), 5)

def demo():
    print("Demo for Moments and Observations:")
    m = Moments(0.8, 0.5, 0.1)
    print("Moments:\n  ", m)
    o2 = m.sample(100)
    print("Sampled Observations (N=100):\n  ", o2)
    o3 = o2.resample(50)
    print("Resampled Observations (N=50):\n  ", o3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run the test suite")
    parser.add_argument("--demo", action="store_true", help="Run the demo")
    args = parser.parse_args()
    if args.test:
        unittest.main(argv=[__file__], verbosity=2, exit=False)
    if args.demo:
        demo() 