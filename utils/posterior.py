#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from typing import Tuple, List
from vendor.ezas.base import ez_equations as ez
from vendor.ezas.classes.parameters import Parameters
import unittest
import argparse
import time
import inspect
import os

"""
Posterior summary object
"""
class PosteriorSummary:
    """Posterior summary object"""
    def __init__(self, true_values, means, stds, lower_quantiles, upper_quantiles, field_names):
        self._true_values = true_values
        self._means = means
        self._stds = stds
        self._lower_quantiles = lower_quantiles
        self._upper_quantiles = upper_quantiles
        self._field_names = field_names
        
    def field_names(self):
        return self._field_names
    
    def coverage(self):
        return (self._lower_quantiles < self._true_values) & (self._true_values < self._upper_quantiles)
    
    def joint_coverage(self):
        return np.mean(self.coverage())

    def true_values(self):
        return self._true_values
    
    def means(self):
        return self._means
    
    def stds(self):
        return self._stds
    
    def lower_quantiles(self):
        return self._lower_quantiles
    
    def upper_quantiles(self):
        return self._upper_quantiles
    
    def __str__(self):
        return f"{'True values':16s} : {self.true_values}\n" + \
               f"{'Means':16s} : {self.means}\n" + \
               f"{'Stds':16s} : {self.stds}\n" + \
               f"{'Lower quantiles':16s} : {self.lower_quantiles}\n" + \
               f"{'Upper quantiles':16s} : {self.upper_quantiles}\n" + \
               f"{'Field names':16s} : {self.field_names}"

"""
Test suite
"""
class TestSuite(unittest.TestCase):
    pass

"""
Demo
"""
def demo():
    print("Demo for posterior:")
    print("No demo available")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--test", action="store_true", help="Run the test suite")
    parser.add_argument("--demo", action="store_true", help="Run the demo")
    
    args = parser.parse_args()  
    
    if args.test:
        unittest.main(argv=[__file__], verbosity=0, failfast=True)

    if args.demo:
        demo()
    