#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

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
Announce utility
"""
def announce(text: str = None):
    """Announce a message"""
    frame = inspect.stack()[1]
    full_path = frame.filename
    line_number = frame.lineno
    function_name = frame.function
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    print(f"\n# [{timestamp}] {full_path}:{line_number}#{function_name}()\n")
    if text is not None:
        print(f"\n# [{timestamp}] {text}\n")


"""
Test suite
"""
class TestSuite(unittest.TestCase):
    def test_announce(self):
        """
        Test that the announce function works correctly.
        """
        announce("Test announce")
        self.assertTrue(True)

"""
Demo
"""
def demo():
    announce("Demo for debug.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--test", action="store_true", help="Run the test suite")
    parser.add_argument("--demo", action="store_true", help="Run the demo")
    
    args = parser.parse_args()  
    
    if args.test:
        unittest.main(argv=[__file__], verbosity=0, failfast=True)

    if args.demo:
        demo()
    