import sys
import os
import unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Test all imports
print("Test that all imports work.", end="")
import base
# Check that ez.inverse now exists
if not hasattr(base, 'inverse'):
    raise ImportError("ez.inverse not found")

import classes
# Check that Parameters now exists
if not hasattr(classes, 'Parameters'):
    raise ImportError("Parameters not found")

import qnd
# Check that qnd.qnd_single now exists
if not hasattr(qnd, 'qnd_single_estimation'):
    raise ImportError("qnd.qnd_single_estimation not found")

import utils
# Check that utils.linear_prediction now exists
if not hasattr(utils, 'linear_prediction'):
    raise ImportError("utils.linear_prediction not found")

print(" ... ok")

# Run test suites
suite = unittest.TestSuite()

## Base
suite.addTest(unittest.TestLoader().loadTestsFromName('base.ez_equations.TestSuite'))

## Classes
suite.addTest(unittest.TestLoader().loadTestsFromName('classes.moments.TestSuite'))
suite.addTest(unittest.TestLoader().loadTestsFromName('classes.parameters.TestSuite'))
suite.addTest(unittest.TestLoader().loadTestsFromName('classes.design_matrix.TestSuite'))

## QND
suite.addTest(unittest.TestLoader().loadTestsFromName('qnd.qnd_single.TestSuite'))
suite.addTest(unittest.TestLoader().loadTestsFromName('qnd.qnd_beta_weights.TestSuite'))

## Utils
suite.addTest(unittest.TestLoader().loadTestsFromName('utils.debug.TestSuite'))
suite.addTest(unittest.TestLoader().loadTestsFromName('utils.linear_algebra.TestSuite'))
suite.addTest(unittest.TestLoader().loadTestsFromName('utils.prettify.TestSuite'))
suite.addTest(unittest.TestLoader().loadTestsFromName('utils.posterior.TestSuite'))
suite.addTest(unittest.TestLoader().loadTestsFromName('utils.simulation.TestSuite'))

## Main
# suite.addTest(unittest.TestLoader().loadTestsFromName('qnd.qnd_single.TestSuite'))


# Run the test suite
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)



# import os
# import time
# import pathlib

# # Define print function with timestamp
# tprint = lambda message: print(f"[{time.strftime('%H:%M:%S')}] {message}")

# # Get the current directory
# here = pathlib.Path(__file__).parent
# tprint(f"Current directory: {here}")

# # Find all .py files in the current directory
# py_files = [f for f in here.rglob('*.py') if f.name != 'test.py']

# # Execute each .py file with the --test flag
# for file in py_files:
#     tprint(f"Running tests for {file}...")
#     os.system(f'python {here / file} --test')
#     tprint(f"Tests for {file} completed")

# # Print a message to indicate that the tests have been run
# tprint("All tests have been run")