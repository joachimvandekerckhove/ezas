import os
import time
import pathlib

# Define print function with timestamp
tprint = lambda message: print(f"[{time.strftime('%H:%M:%S')}] {message}")

# Get the current directory
here = pathlib.Path(__file__).parent
tprint(f"Current directory: {here}")

# Find all .py files in the current directory
py_files = [f for f in here.glob('*.py') if f.name != 'test.py']

# Execute each .py file with the --test flag
for file in py_files:
    tprint(f"Running tests for {file}...")
    os.system(f'python {here / file} --test')
    tprint(f"Tests for {file} completed")

# Print a message to indicate that the tests have been run
tprint("All tests have been run")