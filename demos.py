#!/usr/bin/env python3

import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ez_pymc.pymc_single        import demo as pymc_single_demo
from ez_pymc.pymc_multiple      import demo as pymc_multiple_demo
from ez_pymc.pymc_beta_weights  import demo as pymc_beta_weights_demo

from qnd.qnd_single             import demo as qnd_single_demo
from qnd.qnd_beta_weights       import demo as qnd_beta_weights_demo
from qnd.qnd_inference          import demo as qnd_inference_demo

from jags.jags_single           import demo as jags_single_demo
from jags.jags_multiple         import demo as jags_multiple_demo
from jags.jags_beta_weights     import demo as jags_beta_weights_demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--engine", 
        choices=["pymc", "qnd", "jags"], 
        help="Select the engine to use"
    )
    parser.add_argument(
        "--demo", 
        choices=["single", "multiple", "beta_weights", "inference"], 
        help="Select the demo to run"
    )
    
    args = parser.parse_args()
    
    if args.engine == "pymc":
        if args.demo == "single":
            pymc_single_demo()
        if args.demo == "multiple":
            pymc_multiple_demo()
        if args.demo == "beta_weights":
            pymc_beta_weights_demo()
        if args.demo == "inference":
            raise NotImplementedError("Inference demo not yet implemented for PyMC")

    if args.engine == "qnd":
        if args.demo == "single":
            qnd_single_demo()
        if args.demo == "multiple":
            raise NotImplementedError("Multiple parameter estimation not yet implemented for QND")
        if args.demo == "beta_weights":
            qnd_beta_weights_demo()
        if args.demo == "inference":
            qnd_inference_demo()
    
    if args.engine == "jags":
        if args.demo == "single":
            jags_single_demo()
        if args.demo == "multiple":
            jags_multiple_demo()
        if args.demo == "beta_weights":
            jags_beta_weights_demo()
        if args.demo == "inference":
            raise NotImplementedError("Inference not yet implemented for JAGS")
    