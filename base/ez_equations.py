import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import unittest
import argparse

from vendor.ezas.classes.parameters import Parameters
from vendor.ezas.classes.moments import Moments, Observations

def forward(params: Parameters | list[Parameters]) -> Moments | list[Moments]:
    if not (isinstance(params, Parameters) or (isinstance(params, list) and all(isinstance(x, Parameters) for x in params))):
        raise TypeError("params must be a Parameters instance or a list of Parameters")
    if isinstance(params, Parameters):
        return _forward_single(params)
    if isinstance(params, list):
        return [_forward_single(x) for x in params]
    
def _forward_single(params: Parameters) -> Moments:
    if params.boundary() <= 0:
        raise ValueError("Boundary must be strictly positive.")
    if params.drift() == 0:
        return Moments(0.5, params.ndt() + params.boundary(), params.boundary())
    y = np.exp(-params.boundary() * params.drift())
    pred_acc = 1 / (y + 1)
    pred_mean = (params.ndt() + 
                (params.boundary() / (2 * params.drift())) * 
                ((1 - y) / (1 + y)))
    pred_var = ((params.boundary() / (2 * params.drift()**3)) * 
                ((1 - 2*params.boundary()*params.drift()*y - y**2) / 
                 ((y + 1)**2)))
    return Moments(pred_acc, pred_mean, pred_var)

def inverse(moments: Moments | list[Moments]) -> Parameters | list[Parameters]:
    if isinstance(moments, Observations):
        raise ValueError("Observations must be converted to Moments before using inverse")
        # obs_stats = obs_stats.to_moments()
    if isinstance(moments, list) and all(isinstance(x, Observations) for x in moments):
        raise ValueError("Observations must be converted to Moments before using inverse_equations")
        # obs_stats = [s.to_moments() for s in obs_stats]
    if not (isinstance(moments, Moments) or (isinstance(moments, list) and all(isinstance(x, Moments) for x in moments))):
        raise TypeError("obs_stats must be a Moments instance or a list of Moments or a Observations instance or a list of Observations")
    if isinstance(moments, Moments):
        return _inverse_single(moments)
    if isinstance(moments, list):
        return [_inverse_single(x) for x in moments]

def _inverse_single(moments: Moments) -> Parameters:
    if moments.accuracy == 0 or moments.accuracy == 1:
        return Parameters(0, 0, 0)
    logit_acc = np.log(moments.accuracy / (1 - moments.accuracy))
    numerator = logit_acc * (
        moments.accuracy**2 * logit_acc - 
        moments.accuracy * logit_acc + 
        moments.accuracy - 0.5
    )
    sign = 1 if moments.accuracy > 0.5 else -1
    try:
        est_drift = sign * np.power(numerator / moments.var_rt, 0.25)
        if abs(est_drift) < 1e-9:
            return Parameters(1.0, 0.0, moments.mean_rt)
        est_bound = abs(logit_acc / est_drift)
        y = np.exp(-est_drift * est_bound)
        est_ndt = moments.mean_rt - (
            (est_bound / (2 * est_drift)) * 
            ((1 - y) / (1 + y))
        )
        return Parameters(est_bound, est_drift, est_ndt)
    except (ValueError, RuntimeWarning, ZeroDivisionError):
        print(f"WARNING: Numerical failure for moments: {moments}")
        return Parameters(1.0, 0.0, moments.mean_rt)

def random(lower_bound: Parameters, upper_bound: Parameters) -> Parameters:
    return Parameters.random(lower_bound, upper_bound)

# --- TESTS AND DEMO ---
class TestEZEquations(unittest.TestCase):
    def test_forward_inverse(self):
        p = Parameters(1.0, 0.5, -0.2)
        m = forward(p)
        self.assertIsInstance(m, Moments)
        p2 = inverse(m)
        self.assertIsInstance(p2, Parameters)
        self.assertAlmostEqual(p.drift(), p2.drift(), places=2)
        self.assertAlmostEqual(p.boundary(), p2.boundary(), places=2)
        self.assertAlmostEqual(p.ndt(), p2.ndt(), places=2)
    def test_forward_list(self):
        plist = [Parameters(1.0, 0.5, 0.2), Parameters(1.2, 0.6, -0.3)]
        mlist = forward(plist)
        self.assertEqual(len(mlist), 2)
        self.assertIsInstance(mlist[0], Moments)
    def test_inverse_list(self):
        mlist = [Moments(0.7, 0.4, 0.2), Moments(0.8, 0.5, 0.1)]
        plist = inverse(mlist)
        self.assertEqual(len(plist), 2)
        self.assertIsInstance(plist[0], Parameters)
    def test_generate_random_parameters(self):
        lb = Parameters(0, 0, 0)
        ub = Parameters(1, 1, 1)
        p = random(lb, ub)
        self.assertTrue(lb.boundary() <= p.boundary() <= ub.boundary())
        self.assertTrue(lb.drift() <= p.drift() <= ub.drift())
        self.assertTrue(lb.ndt() <= p.ndt() <= ub.ndt())

def demo():
    print("Demo for EZ equations:")
    p = Parameters(1.0, 0.5, 0.2)
    print("Parameters:", p)
    m = forward(p)
    print("Forward equations (Moments):", m)
    o = m.sample(10)
    print("Sampled Observations (N=10):", o)
    p2 = inverse(m)
    print("Inverse equations (Recovered Parameters):", p2)
    lb = Parameters(0, 0, 0)
    ub = Parameters(2, 2, 2)
    rand_p = random(lb, ub)
    print("Random parameters in [0,2]:", rand_p)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run the test suite")
    parser.add_argument("--demo", action="store_true", help="Run the demo")
    args = parser.parse_args()
    if args.test:
        unittest.main(argv=[__file__], verbosity=2, exit=False)
    if args.demo:
        demo() 