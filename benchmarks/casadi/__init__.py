"""
CasADi Benchmark Suites

Four benchmark suites for comprehensive CasADi certification:

Suite HS: Hock-Schittkowski Problems (THE standard NLP test suite)
    - HS071, HS076, HS035, HS038, HS044, HS065, HS100
    - Rosenbrock 2D/5D/10D
    - Known optimal solutions for verification

Suite A: Industrial NLP (robotics/control)
    - Direct multiple shooting OCP
    - Direct collocation OCP
    - Parameter estimation under constraints

Suite B: Regression / System Identification (NIST-like)
    - Objectives as ResidualIR
    - Least-squares structure

Suite C: MINLP / Mixed Discrete Design
    - Integer variables
    - Combinatorial structure

Contract L (KKT): Applied to all suites
Contract G (Global): Applied to small/medium problems
"""

from .hock_schittkowski import get_hock_schittkowski_problems, KNOWN_OPTIMA
from .suite_b_regression import get_regression_problems
from .suite_a_industrial import get_industrial_problems
from .suite_c_minlp import get_minlp_problems

__all__ = [
    'get_hock_schittkowski_problems',
    'KNOWN_OPTIMA',
    'get_regression_problems',
    'get_industrial_problems',
    'get_minlp_problems',
]
