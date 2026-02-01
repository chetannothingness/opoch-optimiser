"""
BBOB (Black-Box Optimization Benchmark) function definitions.

These are the standard test functions from the COCO/BBOB benchmark suite.
"""

# BBOB functions are defined in the benchmark runner script
# This module provides a reference for the function definitions

BBOB_FUNCTIONS = {
    1: {
        'name': 'Sphere',
        'description': 'Separable, unimodal',
        'formula': 'f(x) = sum((x_i - shift_i)^2)',
        'optimum': 'x* = shift, f* = 0',
        'bounds': (-5, 5),
    },
    2: {
        'name': 'Ellipsoid',
        'description': 'Separable, unimodal, ill-conditioned',
        'formula': 'f(x) = sum(10^(6*(i-1)/(n-1)) * (x_i - shift_i)^2)',
        'optimum': 'x* = shift, f* = 0',
        'bounds': (-5, 5),
    },
    3: {
        'name': 'Rastrigin',
        'description': 'Separable, multimodal (~10^n local minima)',
        'formula': 'f(x) = 10n + sum((x_i-s_i)^2 - 10*cos(2*pi*(x_i-s_i)))',
        'optimum': 'x* = shift, f* = 0',
        'bounds': (-5.12, 5.12),
    },
    8: {
        'name': 'Rosenbrock',
        'description': 'Non-separable, narrow valley',
        'formula': 'f(x) = sum(100*(x_{i+1}-x_i^2)^2 + (1-x_i)^2)',
        'optimum': 'x* = (1, 1, ..., 1), f* = 0',
        'bounds': (-5, 10),
    },
}

__all__ = ['BBOB_FUNCTIONS']
