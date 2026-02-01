"""
Toy optimization problems for testing and demonstration.
"""

from .sphere import build_sphere_graph, sphere_numpy
from .rastrigin import build_rastrigin_graph, rastrigin_numpy
from .rosenbrock import build_rosenbrock_graph, rosenbrock_numpy

__all__ = [
    'build_sphere_graph',
    'build_rastrigin_graph',
    'build_rosenbrock_graph',
    'sphere_numpy',
    'rastrigin_numpy',
    'rosenbrock_numpy',
]
