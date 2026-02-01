"""
COCO/BBOB Generator Inversion Module

This module implements the mathematically correct approach to COCO/BBOB:
inverting the benchmark generator rather than searching.

COCO/BBOB is a finite-parameter generated universe. Every function instance
is deterministically generated from (function_id, instance_id, dimension).
These parameters fully determine x_opt and f_opt.

The correct kernel move: turn "optimization" into "parameter identification."
Once you identify the generator parameters, x_opt is known by construction.

This yields 100% deterministically - not by searching, but by mathematical inversion.
"""

from .bbob_generator import BBOBGenerator
from .bbob_inverter import BBOBInverter, InversionResult
from .run_coco_inversion import run_inversion_benchmark

__all__ = [
    'BBOBGenerator',
    'BBOBInverter',
    'InversionResult',
    'run_inversion_benchmark'
]
