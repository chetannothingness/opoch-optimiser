"""
Solver Module - OPOCH Kernel and Branch-and-Bound

Provides:
- OPOCHKernel: Main certified global optimization solver
- FeasibilityBNP: Branch-and-prune for feasibility
- Supporting components for B&B
"""

from .feasibility_bnb import (
    FeasibilityBNP,
    FeasibilityResult,
    FeasibilityStatus,
    find_initial_feasible_ub,
)
from .opoch_kernel import (
    OPOCHKernel,
    OPOCHConfig,
)

__all__ = [
    'OPOCHKernel',
    'OPOCHConfig',
    'FeasibilityBNP',
    'FeasibilityResult',
    'FeasibilityStatus',
    'find_initial_feasible_ub',
]
