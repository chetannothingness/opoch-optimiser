"""
GLOBALLib Benchmark Suite

Pure mathematical certification:
- UNIQUE-OPT: Certified via gap closure (UB - LB ≤ ε)
- UNSAT: Certified via refutation cover (Δ* proves empty)
- No reference to external optimal values
- All results verifiable via replay
"""

from .globallib_problems import (
    GLOBALLibProblem,
    get_problem,
    get_all_problems,
    PROBLEM_REGISTRY,
)
from .globallib_runner import (
    run_globallib_benchmark,
    GLOBALLibResult,
)

__all__ = [
    'GLOBALLibProblem',
    'get_problem',
    'get_all_problems',
    'PROBLEM_REGISTRY',
    'run_globallib_benchmark',
    'GLOBALLibResult',
]
