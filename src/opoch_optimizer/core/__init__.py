"""
Core Module - Foundational Components

Provides:
- Canonical JSON serialization
- Output gate (enforces 3-verdict contract)
- Certificate types
"""

from .canonical_json import canonical_dumps, canonical_hash
from .output_gate import (
    Verdict,
    Certificate,
    OptimalityResult,
    UnsatResult,
    OmegaGapResult,
    OutputGate,
    SolverResult,
)

__all__ = [
    'canonical_dumps',
    'canonical_hash',
    'Verdict',
    'Certificate',
    'OptimalityResult',
    'UnsatResult',
    'OmegaGapResult',
    'OutputGate',
    'SolverResult',
]
