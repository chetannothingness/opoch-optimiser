"""
Primal Module - Upper Bound Discovery (Delta*_primal)

Provides deterministic primal acts for UB discovery:
- Sobol: Low-discrepancy quasi-random sampling
- PhaseProbe: DFT-based phase extraction for periodic functions
- Portfolio: Combined exploration strategy
"""

from .sobol import SobolGenerator, SobolPoint
from .phase_probe import PhaseProbe, PhaseProbeResult
from .portfolio import PrimalPortfolio, PrimalAct, PrimalActType

__all__ = [
    'SobolGenerator',
    'SobolPoint',
    'PhaseProbe',
    'PhaseProbeResult',
    'PrimalPortfolio',
    'PrimalAct',
    'PrimalActType',
]
