"""
Bounds Module — Certified Lower Bound Computation

Provides tiers of relaxations:
- Tier 0: Interval arithmetic (universal)
- Tier 1: McCormick convex relaxations (factorable)
- Tier 2a: FBBT for equalities
- Tier 2b: OBBT (optional)
- Tier 3: SOS relaxations (polynomial, optional)

Each tier produces certified bounds with replay-verifiable certificates.
"""

from .interval import (
    Interval,
    IntervalEvaluator,
    interval_evaluate,
    ROUND_EPS,
)
from .mccormick import (
    McCormickRelaxation,
    McCormickBounds,
)
from .fbbt import (
    FBBTOperator,
    FBBTInequalityOperator,
    FBBTResult,
    apply_fbbt_all_constraints,
)
from .interval_newton import (
    IntervalNewtonOperator,
    IntervalNewtonResult,
    IntervalNewtonStatus,
    apply_interval_newton_all_constraints,
)
from .krawczyk import (
    KrawczykOperator,
    KrawczykResult,
    KrawczykStatus,
    apply_krawczyk_all_constraints,
)

__all__ = [
    # Interval (Tier 0)
    'Interval',
    'IntervalEvaluator',
    'interval_evaluate',
    'ROUND_EPS',
    # McCormick (Tier 1)
    'McCormickRelaxation',
    'McCormickBounds',
    # FBBT (Tier 2a)
    'FBBTOperator',
    'FBBTInequalityOperator',
    'FBBTResult',
    'apply_fbbt_all_constraints',
    # Interval Newton (Tier 2b)
    'IntervalNewtonOperator',
    'IntervalNewtonResult',
    'IntervalNewtonStatus',
    'apply_interval_newton_all_constraints',
    # Krawczyk (Tier 2b - manifolds)
    'KrawczykOperator',
    'KrawczykResult',
    'KrawczykStatus',
    'apply_krawczyk_all_constraints',
]
