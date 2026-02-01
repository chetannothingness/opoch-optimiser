"""
OPOCH Optimizer - Deterministic Global Nonlinear Optimization with Auditable Certificates

This package provides a complete branch-and-reduce optimizer that returns:
- UNIQUE-OPT: Globally optimal solution with certificate
- UNSAT: Infeasibility proof
- Omega-GAP: Exact remaining gap with next separator action

Every run is deterministic and replayable from receipts.

Key Features:
- Certified lower bounds via interval arithmetic and McCormick relaxations
- FBBT + Interval Newton for equality/inequality constraint propagation
- PhaseProbe for periodic function identification
- Deterministic control via canonical fingerprints
- Receipt chain for replay verification
"""

from .contract import (
    ProblemContract,
    Region,
    Bounds,
)
from .expr_graph import (
    ExpressionGraph,
    ExprNode,
    Variable,
    Constant,
    UnaryOp,
    BinaryOp,
    OpType,
    TracedVar,
)
from .receipts import (
    Receipt,
    ReceiptChain,
    ActionType,
    canonical_dumps,
    canonical_hash,
)
from .tie_safe import (
    TieSafeChoice,
    canonical_fingerprint,
    deterministic_argmin,
    deterministic_argmax,
)
from .costs import (
    CostModel,
    ActType,
    WitnessTier,
    DEFAULT_COSTS,
)
from .core.output_gate import (
    Verdict,
    OptimalityResult,
    UnsatResult,
    OmegaGapResult,
    OutputGate,
    Certificate,
)
from .bounds.interval import (
    Interval,
    IntervalEvaluator,
    interval_evaluate,
    ROUND_EPS,
)
from .bounds.mccormick import (
    McCormickRelaxation,
    McCormickBounds,
)
from .bounds.fbbt import (
    FBBTOperator,
    FBBTInequalityOperator,
    FBBTResult,
    apply_fbbt_all_constraints,
)
from .bounds.interval_newton import (
    IntervalNewtonOperator,
    IntervalNewtonResult,
    apply_interval_newton_all_constraints,
)
from .primal.sobol import SobolGenerator, SobolPoint
from .primal.phase_probe import PhaseProbe, PhaseProbeResult
from .primal.portfolio import PrimalPortfolio, PrimalAct, PrimalActType
from .solver.feasibility_bnb import (
    FeasibilityBNP,
    FeasibilityResult,
    FeasibilityStatus,
)
from .solver.opoch_kernel import (
    OPOCHKernel,
    OPOCHConfig,
)

__version__ = "0.1.0"
__author__ = "OPOCH Team"

__all__ = [
    # Contract
    "ProblemContract",
    "Region",
    "Bounds",
    # Expression Graph
    "ExpressionGraph",
    "ExprNode",
    "Variable",
    "Constant",
    "UnaryOp",
    "BinaryOp",
    "OpType",
    "TracedVar",
    # Receipts
    "Receipt",
    "ReceiptChain",
    "ActionType",
    "canonical_dumps",
    "canonical_hash",
    # Tie-safe
    "TieSafeChoice",
    "canonical_fingerprint",
    "deterministic_argmin",
    "deterministic_argmax",
    # Costs
    "CostModel",
    "ActType",
    "WitnessTier",
    "DEFAULT_COSTS",
    # Output gate
    "Verdict",
    "OptimalityResult",
    "UnsatResult",
    "OmegaGapResult",
    "OutputGate",
    "Certificate",
    # Bounds - Interval (Tier 0)
    "Interval",
    "IntervalEvaluator",
    "interval_evaluate",
    "ROUND_EPS",
    # Bounds - McCormick (Tier 1)
    "McCormickRelaxation",
    "McCormickBounds",
    # Bounds - FBBT (Tier 2a)
    "FBBTOperator",
    "FBBTInequalityOperator",
    "FBBTResult",
    "apply_fbbt_all_constraints",
    # Bounds - Interval Newton (Tier 2b)
    "IntervalNewtonOperator",
    "IntervalNewtonResult",
    "apply_interval_newton_all_constraints",
    # Primal
    "SobolGenerator",
    "SobolPoint",
    "PhaseProbe",
    "PhaseProbeResult",
    "PrimalPortfolio",
    "PrimalAct",
    "PrimalActType",
    # Solver
    "FeasibilityBNP",
    "FeasibilityResult",
    "FeasibilityStatus",
    "OPOCHKernel",
    "OPOCHConfig",
]
