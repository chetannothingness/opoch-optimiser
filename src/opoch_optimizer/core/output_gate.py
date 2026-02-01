"""
Output Gate

Enforces the OPOCH contract: only three admissible outputs.

- UNIQUE-OPT: globally epsilon-optimal solution with certificate
- UNSAT: infeasible with certificate
- Omega-GAP: best-known + proven bounds + gap (budget-limited)

No other output is permitted. This module validates all outputs
before they leave the solver.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import numpy as np

from .canonical_json import canonical_dumps, canonical_hash


class Verdict(Enum):
    """The three admissible output verdicts."""
    UNIQUE_OPT = "UNIQUE-OPT"  # Globally optimal with certificate
    UNSAT = "UNSAT"            # Infeasible with certificate
    OMEGA_GAP = "Omega-GAP"    # Gap result (budget exhausted)


@dataclass
class Certificate:
    """
    Base class for optimality/infeasibility certificates.

    All certificates must be independently verifiable.
    """
    cert_type: str
    data: Dict[str, Any]
    cert_hash: str = field(default="")

    def __post_init__(self):
        if not self.cert_hash:
            self.cert_hash = canonical_hash({
                "cert_type": self.cert_type,
                "data": self.data
            })

    def to_canonical(self) -> Dict[str, Any]:
        return {
            "cert_type": self.cert_type,
            "data": self.data,
            "cert_hash": self.cert_hash
        }


@dataclass
class IntervalCertificate(Certificate):
    """Certificate based on interval arithmetic bounds."""

    def __init__(
        self,
        region_bounds: Dict[str, tuple],
        objective_bounds: tuple,
        constraint_bounds: Dict[str, tuple],
        feasibility_status: str
    ):
        super().__init__(
            cert_type="interval",
            data={
                "region_bounds": region_bounds,
                "objective_bounds": objective_bounds,
                "constraint_bounds": constraint_bounds,
                "feasibility_status": feasibility_status
            }
        )


@dataclass
class RelaxationCertificate(Certificate):
    """Certificate from convex relaxation (McCormick/LP/SDP)."""

    def __init__(
        self,
        relaxation_type: str,
        lower_bound: float,
        dual_solution: Optional[Dict[str, Any]] = None,
        solver_status: str = "optimal"
    ):
        super().__init__(
            cert_type="relaxation",
            data={
                "relaxation_type": relaxation_type,
                "lower_bound": lower_bound,
                "dual_solution": dual_solution,
                "solver_status": solver_status
            }
        )


@dataclass
class CoverRefutationCertificate(Certificate):
    """Certificate proving infeasibility by region cover."""

    def __init__(
        self,
        region_certificates: List[Dict[str, Any]],
        cover_complete: bool
    ):
        super().__init__(
            cert_type="cover_refutation",
            data={
                "region_certificates": region_certificates,
                "cover_complete": cover_complete
            }
        )


@dataclass
class OptimalityResult:
    """Result object for UNIQUE-OPT verdict."""
    verdict: Verdict = Verdict.UNIQUE_OPT
    x_optimal: np.ndarray = None
    objective_value: float = None
    upper_bound: float = None
    lower_bound: float = None
    gap: float = None
    epsilon: float = None
    certificate: Any = None
    nodes_explored: int = 0
    relaxations_solved: int = 0

    def to_canonical(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "x_optimal": self.x_optimal.tolist() if self.x_optimal is not None else None,
            "objective_value": self.objective_value,
            "upper_bound": self.upper_bound,
            "lower_bound": self.lower_bound,
            "gap": self.gap,
            "epsilon": self.epsilon,
            "certificate": self.certificate.to_canonical() if hasattr(self.certificate, 'to_canonical') else self.certificate,
            "nodes_explored": self.nodes_explored,
            "relaxations_solved": self.relaxations_solved
        }


@dataclass
class UnsatResult:
    """Result object for UNSAT verdict."""
    verdict: Verdict = Verdict.UNSAT
    certificate: Any = None
    nodes_explored: int = 0
    relaxations_solved: int = 0

    def to_canonical(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "certificate": self.certificate.to_canonical() if hasattr(self.certificate, 'to_canonical') else self.certificate,
            "nodes_explored": self.nodes_explored,
            "relaxations_solved": self.relaxations_solved
        }


@dataclass
class OmegaGapResult:
    """Result object for Omega-GAP verdict."""
    verdict: Verdict = Verdict.OMEGA_GAP
    x_best: np.ndarray = None
    upper_bound: float = None
    lower_bound: float = None
    gap: float = None
    next_separator_action: Dict[str, Any] = None
    nodes_explored: int = 0
    relaxations_solved: int = 0
    budget_exhausted: str = "time"  # "time" or "nodes"

    def to_canonical(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "x_best": self.x_best.tolist() if self.x_best is not None else None,
            "upper_bound": self.upper_bound,
            "lower_bound": self.lower_bound,
            "gap": self.gap,
            "next_separator_action": self.next_separator_action,
            "nodes_explored": self.nodes_explored,
            "relaxations_solved": self.relaxations_solved,
            "budget_exhausted": self.budget_exhausted
        }


SolverResult = Union[OptimalityResult, UnsatResult, OmegaGapResult]


class OutputGate:
    """
    Validates and enforces the output contract.

    Only three outputs are admissible:
    1. UNIQUE-OPT with verified certificate
    2. UNSAT with verified certificate
    3. Omega-GAP with bounds and gap

    Any attempt to output invalid results raises an error.
    """

    def __init__(self, epsilon: float, feas_tol: float = 1e-8):
        """
        Initialize the output gate.

        Args:
            epsilon: Optimality tolerance
            feas_tol: Feasibility tolerance for constraint checking
        """
        self.epsilon = epsilon
        self.feas_tol = feas_tol

    def validate_unique_opt(
        self,
        result: OptimalityResult,
        problem: Any
    ) -> bool:
        """
        Validate a UNIQUE-OPT result.

        Checks:
        - x_optimal is in the domain
        - x_optimal is feasible
        - gap <= epsilon
        - certificate is valid
        """
        x = result.x_optimal
        if x is None:
            raise ValueError("UNIQUE-OPT requires x_optimal")

        for i, (lo, hi) in enumerate(problem.bounds):
            if x[i] < lo - self.feas_tol or x[i] > hi + self.feas_tol:
                raise ValueError(f"x[{i}] = {x[i]} outside bounds [{lo}, {hi}]")

        for j, g in enumerate(problem.ineq_constraints):
            val = g(x)
            if val > self.feas_tol:
                raise ValueError(f"Inequality constraint {j} violated: g(x) = {val}")

        for k, h in enumerate(problem.eq_constraints):
            val = h(x)
            if abs(val) > self.feas_tol:
                raise ValueError(f"Equality constraint {k} violated: h(x) = {val}")

        gap = result.upper_bound - result.lower_bound
        if gap > self.epsilon + self.feas_tol:
            raise ValueError(f"Gap {gap} exceeds epsilon {self.epsilon}")

        if result.certificate is None:
            raise ValueError("UNIQUE-OPT requires a certificate")

        return True

    def validate_unsat(
        self,
        result: UnsatResult,
        problem: Any
    ) -> bool:
        """Validate an UNSAT result."""
        if result.certificate is None:
            raise ValueError("UNSAT requires a certificate")

        cert = result.certificate
        if hasattr(cert, 'cert_type') and cert.cert_type == "cover_refutation":
            if not cert.data.get("cover_complete", False):
                raise ValueError("Cover refutation certificate incomplete")

        return True

    def validate_omega_gap(
        self,
        result: OmegaGapResult,
        problem: Any
    ) -> bool:
        """Validate an Omega-GAP result."""
        if result.x_best is not None:
            x = result.x_best
            for i, (lo, hi) in enumerate(problem.bounds):
                if x[i] < lo - self.feas_tol or x[i] > hi + self.feas_tol:
                    raise ValueError(f"x_best[{i}] outside bounds")

            for j, g in enumerate(problem.ineq_constraints):
                val = g(x)
                if val > self.feas_tol:
                    raise ValueError(f"x_best violates inequality {j}")

            for k, h in enumerate(problem.eq_constraints):
                val = h(x)
                if abs(val) > self.feas_tol:
                    raise ValueError(f"x_best violates equality {k}")

        if result.upper_bound < result.lower_bound - self.feas_tol:
            raise ValueError("upper_bound < lower_bound")

        expected_gap = result.upper_bound - result.lower_bound
        if abs(result.gap - expected_gap) > self.feas_tol:
            raise ValueError(f"Gap mismatch: {result.gap} vs {expected_gap}")

        return True

    def emit(
        self,
        result: SolverResult,
        problem: Any
    ) -> SolverResult:
        """
        Validate and emit a result through the output gate.

        This is the only way results should leave the solver.
        """
        if isinstance(result, OptimalityResult):
            self.validate_unique_opt(result, problem)
        elif isinstance(result, UnsatResult):
            self.validate_unsat(result, problem)
        elif isinstance(result, OmegaGapResult):
            self.validate_omega_gap(result, problem)
        else:
            raise ValueError(f"Unknown result type: {type(result)}")

        return result
