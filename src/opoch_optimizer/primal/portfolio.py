"""
Primal Portfolio: Delta*_primal Test Algebra

Defines a family of primal acts for UB discovery:
1. Global Sobol sampling
2. Multi-start local refinement
3. Region-focused exploration (guided by LB)
4. PhaseProbe for periodic/shifted multimodal functions

All acts are deterministic with explicit costs.
This is the closure that makes UB discovery as powerful as LB computation.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import hashlib

from .sobol import SobolGenerator, SobolPoint
from .phase_probe import PhaseProbe


class PrimalActType(Enum):
    """Types of primal acts in Delta*_primal."""
    GLOBAL_SOBOL = "global_sobol"          # Low-discrepancy global sampling
    LOCAL_REFINE = "local_refine"           # L-BFGS-B from a point
    MULTI_START = "multi_start"             # Multiple local refinements
    REGION_SOBOL = "region_sobol"           # Sobol within promising region
    REGION_REFINE = "region_refine"         # Sample + refine in region
    PHASE_PROBE = "phase_probe"             # DFT-based phase extraction


@dataclass
class PrimalAct:
    """A primal act with cost and result."""
    act_type: PrimalActType
    cost: int  # Number of function evaluations
    x: Optional[np.ndarray] = None  # Best point found
    f: float = float('inf')  # Best value found
    points_evaluated: int = 0
    region_hash: Optional[str] = None


@dataclass
class PrimalState:
    """State of the primal portfolio."""
    best_x: Optional[np.ndarray] = None
    best_f: float = float('inf')
    total_evals: int = 0
    sobol_evals: int = 0
    local_evals: int = 0
    best_k_points: List[Tuple[np.ndarray, float]] = field(default_factory=list)


class PrimalPortfolio:
    """
    Portfolio of primal acts for UB discovery.

    Implements Delta*_primal as a closed algebra of exploration acts.
    All acts are deterministic and logged.
    """

    def __init__(
        self,
        dimension: int,
        bounds: List[Tuple[float, float]],
        objective: Callable[[np.ndarray], float],
        eval_tracker: Optional[Callable[[float], None]] = None
    ):
        """
        Initialize primal portfolio.

        Args:
            dimension: Number of variables
            bounds: Variable bounds
            objective: Function to minimize (each call is an evaluation)
            eval_tracker: Optional callback to track evaluations
        """
        self.dimension = dimension
        self.bounds = bounds
        self.lb = np.array([b[0] for b in bounds])
        self.ub = np.array([b[1] for b in bounds])
        self.objective = objective
        self.eval_tracker = eval_tracker

        # Sobol generator for deterministic sampling
        self.sobol = SobolGenerator(dimension, bounds)

        # State
        self.state = PrimalState()

        # Configuration
        self.sobol_budget_fraction = 0.3
        self.top_k = min(50, max(10, dimension * 3))
        self.local_maxiter = 100

        # PhaseProbe for periodic function identification
        self.phase_probe = PhaseProbe(
            objective=self._raw_objective,
            dimension=dimension,
            bounds=bounds
        )
        self.phase_probe_enabled = True
        self.phase_probe_samples = 32

    def _raw_objective(self, x: np.ndarray) -> float:
        """Raw objective without tracking."""
        x = np.clip(x, self.lb, self.ub)
        return self.objective(x)

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate objective and track."""
        x = np.clip(x, self.lb, self.ub)
        f = self.objective(x)
        self.state.total_evals += 1

        if self.eval_tracker:
            self.eval_tracker(f)

        if f < self.state.best_f:
            self.state.best_f = f
            self.state.best_x = x.copy()

            self.state.best_k_points.append((x.copy(), f))
            self.state.best_k_points.sort(key=lambda p: p[1])
            self.state.best_k_points = self.state.best_k_points[:self.top_k]

        return f

    def global_sobol_seeding(self, n_points: int) -> PrimalAct:
        """Global UB seeding by deterministic Sobol sampling."""
        points = self.sobol.generate(n_points)
        best_x = None
        best_f = float('inf')

        for pt in points:
            f = self.evaluate(pt.x)
            self.state.sobol_evals += 1
            if f < best_f:
                best_f = f
                best_x = pt.x.copy()

        return PrimalAct(
            act_type=PrimalActType.GLOBAL_SOBOL,
            cost=n_points,
            x=best_x,
            f=best_f,
            points_evaluated=n_points
        )

    def local_refine(self, x0: np.ndarray, maxiter: Optional[int] = None) -> PrimalAct:
        """Local refinement using L-BFGS-B."""
        if maxiter is None:
            maxiter = self.local_maxiter

        try:
            from scipy.optimize import minimize

            evals_before = self.state.total_evals

            result = minimize(
                self.evaluate,
                x0,
                method='L-BFGS-B',
                bounds=list(zip(self.lb, self.ub)),
                options={'maxiter': maxiter, 'disp': False}
            )

            evals_used = self.state.total_evals - evals_before
            self.state.local_evals += evals_used

            return PrimalAct(
                act_type=PrimalActType.LOCAL_REFINE,
                cost=evals_used,
                x=result.x if result.success else x0,
                f=result.fun if result.success else self.objective(x0),
                points_evaluated=evals_used
            )
        except Exception:
            return PrimalAct(
                act_type=PrimalActType.LOCAL_REFINE,
                cost=1,
                x=x0,
                f=self.evaluate(x0),
                points_evaluated=1
            )

    def phase_probe_identification(self) -> Tuple[bool, Optional[PrimalAct]]:
        """
        Try to identify shift using PhaseProbe.

        Returns:
            (success, act): success=True if periodic structure detected
        """
        if not self.phase_probe_enabled:
            return False, None

        self.phase_probe.reset()

        x0 = (self.lb + self.ub) / 2
        periodic_dims = 0

        for i in range(self.dimension):
            is_periodic, energy_ratio = self.phase_probe.detect_periodicity(x0, i, M=32)
            if is_periodic:
                periodic_dims += 1

        if periodic_dims < self.dimension * 0.5:
            detection_evals = self.phase_probe.total_evals
            self.state.total_evals += detection_evals
            return False, None

        result = self.phase_probe.identify_and_refine(M=self.phase_probe_samples)
        candidate_f = self.evaluate(result.candidate_x)

        act = PrimalAct(
            act_type=PrimalActType.PHASE_PROBE,
            cost=result.total_evals + 1,
            x=result.candidate_x,
            f=candidate_f,
            points_evaluated=result.total_evals + 1
        )

        return True, act

    def multi_start_refine(self, k: Optional[int] = None) -> PrimalAct:
        """Multi-start local refinement from top K Sobol points."""
        if k is None:
            k = self.top_k

        if not self.state.best_k_points:
            center = (self.lb + self.ub) / 2
            return self.local_refine(center)

        best_x = None
        best_f = float('inf')
        total_evals = 0

        for x0, _ in self.state.best_k_points[:k]:
            act = self.local_refine(x0)
            total_evals += act.cost
            if act.f < best_f:
                best_f = act.f
                best_x = act.x

        return PrimalAct(
            act_type=PrimalActType.MULTI_START,
            cost=total_evals,
            x=best_x,
            f=best_f,
            points_evaluated=total_evals
        )

    def region_sobol(
        self,
        region_lb: np.ndarray,
        region_ub: np.ndarray,
        n_points: int,
        region_hash: str
    ) -> PrimalAct:
        """Region-focused Sobol sampling."""
        points = self.sobol.generate_in_region(n_points, region_lb, region_ub, region_hash)
        best_x = None
        best_f = float('inf')

        for pt in points:
            f = self.evaluate(pt.x)
            self.state.sobol_evals += 1
            if f < best_f:
                best_f = f
                best_x = pt.x.copy()

        return PrimalAct(
            act_type=PrimalActType.REGION_SOBOL,
            cost=n_points,
            x=best_x,
            f=best_f,
            points_evaluated=n_points,
            region_hash=region_hash
        )

    def region_refine(
        self,
        region_lb: np.ndarray,
        region_ub: np.ndarray,
        n_samples: int,
        region_hash: str
    ) -> PrimalAct:
        """Combined: Sample in region + local refine from best."""
        sample_act = self.region_sobol(region_lb, region_ub, n_samples, region_hash)

        if sample_act.x is None:
            return sample_act

        refine_act = self.local_refine(sample_act.x)

        return PrimalAct(
            act_type=PrimalActType.REGION_REFINE,
            cost=sample_act.cost + refine_act.cost,
            x=refine_act.x if refine_act.f < sample_act.f else sample_act.x,
            f=min(refine_act.f, sample_act.f),
            points_evaluated=sample_act.points_evaluated + refine_act.points_evaluated,
            region_hash=region_hash
        )

    def full_exploration(self, total_budget: int) -> PrimalAct:
        """
        Complete deterministic exploration strategy.

        Stage -1: PhaseProbe identification
        Stage 0: Global Sobol seeding
        Stage 1: Multi-start refinement from top-k
        """
        if self.phase_probe_enabled:
            success, phase_act = self.phase_probe_identification()
            if success and phase_act is not None:
                if phase_act.x is not None:
                    refine_act = self.local_refine(phase_act.x)
                    if refine_act.f < phase_act.f:
                        phase_act.x = refine_act.x
                        phase_act.f = refine_act.f
                        phase_act.cost += refine_act.cost
                        phase_act.points_evaluated += refine_act.points_evaluated

        sobol_budget = int(total_budget * self.sobol_budget_fraction)
        sobol_budget = max(sobol_budget, self.dimension * 10)

        self.global_sobol_seeding(sobol_budget)
        self.multi_start_refine()

        return PrimalAct(
            act_type=PrimalActType.MULTI_START,
            cost=self.state.total_evals,
            x=self.state.best_x,
            f=self.state.best_f,
            points_evaluated=self.state.total_evals
        )

    def get_upper_bound(self) -> Tuple[float, Optional[np.ndarray]]:
        """Get current best UB and solution."""
        return self.state.best_f, self.state.best_x

    def reset(self):
        """Reset portfolio state."""
        self.state = PrimalState()
        self.sobol.reset()
        self.phase_probe.reset()
