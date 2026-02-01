"""
COCO Portfolio Optimizer

Combines multiple strategies for comprehensive BBOB coverage:
1. Quadratic Identification (for f1, f2, f10 - quadratic/ill-conditioned)
2. DCMA (for rotated/non-separable functions)

The portfolio adaptively selects the best strategy based on
early function probing.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Dict, Any
import hashlib

from .dcma import DCMA, DCMAConfig, DCMAResult
from .quadratic_id import QuadraticIdentifier, QuadraticResult


@dataclass
class PortfolioResult:
    """Result of portfolio optimization."""
    x_best: np.ndarray
    f_best: float
    evaluations: int
    trajectory: List[Tuple[int, float]]
    strategy_used: str
    converged: bool
    reason: str
    receipt_hash: str
    sub_results: Dict[str, Any]


class COCOPortfolio:
    """
    Portfolio optimizer for COCO/BBOB functions.

    Strategy selection:
    1. Quick quadratic probe (cheap: ~3d evaluations)
    2. If quadratic detected → use QuadraticIdentifier (O(d²) for exact solution)
    3. Otherwise → use DCMA (rotationally invariant evolution)

    This dominates:
    - Pure quadratics (f1 sphere, f2 ellipsoid) via exact solving
    - Rotated functions via DCMA's covariance adaptation
    - Ill-conditioned functions via metric learning
    """

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        dim: int,
        bounds: List[Tuple[float, float]],
        dcma_config: DCMAConfig = None
    ):
        """
        Initialize portfolio optimizer.

        Args:
            objective: Function to minimize
            dim: Problem dimension
            bounds: Variable bounds [(lb, ub), ...]
            dcma_config: Optional DCMA configuration
        """
        self.objective = objective
        self.dim = dim
        self.bounds = bounds
        self.lb = np.array([b[0] for b in bounds])
        self.ub = np.array([b[1] for b in bounds])
        self.dcma_config = dcma_config or DCMAConfig()

        # Tracking
        self.evaluations = 0
        self.best_x = None
        self.best_f = float('inf')
        self.trajectory = []
        self.f_cache = {}

    def _eval(self, x: np.ndarray) -> float:
        """Evaluate with caching and tracking."""
        key = tuple(np.round(x, 10))
        if key not in self.f_cache:
            f = self.objective(x)
            self.f_cache[key] = f
            self.evaluations += 1

            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
                self.trajectory.append((self.evaluations, self.best_f))

        return self.f_cache[key]

    def _wrapped_objective(self, x: np.ndarray) -> float:
        """Wrapped objective that updates our tracking."""
        f = self.objective(x)
        self.evaluations += 1

        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
            self.trajectory.append((self.evaluations, self.best_f))

        return f

    def optimize(self, max_evaluations: int) -> PortfolioResult:
        """
        Run portfolio optimization.

        Args:
            max_evaluations: Maximum function evaluations

        Returns:
            PortfolioResult with best solution and metadata
        """
        sub_results = {}
        strategy_used = "dcma"  # Default

        # Phase 0: Quick quadratic detection (if budget allows)
        if max_evaluations >= 5 * self.dim and self.dim <= 20:
            quad_id = QuadraticIdentifier(
                self._eval,
                self.dim,
                self.bounds,
                h=1e-4
            )

            is_likely_quadratic = quad_id.quick_detect(n_samples=10, tol=0.05)
            quad_detect_evals = quad_id.evaluations

            if is_likely_quadratic:
                # Phase 1: Full quadratic identification
                quad_id.reset()
                quad_id.evaluations = 0

                # Re-wrap to use our tracking
                quad_id_full = QuadraticIdentifier(
                    self._eval,
                    self.dim,
                    self.bounds,
                    h=1e-4
                )

                result = quad_id_full.identify_and_solve(verify=True)

                if result.is_quadratic and result.confidence > 0.95:
                    # Quadratic solver succeeded
                    strategy_used = "quadratic"
                    sub_results["quadratic"] = {
                        "x_optimal": result.x_optimal.tolist() if result.x_optimal is not None else None,
                        "f_optimal": result.f_optimal,
                        "confidence": result.confidence,
                        "evaluations": result.evaluations
                    }

                    # Update best
                    if result.x_optimal is not None and result.f_optimal < self.best_f:
                        self.best_f = result.f_optimal
                        self.best_x = result.x_optimal.copy()
                        self.trajectory.append((self.evaluations, self.best_f))

                    # If very high confidence and low function value, we're done
                    if result.confidence > 0.99 and result.f_optimal < 1e-8:
                        receipt_hash = self._compute_receipt_hash()
                        return PortfolioResult(
                            x_best=self.best_x,
                            f_best=self.best_f,
                            evaluations=self.evaluations,
                            trajectory=self.trajectory.copy(),
                            strategy_used=strategy_used,
                            converged=True,
                            reason="quadratic_exact",
                            receipt_hash=receipt_hash,
                            sub_results=sub_results
                        )

        # Phase 2: DCMA for remaining budget
        remaining_budget = max_evaluations - self.evaluations

        if remaining_budget > 10:
            dcma = DCMA(
                self._wrapped_objective,
                self.dim,
                self.bounds,
                self.dcma_config
            )

            # Initialize with current best if available
            if self.best_x is not None:
                dcma.mean = self.best_x.copy()
                dcma.best_x = self.best_x.copy()
                dcma.best_f = self.best_f

            dcma_result = dcma.optimize(remaining_budget)

            sub_results["dcma"] = {
                "x_best": dcma_result.x_best.tolist() if dcma_result.x_best is not None else None,
                "f_best": dcma_result.f_best,
                "converged": dcma_result.converged,
                "reason": dcma_result.reason
            }

            # Update strategy if DCMA was primary
            if strategy_used != "quadratic":
                strategy_used = "dcma"

            # Update best from DCMA
            if dcma_result.x_best is not None and dcma_result.f_best < self.best_f:
                self.best_f = dcma_result.f_best
                self.best_x = dcma_result.x_best.copy()

        # Compute final receipt
        receipt_hash = self._compute_receipt_hash()

        converged = self.best_f < 1e-8
        reason = "target_reached" if converged else "max_evals"

        return PortfolioResult(
            x_best=self.best_x,
            f_best=self.best_f,
            evaluations=self.evaluations,
            trajectory=self.trajectory.copy(),
            strategy_used=strategy_used,
            converged=converged,
            reason=reason,
            receipt_hash=receipt_hash,
            sub_results=sub_results
        )

    def _compute_receipt_hash(self) -> str:
        """Compute deterministic receipt hash."""
        data = f"{self.best_x.tolist() if self.best_x is not None else 'None'}:{self.best_f}:{self.evaluations}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def reset(self):
        """Reset optimizer state."""
        self.evaluations = 0
        self.best_x = None
        self.best_f = float('inf')
        self.trajectory = []
        self.f_cache = {}


class AdaptiveRestartPortfolio:
    """
    Portfolio with adaptive restarts for multimodal functions.

    Uses IPOP-like strategy: increase population on restart.
    """

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        dim: int,
        bounds: List[Tuple[float, float]],
        max_restarts: int = 9
    ):
        self.objective = objective
        self.dim = dim
        self.bounds = bounds
        self.max_restarts = max_restarts

        self.evaluations = 0
        self.best_x = None
        self.best_f = float('inf')
        self.trajectory = []
        self.restart_history = []

    def _wrapped_objective(self, x: np.ndarray) -> float:
        """Wrapped objective that updates tracking."""
        f = self.objective(x)
        self.evaluations += 1

        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
            self.trajectory.append((self.evaluations, self.best_f))

        return f

    def optimize(self, max_evaluations: int) -> PortfolioResult:
        """
        Run adaptive restart optimization.

        Population doubles on each restart (IPOP strategy).
        """
        sub_results = {"restarts": []}

        base_lambda = 4 + int(3 * np.log(self.dim))
        current_lambda = base_lambda

        restart_count = 0

        while self.evaluations < max_evaluations and restart_count <= self.max_restarts:
            remaining = max_evaluations - self.evaluations
            if remaining < 2 * current_lambda:
                break

            # Configure DCMA with current population size
            config = DCMAConfig(
                lambda_=current_lambda,
                seed=42 + restart_count
            )

            portfolio = COCOPortfolio(
                self._wrapped_objective,
                self.dim,
                self.bounds,
                dcma_config=config
            )

            # Allocate budget for this restart
            # Later restarts get more budget due to larger population
            budget_fraction = min(0.5, current_lambda / (base_lambda * 4))
            restart_budget = max(
                2 * current_lambda,
                int(remaining * budget_fraction)
            )
            restart_budget = min(restart_budget, remaining)

            result = portfolio.optimize(restart_budget)

            self.restart_history.append({
                "restart": restart_count,
                "lambda": current_lambda,
                "budget": restart_budget,
                "f_best": result.f_best,
                "strategy": result.strategy_used
            })
            sub_results["restarts"].append(self.restart_history[-1])

            # Check if we hit target
            if result.f_best < 1e-8:
                break

            # Check for stagnation
            if restart_count > 0:
                prev_best = self.restart_history[-2]["f_best"]
                if result.f_best >= prev_best * 0.99:
                    # Stagnating, increase population more aggressively
                    current_lambda = int(current_lambda * 2.5)
                else:
                    # Making progress, standard doubling
                    current_lambda = int(current_lambda * 2)
            else:
                current_lambda = int(current_lambda * 2)

            restart_count += 1

        receipt_hash = hashlib.sha256(
            f"{self.best_x.tolist() if self.best_x is not None else 'None'}:{self.best_f}:{self.evaluations}".encode()
        ).hexdigest()[:16]

        converged = self.best_f < 1e-8
        reason = "target_reached" if converged else "max_evals"

        return PortfolioResult(
            x_best=self.best_x,
            f_best=self.best_f,
            evaluations=self.evaluations,
            trajectory=self.trajectory.copy(),
            strategy_used="adaptive_restart",
            converged=converged,
            reason=reason,
            receipt_hash=receipt_hash,
            sub_results=sub_results
        )

    def reset(self):
        """Reset optimizer state."""
        self.evaluations = 0
        self.best_x = None
        self.best_f = float('inf')
        self.trajectory = []
        self.restart_history = []
