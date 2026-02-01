"""
Multi-Start Deterministic CMA (MS-DCMA)

For multimodal functions, a single DCMA run converges to one local optimum.
Mathematical solution: Launch multiple independent DCMA searches from
Sobol-distributed starting points to deterministically cover all basins.

Key insight: Sobol sequence provides optimal space-filling coverage,
guaranteeing that starting points sample all major basins.
"""

import numpy as np
from scipy.stats import qmc
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Dict
import hashlib

from .dcma import DCMA, DCMAConfig, DCMAResult
from .quadratic_id import QuadraticIdentifier


@dataclass
class MultiStartResult:
    """Result of multi-start optimization."""
    x_best: np.ndarray
    f_best: float
    evaluations: int
    trajectory: List[Tuple[int, float]]
    n_starts: int
    successful_starts: int
    best_start_idx: int
    receipt_hash: str


class MultiStartDCMA:
    """
    Multi-Start Deterministic CMA for multimodal functions.

    Uses Sobol sequence to generate starting points that uniformly
    cover the search space, then runs independent DCMA from each.

    For a function with K global basins, O(K) starting points
    statistically guarantee finding the global optimum.

    The number of starts scales with dimension:
    - Low dim (d <= 5): More starts (8-16) since budget allows
    - Medium dim (5 < d <= 20): Moderate starts (4-8)
    - High dim (d > 20): Fewer starts (2-4) to preserve per-run budget
    """

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        dim: int,
        bounds: List[Tuple[float, float]],
        seed: int = 42
    ):
        self.objective = objective
        self.dim = dim
        self.bounds = bounds
        self.lb = np.array([b[0] for b in bounds])
        self.ub = np.array([b[1] for b in bounds])
        self.seed = seed

        # Tracking
        self.evaluations = 0
        self.best_x = None
        self.best_f = float('inf')
        self.trajectory = []

    def _generate_starting_points(self, n_starts: int) -> np.ndarray:
        """
        Generate n_starts points using Sobol sequence.

        Sobol provides low-discrepancy coverage - mathematically
        guaranteed to fill the space more uniformly than random.
        """
        try:
            sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.seed)
            # Sample in [0, 1]^d
            samples = sampler.random(n_starts)
        except:
            # Fallback to stratified random
            np.random.seed(self.seed)
            samples = np.random.rand(n_starts, self.dim)

        # Scale to bounds
        starting_points = self.lb + samples * (self.ub - self.lb)

        return starting_points

    def _wrapped_objective(self, x: np.ndarray) -> float:
        """Wrapped objective that updates global tracking."""
        f = self.objective(x)
        self.evaluations += 1

        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
            self.trajectory.append((self.evaluations, self.best_f))

        return f

    def _determine_n_starts(self, budget: int) -> int:
        """
        Determine optimal number of starts based on dimension and budget.

        Mathematical reasoning:
        - Each DCMA run needs ~O(d^2) evaluations for covariance learning
        - Multimodal functions typically have O(2^d) local optima
        - We can't cover all, but Sobol ensures we sample major basins
        """
        # Minimum budget per DCMA run for meaningful optimization
        min_budget_per_run = max(100, 10 * self.dim)

        # Maximum starts based on dimension
        if self.dim <= 3:
            max_starts = 32
        elif self.dim <= 5:
            max_starts = 16
        elif self.dim <= 10:
            max_starts = 8
        elif self.dim <= 20:
            max_starts = 4
        else:
            max_starts = 2

        # Actual starts limited by budget
        n_starts = min(max_starts, budget // min_budget_per_run)

        return max(1, n_starts)

    def optimize(self, max_evaluations: int) -> MultiStartResult:
        """
        Run multi-start optimization.

        Strategy:
        1. Determine number of starts based on budget
        2. Generate Sobol-distributed starting points
        3. Run DCMA from each, tracking best result
        4. Return global best
        """
        n_starts = self._determine_n_starts(max_evaluations)
        starting_points = self._generate_starting_points(n_starts)

        budget_per_start = max_evaluations // n_starts
        successful_starts = 0
        best_start_idx = 0

        for i, start_point in enumerate(starting_points):
            if self.evaluations >= max_evaluations:
                break

            remaining = max_evaluations - self.evaluations
            run_budget = min(budget_per_start, remaining)

            if run_budget < 10:
                break

            # Configure DCMA for this run
            config = DCMAConfig(
                sigma0=0.3,
                seed=self.seed + i
            )

            dcma = DCMA(
                self._wrapped_objective,
                self.dim,
                self.bounds,
                config
            )

            # Set starting point
            dcma.mean = start_point.copy()

            # Run optimization
            result = dcma.optimize(run_budget)

            successful_starts += 1

            # Check if this is the best run
            if result.f_best < self.best_f:
                best_start_idx = i

        # Compute receipt hash
        receipt_hash = hashlib.sha256(
            f"{self.best_x.tolist() if self.best_x is not None else 'None'}:{self.best_f}:{self.evaluations}".encode()
        ).hexdigest()[:16]

        return MultiStartResult(
            x_best=self.best_x,
            f_best=self.best_f,
            evaluations=self.evaluations,
            trajectory=self.trajectory.copy(),
            n_starts=n_starts,
            successful_starts=successful_starts,
            best_start_idx=best_start_idx,
            receipt_hash=receipt_hash
        )

    def reset(self):
        """Reset optimizer state."""
        self.evaluations = 0
        self.best_x = None
        self.best_f = float('inf')
        self.trajectory = []


class AdaptiveMultiStartDCMA:
    """
    Adaptive Multi-Start DCMA with intelligent budget allocation.

    Phase 1: Quick quadratic check (O(d) evals)
    Phase 2: If quadratic, solve exactly
    Phase 3: If unimodal detected, single DCMA run
    Phase 4: If multimodal, multi-start DCMA

    Detection of modality:
    - Sample function at Sobol points
    - Check variance of local gradients
    - High variance = likely multimodal
    """

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        dim: int,
        bounds: List[Tuple[float, float]],
        seed: int = 42
    ):
        self.objective = objective
        self.dim = dim
        self.bounds = bounds
        self.lb = np.array([b[0] for b in bounds])
        self.ub = np.array([b[1] for b in bounds])
        self.seed = seed

        self.evaluations = 0
        self.best_x = None
        self.best_f = float('inf')
        self.trajectory = []
        self.f_cache = {}

    def _eval(self, x: np.ndarray) -> float:
        """Cached evaluation."""
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

    def _detect_modality(self, n_samples: int = 10) -> str:
        """
        Detect if function is likely unimodal or multimodal.

        Method: Sample gradient directions at multiple points.
        - Unimodal: Gradients roughly point toward same region
        - Multimodal: Gradients point in diverse directions

        Returns: "quadratic", "unimodal", or "multimodal"
        """
        try:
            sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.seed)
            samples = sampler.random(n_samples)
        except:
            np.random.seed(self.seed)
            samples = np.random.rand(n_samples, self.dim)

        points = self.lb + samples * (self.ub - self.lb)
        h = 1e-4 * np.mean(self.ub - self.lb)

        # Compute gradient directions at each point
        gradient_directions = []
        f_values = []

        for p in points:
            f_p = self._eval(p)
            f_values.append(f_p)

            # Estimate gradient
            grad = np.zeros(self.dim)
            for i in range(min(self.dim, 5)):  # Only first 5 dims for speed
                p_plus = p.copy()
                p_plus[i] = min(p[i] + h, self.ub[i])
                p_minus = p.copy()
                p_minus[i] = max(p[i] - h, self.lb[i])

                f_plus = self._eval(p_plus)
                f_minus = self._eval(p_minus)

                grad[i] = (f_plus - f_minus) / (2 * h)

            norm = np.linalg.norm(grad)
            if norm > 1e-10:
                gradient_directions.append(grad / norm)

        if len(gradient_directions) < 2:
            return "unimodal"

        # Check gradient alignment
        # Compute pairwise dot products
        alignments = []
        for i in range(len(gradient_directions)):
            for j in range(i + 1, len(gradient_directions)):
                dot = np.dot(gradient_directions[i], gradient_directions[j])
                alignments.append(dot)

        if len(alignments) == 0:
            return "unimodal"

        mean_alignment = np.mean(alignments)

        # Check if function values have high variance (multimodal indicator)
        f_std = np.std(f_values)
        f_range = max(f_values) - min(f_values)

        # Decision logic
        if mean_alignment > 0.7:
            # Gradients well-aligned -> likely unimodal
            return "unimodal"
        elif mean_alignment < 0.3 and f_range > 1.0:
            # Gradients diverse and high f variance -> multimodal
            return "multimodal"
        else:
            # Uncertain -> treat as multimodal to be safe
            return "multimodal"

    def optimize(self, max_evaluations: int) -> MultiStartResult:
        """
        Run adaptive optimization.
        """
        # Phase 1: Quick modality detection
        modality = self._detect_modality()

        remaining = max_evaluations - self.evaluations

        # Phase 2: Quadratic check if budget allows
        if remaining > 3 * self.dim ** 2 and self.dim <= 20:
            quad_id = QuadraticIdentifier(
                self._eval,
                self.dim,
                self.bounds,
                h=1e-4
            )

            if quad_id.quick_detect(tol=0.01):
                result = quad_id.identify_and_solve(verify=True)

                if result.is_quadratic and result.confidence > 0.95:
                    return MultiStartResult(
                        x_best=self.best_x,
                        f_best=self.best_f,
                        evaluations=self.evaluations,
                        trajectory=self.trajectory.copy(),
                        n_starts=1,
                        successful_starts=1,
                        best_start_idx=0,
                        receipt_hash=result.receipt_hash
                    )

        remaining = max_evaluations - self.evaluations

        # Phase 3/4: Run appropriate optimizer
        if modality == "unimodal":
            # Single DCMA run
            config = DCMAConfig(seed=self.seed)
            dcma = DCMA(
                self._eval,
                self.dim,
                self.bounds,
                config
            )

            if self.best_x is not None:
                dcma.mean = self.best_x.copy()

            dcma_result = dcma.optimize(remaining)
            n_starts = 1
        else:
            # Multi-start for multimodal
            ms_dcma = MultiStartDCMA(
                self._eval,
                self.dim,
                self.bounds,
                seed=self.seed
            )

            if self.best_x is not None:
                ms_dcma.best_x = self.best_x.copy()
                ms_dcma.best_f = self.best_f

            ms_result = ms_dcma.optimize(remaining)
            n_starts = ms_result.n_starts

        receipt_hash = hashlib.sha256(
            f"{self.best_x.tolist() if self.best_x is not None else 'None'}:{self.best_f}:{self.evaluations}".encode()
        ).hexdigest()[:16]

        return MultiStartResult(
            x_best=self.best_x,
            f_best=self.best_f,
            evaluations=self.evaluations,
            trajectory=self.trajectory.copy(),
            n_starts=n_starts,
            successful_starts=n_starts,
            best_start_idx=0,
            receipt_hash=receipt_hash
        )

    def reset(self):
        """Reset optimizer state."""
        self.evaluations = 0
        self.best_x = None
        self.best_f = float('inf')
        self.trajectory = []
        self.f_cache = {}
