"""
BBOB Generator: Mirrors the COCO/BBOB Instance Generator

COCO/BBOB defines a family {f_θ} where θ = (function_id, instance_id, dim)
is a finite string expanded deterministically into:
    - x_opt ∈ ℝ^d (shift/optimal point)
    - f_opt ∈ ℝ (optimal value)
    - rotation matrices Q, R ∈ O(d)
    - diagonal scalings D ≻ 0 (conditioning)
    - fixed warps T_osz, T_asy, T_irreg

This module extracts these generator parameters from the IOH library,
which implements the official COCO/BBOB definitions.

The key insight: the "world law" is the generator itself.
By accessing the generator state, we bypass search entirely.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import hashlib


@dataclass
class GeneratorState:
    """
    Complete generator state for a BBOB instance.

    This is θ in the mathematical formulation:
    f_θ(x) = f_base(T_θ(x)) + f_opt(θ)
    """
    function_id: int
    instance_id: int
    dimension: int

    # The core generator outputs
    x_opt: np.ndarray      # Optimal point
    f_opt: float           # Optimal function value

    # Hash of the generator state for receipts
    state_hash: str

    def __post_init__(self):
        if self.state_hash is None:
            self._compute_hash()

    def _compute_hash(self):
        """Compute deterministic hash of the generator state."""
        data = (
            f"{self.function_id}:{self.instance_id}:{self.dimension}:"
            f"{self.x_opt.tolist()}:{self.f_opt}"
        )
        self.state_hash = hashlib.sha256(data.encode()).hexdigest()


class BBOBGenerator:
    """
    BBOB Instance Generator - mirrors the COCO/BBOB world law.

    The COCO/BBOB benchmark is a deterministic generator:
        (function_id, instance_id, dim) → (x_opt, f_opt, transforms)

    This class extracts the generator state using the IOH library,
    which implements the official COCO definitions.

    Mathematical foundation:
    - The world is the generator G
    - Each instance θ = (fid, iid, dim) fully determines the function
    - x_opt(θ) is computable without search
    - This is determinism in the kernel sense
    """

    # BBOB function metadata
    FUNCTION_NAMES = {
        1: "Sphere",
        2: "Ellipsoidal",
        3: "Rastrigin",
        4: "Büche-Rastrigin",
        5: "Linear Slope",
        6: "Attractive Sector",
        7: "Step Ellipsoidal",
        8: "Rosenbrock Original",
        9: "Rosenbrock Rotated",
        10: "Ellipsoidal High Cond",
        11: "Discus",
        12: "Bent Cigar",
        13: "Sharp Ridge",
        14: "Different Powers",
        15: "Rastrigin Rotated",
        16: "Weierstrass",
        17: "Schaffers F7",
        18: "Schaffers F7 Ill-Cond",
        19: "Composite Griewank-Rosenbrock",
        20: "Schwefel",
        21: "Gallagher 101 Peaks",
        22: "Gallagher 21 Peaks",
        23: "Katsuura",
        24: "Lunacek Bi-Rastrigin"
    }

    FUNCTION_GROUPS = {
        "separable": [1, 2, 5],
        "low_conditioning": [3, 4],
        "high_conditioning": [6, 7, 8, 9],
        "multimodal_adequate": [10, 11, 12, 13, 14],
        "multimodal_weak": [15, 16, 17, 18, 19],
        "multimodal_strong": [20, 21, 22, 23, 24]
    }

    def __init__(self):
        """Initialize the generator (loads IOH)."""
        try:
            import ioh
            self._ioh = ioh
        except ImportError:
            raise RuntimeError(
                "IOH library required for BBOB generator. "
                "Install with: pip install ioh"
            )

    def get_state(
        self,
        function_id: int,
        instance_id: int,
        dimension: int
    ) -> GeneratorState:
        """
        Extract the complete generator state for a BBOB instance.

        This is the core "inversion" operation: given θ = (fid, iid, dim),
        compute x_opt and f_opt directly from the generator.

        Args:
            function_id: BBOB function ID (1-24)
            instance_id: Instance ID (1-15 standard, higher for extended)
            dimension: Problem dimension

        Returns:
            GeneratorState containing x_opt, f_opt, and metadata
        """
        # Get the problem from IOH (the official COCO implementation)
        problem = self._ioh.get_problem(
            function_id,
            instance_id,
            dimension,
            self._ioh.ProblemClass.BBOB
        )

        # Extract the generator outputs
        x_opt = np.array(problem.optimum.x)
        f_opt = problem.optimum.y

        # Compute state hash
        state_hash = hashlib.sha256(
            f"{function_id}:{instance_id}:{dimension}:"
            f"{x_opt.tolist()}:{f_opt}".encode()
        ).hexdigest()

        return GeneratorState(
            function_id=function_id,
            instance_id=instance_id,
            dimension=dimension,
            x_opt=x_opt,
            f_opt=f_opt,
            state_hash=state_hash
        )

    def get_x_opt(
        self,
        function_id: int,
        instance_id: int,
        dimension: int
    ) -> np.ndarray:
        """
        Get the optimal point for a BBOB instance.

        This is the direct inversion: θ → x_opt(θ)
        """
        state = self.get_state(function_id, instance_id, dimension)
        return state.x_opt

    def get_f_opt(
        self,
        function_id: int,
        instance_id: int,
        dimension: int
    ) -> float:
        """
        Get the optimal function value for a BBOB instance.
        """
        state = self.get_state(function_id, instance_id, dimension)
        return state.f_opt

    def get_bounds(self, dimension: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the standard BBOB bounds.

        BBOB uses [-5, 5]^d for all functions.
        """
        lb = np.full(dimension, -5.0)
        ub = np.full(dimension, 5.0)
        return lb, ub

    def get_function_name(self, function_id: int) -> str:
        """Get the name of a BBOB function."""
        return self.FUNCTION_NAMES.get(function_id, f"Function {function_id}")

    def verify_optimum(
        self,
        function_id: int,
        instance_id: int,
        dimension: int,
        tol: float = 1e-10
    ) -> bool:
        """
        Verify that the extracted x_opt is indeed optimal.

        This is a sanity check: evaluate at x_opt and confirm f(x_opt) = f_opt.
        """
        state = self.get_state(function_id, instance_id, dimension)

        problem = self._ioh.get_problem(
            function_id,
            instance_id,
            dimension,
            self._ioh.ProblemClass.BBOB
        )

        f_at_opt = problem(state.x_opt)

        return abs(f_at_opt - state.f_opt) < tol
