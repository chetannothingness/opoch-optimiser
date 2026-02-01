"""
Quadratic Function Identification

For strictly quadratic functions f(x) = 0.5 * x'Hx + b'x + c,
we can reconstruct H and b from O(d^2) evaluations and solve exactly.

This dominates CMA-ES on ill-conditioned quadratics (BBOB f1, f2, f10).
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional
import hashlib


@dataclass
class QuadraticResult:
    """Result of quadratic identification."""
    x_optimal: Optional[np.ndarray]
    f_optimal: float
    is_quadratic: bool
    confidence: float
    evaluations: int
    H: Optional[np.ndarray]
    b: Optional[np.ndarray]
    receipt_hash: str


class QuadraticIdentifier:
    """
    Identifies and solves quadratic functions exactly.

    For f(x) = 0.5 * x'Hx + b'x + c:
    1. Estimate Hessian H via finite differences
    2. Estimate gradient/linear term b
    3. Solve x* = -H^{-1} b
    4. Verify quadratic structure

    Complexity: O(d^2) evaluations for full identification.
    """

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        dim: int,
        bounds: List[Tuple[float, float]],
        h: float = 1e-4
    ):
        """
        Initialize quadratic identifier.

        Args:
            objective: Function to identify
            dim: Dimension
            bounds: Variable bounds
            h: Finite difference step size
        """
        self.objective = objective
        self.dim = dim
        self.bounds = bounds
        self.lb = np.array([b[0] for b in bounds])
        self.ub = np.array([b[1] for b in bounds])
        self.h = h

        self.evaluations = 0
        self.f_cache = {}

    def _eval(self, x: np.ndarray) -> float:
        """Evaluate with caching."""
        key = tuple(np.round(x, 10))
        if key not in self.f_cache:
            self.f_cache[key] = self.objective(x)
            self.evaluations += 1
        return self.f_cache[key]

    def _project(self, x: np.ndarray) -> np.ndarray:
        """Project to bounds."""
        return np.clip(x, self.lb, self.ub)

    def estimate_hessian(self, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Estimate Hessian and gradient at x0 using central differences.

        Returns:
            H: Estimated Hessian matrix
            g: Estimated gradient
            f0: Function value at x0
        """
        h = self.h
        f0 = self._eval(x0)

        # Gradient estimation
        g = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus = x0.copy()
            x_plus[i] += h
            x_minus = x0.copy()
            x_minus[i] -= h

            f_plus = self._eval(self._project(x_plus))
            f_minus = self._eval(self._project(x_minus))

            g[i] = (f_plus - f_minus) / (2 * h)

        # Hessian estimation
        H = np.zeros((self.dim, self.dim))

        # Diagonal elements
        for i in range(self.dim):
            x_plus = x0.copy()
            x_plus[i] += h
            x_minus = x0.copy()
            x_minus[i] -= h

            f_plus = self._eval(self._project(x_plus))
            f_minus = self._eval(self._project(x_minus))

            H[i, i] = (f_plus - 2 * f0 + f_minus) / (h ** 2)

        # Off-diagonal elements
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                x_pp = x0.copy()
                x_pp[i] += h
                x_pp[j] += h

                x_pm = x0.copy()
                x_pm[i] += h
                x_pm[j] -= h

                x_mp = x0.copy()
                x_mp[i] -= h
                x_mp[j] += h

                x_mm = x0.copy()
                x_mm[i] -= h
                x_mm[j] -= h

                f_pp = self._eval(self._project(x_pp))
                f_pm = self._eval(self._project(x_pm))
                f_mp = self._eval(self._project(x_mp))
                f_mm = self._eval(self._project(x_mm))

                H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * h ** 2)
                H[j, i] = H[i, j]

        return H, g, f0

    def check_quadratic(
        self,
        x0: np.ndarray,
        H: np.ndarray,
        g: np.ndarray,
        f0: float,
        n_test: int = 5,
        tol: float = 1e-4
    ) -> Tuple[bool, float]:
        """
        Check if function is quadratic by testing predictions.

        Args:
            x0: Base point
            H: Estimated Hessian
            g: Estimated gradient at x0
            f0: Function value at x0
            n_test: Number of test points
            tol: Tolerance for quadratic fit

        Returns:
            (is_quadratic, confidence): Whether function appears quadratic
        """
        # Generate test points
        np.random.seed(42)  # Deterministic
        errors = []

        for _ in range(n_test):
            # Random direction
            direction = np.random.randn(self.dim)
            direction = direction / np.linalg.norm(direction)

            # Random step
            step = np.random.uniform(0.1, 0.5)
            x_test = self._project(x0 + step * direction)

            # Actual value
            f_actual = self._eval(x_test)

            # Predicted value from quadratic model
            dx = x_test - x0
            f_predicted = f0 + g @ dx + 0.5 * dx @ H @ dx

            # Relative error
            if abs(f_actual) > 1e-10:
                rel_error = abs(f_actual - f_predicted) / abs(f_actual)
            else:
                rel_error = abs(f_actual - f_predicted)

            errors.append(rel_error)

        avg_error = np.mean(errors)
        max_error = np.max(errors)

        is_quadratic = max_error < tol
        confidence = 1.0 - min(1.0, avg_error)

        return is_quadratic, confidence

    def solve_quadratic(
        self,
        H: np.ndarray,
        g: np.ndarray,
        x0: np.ndarray,
        f0: float
    ) -> Tuple[np.ndarray, float]:
        """
        Solve the quadratic optimization problem.

        For f(x) = f0 + g'(x-x0) + 0.5*(x-x0)'H(x-x0)
        Optimum at x* = x0 - H^{-1} g

        Returns:
            x_opt: Optimal point (projected to bounds)
            f_opt: Optimal value
        """
        try:
            # Solve H * dx = -g
            dx = np.linalg.solve(H, -g)
            x_opt = x0 + dx

            # Project to bounds
            x_opt = self._project(x_opt)

            # Evaluate actual function value
            f_opt = self._eval(x_opt)

            return x_opt, f_opt

        except np.linalg.LinAlgError:
            # Singular Hessian - use pseudoinverse
            try:
                dx = np.linalg.lstsq(H, -g, rcond=None)[0]
                x_opt = self._project(x0 + dx)
                f_opt = self._eval(x_opt)
                return x_opt, f_opt
            except:
                return x0, f0

    def identify_and_solve(
        self,
        x0: Optional[np.ndarray] = None,
        verify: bool = True
    ) -> QuadraticResult:
        """
        Full quadratic identification and solution.

        Args:
            x0: Starting point (default: center of bounds)
            verify: Whether to verify quadratic structure

        Returns:
            QuadraticResult with solution or failure indication
        """
        if x0 is None:
            x0 = (self.lb + self.ub) / 2

        # Estimate Hessian and gradient
        H, g, f0 = self.estimate_hessian(x0)

        # Check if quadratic (optional)
        if verify:
            is_quadratic, confidence = self.check_quadratic(x0, H, g, f0)
        else:
            is_quadratic = True
            confidence = 1.0

        # Solve
        x_opt, f_opt = self.solve_quadratic(H, g, x0, f0)

        # Compute receipt hash
        receipt_data = f"{x_opt.tolist()}:{f_opt}:{self.evaluations}"
        receipt_hash = hashlib.sha256(receipt_data.encode()).hexdigest()[:16]

        return QuadraticResult(
            x_optimal=x_opt,
            f_optimal=f_opt,
            is_quadratic=is_quadratic,
            confidence=confidence,
            evaluations=self.evaluations,
            H=H,
            b=g,
            receipt_hash=receipt_hash
        )

    def quick_detect(self, n_samples: int = 10, tol: float = 0.01) -> bool:
        """
        Quickly detect if function is likely quadratic.

        Uses cheap heuristic: check if second derivatives are constant.

        Args:
            n_samples: Number of sample points
            tol: Tolerance for constancy

        Returns:
            True if function appears quadratic
        """
        if self.dim > 20:
            # Too expensive for high dimensions
            return False

        h = self.h
        center = (self.lb + self.ub) / 2

        # Sample second derivative at center
        H_center = np.zeros(self.dim)
        f_center = self._eval(center)

        for i in range(min(self.dim, 5)):  # Only check first 5 dims
            x_plus = center.copy()
            x_plus[i] += h
            x_minus = center.copy()
            x_minus[i] -= h

            f_plus = self._eval(self._project(x_plus))
            f_minus = self._eval(self._project(x_minus))

            H_center[i] = (f_plus - 2 * f_center + f_minus) / (h ** 2)

        # Sample second derivative at another point
        np.random.seed(42)
        offset = np.random.uniform(-0.3, 0.3, self.dim) * (self.ub - self.lb)
        point2 = self._project(center + offset)

        H_point2 = np.zeros(self.dim)
        f_point2 = self._eval(point2)

        for i in range(min(self.dim, 5)):
            x_plus = point2.copy()
            x_plus[i] += h
            x_minus = point2.copy()
            x_minus[i] -= h

            f_plus = self._eval(self._project(x_plus))
            f_minus = self._eval(self._project(x_minus))

            H_point2[i] = (f_plus - 2 * f_point2 + f_minus) / (h ** 2)

        # Check if second derivatives are approximately constant
        diff = np.abs(H_center[:5] - H_point2[:5])
        scale = np.maximum(np.abs(H_center[:5]), 1e-10)
        rel_diff = diff / scale

        return np.max(rel_diff) < tol

    def reset(self):
        """Reset identifier state."""
        self.evaluations = 0
        self.f_cache = {}
