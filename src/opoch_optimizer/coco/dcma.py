"""
Deterministic Covariance Matrix Adaptation (DCMA)

A deterministic variant of CMA-ES that achieves:
1. Rotational invariance (handles unknown rotations)
2. Metric learning (handles ill-conditioning)
3. Full determinism and replay capability

Uses quasi-random Sobol sequences instead of Gaussian samples,
with rank-based updates that depend only on ordering, not coordinates.
"""

import numpy as np
from scipy.stats import norm
from scipy.linalg import cholesky, solve_triangular
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Optional, Dict, Any
import hashlib


@dataclass
class DCMAConfig:
    """Configuration for DCMA."""
    # Population size (lambda)
    # Default: 4 + floor(3 * ln(d))
    lambda_: Optional[int] = None

    # Number of parents (mu)
    # Default: lambda // 2
    mu: Optional[int] = None

    # Initial step size
    sigma0: float = 0.3

    # Bounds handling
    bounds_handling: str = "projection"  # "projection" or "penalty"

    # Stopping criteria
    tol_fun: float = 1e-12
    tol_x: float = 1e-12
    max_stagnation: int = 100

    # Determinism
    seed: int = 42


@dataclass
class DCMAResult:
    """Result of DCMA optimization."""
    x_best: np.ndarray
    f_best: float
    evaluations: int
    trajectory: List[Tuple[int, float]]
    converged: bool
    reason: str
    receipt_hash: str


class SobolGaussianGenerator:
    """
    Generates deterministic Gaussian-like samples using Sobol sequence.
    Maps Sobol [0,1]^d points through inverse normal CDF.
    """

    def __init__(self, dim: int, seed: int = 42):
        self.dim = dim
        self.seed = seed
        self._index = 0

        try:
            from scipy.stats import qmc
            self._engine = qmc.Sobol(d=dim, scramble=True, seed=seed)
        except ImportError:
            self._engine = None

    def generate(self, n: int) -> np.ndarray:
        """Generate n quasi-Gaussian samples."""
        if self._engine is not None:
            # Use scipy Sobol
            u = self._engine.random(n)
        else:
            # Fallback Halton-like
            u = self._halton_samples(n)

        # Map through inverse normal CDF
        # Clip to avoid infinities at 0 and 1
        u = np.clip(u, 1e-10, 1 - 1e-10)
        z = norm.ppf(u)

        self._index += n
        return z

    def _halton_samples(self, n: int) -> np.ndarray:
        """Fallback Halton sequence."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                  53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

        samples = np.zeros((n, self.dim))
        for j in range(self.dim):
            base = primes[j % len(primes)]
            for i in range(n):
                idx = self._index + i + 1
                f = 1.0 / base
                r = 0.0
                while idx > 0:
                    r += f * (idx % base)
                    idx //= base
                    f /= base
                samples[i, j] = r
        return samples

    def reset(self):
        """Reset generator."""
        self._index = 0
        if self._engine is not None:
            try:
                from scipy.stats import qmc
                self._engine = qmc.Sobol(d=self.dim, scramble=True, seed=self.seed)
            except:
                pass


class DCMA:
    """
    Deterministic Covariance Matrix Adaptation.

    A deterministic variant of CMA-ES using:
    - Sobol sequence for quasi-random sampling
    - Inverse normal CDF for Gaussian-like distribution
    - Rank-based updates (rotationally invariant)
    - Antithetic sampling for variance reduction
    """

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        dim: int,
        bounds: List[Tuple[float, float]],
        config: DCMAConfig = None
    ):
        self.objective = objective
        self.dim = dim
        self.bounds = bounds
        self.lb = np.array([b[0] for b in bounds])
        self.ub = np.array([b[1] for b in bounds])
        self.config = config or DCMAConfig()

        # Set population size
        if self.config.lambda_ is None:
            self.lambda_ = 4 + int(3 * np.log(dim))
        else:
            self.lambda_ = self.config.lambda_

        # Ensure even for antithetic sampling
        if self.lambda_ % 2 == 1:
            self.lambda_ += 1

        # Set number of parents
        if self.config.mu is None:
            self.mu = self.lambda_ // 2
        else:
            self.mu = self.config.mu

        # Compute weights (log-linear)
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)
        self.mueff = 1.0 / np.sum(self.weights ** 2)

        # Learning rates (standard CMA values)
        self.cc = (4 + self.mueff / dim) / (dim + 4 + 2 * self.mueff / dim)
        self.cs = (self.mueff + 2) / (dim + self.mueff + 5)
        self.c1 = 2 / ((dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1,
                       2 * (self.mueff - 2 + 1 / self.mueff) / ((dim + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (dim + 1)) - 1) + self.cs

        # Expected norm of N(0,I) vector
        self.chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialize state
        self._initialize_state()

        # Sobol generator
        self.sobol = SobolGaussianGenerator(dim, self.config.seed)

        # Tracking
        self.evaluations = 0
        self.best_x = None
        self.best_f = float('inf')
        self.trajectory = []
        self.receipt_chain = []

    def _initialize_state(self):
        """Initialize CMA state variables."""
        # Mean (start at center of bounds)
        self.mean = (self.lb + self.ub) / 2

        # Step size
        self.sigma = self.config.sigma0 * np.mean(self.ub - self.lb)

        # Covariance matrix (identity)
        self.C = np.eye(self.dim)
        self.A = np.eye(self.dim)  # Cholesky factor

        # Evolution paths
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)

        # Stagnation counter
        self.stagnation = 0
        self.last_best = float('inf')

    def _project_to_bounds(self, x: np.ndarray) -> np.ndarray:
        """Project point to feasible region."""
        return np.clip(x, self.lb, self.ub)

    def _generate_candidates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate lambda candidate solutions deterministically.
        Uses antithetic sampling for variance reduction.
        """
        half_lambda = self.lambda_ // 2

        # Generate half the samples from Sobol-Gaussian
        z_half = self.sobol.generate(half_lambda)

        # Antithetic samples
        z = np.vstack([z_half, -z_half])

        # Transform through covariance
        # x = mean + sigma * A * z
        candidates = np.zeros((self.lambda_, self.dim))
        for i in range(self.lambda_):
            y = self.A @ z[i]
            x = self.mean + self.sigma * y
            candidates[i] = self._project_to_bounds(x)

        return candidates, z

    def _evaluate_candidates(self, candidates: np.ndarray) -> np.ndarray:
        """Evaluate all candidates."""
        n_candidates = len(candidates)
        f_values = np.zeros(n_candidates)
        for i in range(n_candidates):
            f_values[i] = self.objective(candidates[i])
            self.evaluations += 1

            # Update best
            if f_values[i] < self.best_f:
                self.best_f = f_values[i]
                self.best_x = candidates[i].copy()
                self.trajectory.append((self.evaluations, self.best_f))

        return f_values

    def _update_distribution(
        self,
        candidates: np.ndarray,
        z: np.ndarray,
        f_values: np.ndarray
    ):
        """
        Update mean, covariance, and step size based on ranked results.
        This is the core CMA update - rotationally invariant.
        """
        # Handle case where we have fewer candidates than expected
        n_candidates = len(f_values)
        if n_candidates < 2:
            return  # Not enough candidates to update

        # Sort by fitness (ascending = minimization)
        idx = np.argsort(f_values)

        # Select best mu candidates (or all if fewer available)
        mu_eff = min(self.mu, n_candidates)
        x_sel = candidates[idx[:mu_eff]]
        z_sel = z[idx[:mu_eff]]

        # Recompute weights for actual number of selected
        if mu_eff < self.mu:
            weights = np.log(mu_eff + 0.5) - np.log(np.arange(1, mu_eff + 1))
            weights = weights / np.sum(weights)
        else:
            weights = self.weights

        # Old mean for path update
        mean_old = self.mean.copy()

        # Update mean (weighted recombination)
        self.mean = np.sum(weights[:, None] * x_sel, axis=0)

        # Evolution path for covariance
        y_mean = (self.mean - mean_old) / self.sigma

        # ps update (conjugate evolution path)
        # Use inverse of sqrt(C) which is A^{-1}
        try:
            z_mean = solve_triangular(self.A, y_mean, lower=True)
        except:
            z_mean = np.linalg.lstsq(self.A, y_mean, rcond=None)[0]

        # Effective mu for current weights
        mueff_local = 1.0 / np.sum(weights ** 2)
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * mueff_local) * z_mean

        # Heaviside function for stalling detection
        hsig = (np.linalg.norm(self.ps) /
                np.sqrt(1 - (1 - self.cs) ** (2 * self.evaluations / self.lambda_)) /
                self.chiN < 1.4 + 2 / (self.dim + 1))

        # pc update
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * mueff_local) * y_mean

        # Covariance update
        # Rank-one update
        C_rank1 = np.outer(self.pc, self.pc)

        # Rank-mu update
        C_rankmu = np.zeros((self.dim, self.dim))
        for i in range(mu_eff):
            y_i = (x_sel[i] - mean_old) / self.sigma
            C_rankmu += weights[i] * np.outer(y_i, y_i)

        # Combined update
        self.C = ((1 - self.c1 - self.cmu + (1 - hsig) * self.c1 * self.cc * (2 - self.cc)) * self.C +
                  self.c1 * C_rank1 +
                  self.cmu * C_rankmu)

        # Ensure symmetry
        self.C = (self.C + self.C.T) / 2

        # Update Cholesky factor
        try:
            self.A = cholesky(self.C, lower=True)
        except np.linalg.LinAlgError:
            # If Cholesky fails, regularize
            eigvals = np.linalg.eigvalsh(self.C)
            min_eig = np.min(eigvals)
            if min_eig < 1e-10:
                self.C += (1e-10 - min_eig) * np.eye(self.dim)
            self.A = cholesky(self.C, lower=True)

        # Step size update
        self.sigma *= np.exp((self.cs / self.damps) *
                             (np.linalg.norm(self.ps) / self.chiN - 1))

        # Bound sigma
        self.sigma = np.clip(self.sigma, 1e-20, 1e10)

    def _check_convergence(self) -> Tuple[bool, str]:
        """Check convergence criteria."""
        # Function value tolerance
        if self.best_f < self.config.tol_fun:
            return True, "f_tol"

        # Stagnation
        if abs(self.best_f - self.last_best) < self.config.tol_fun:
            self.stagnation += 1
        else:
            self.stagnation = 0
        self.last_best = self.best_f

        if self.stagnation >= self.config.max_stagnation:
            return True, "stagnation"

        # Sigma too small
        if self.sigma < self.config.tol_x:
            return True, "sigma_tol"

        return False, ""

    def _compute_receipt_hash(self) -> str:
        """Compute hash for current state."""
        data = (f"{self.mean.tolist()}:{self.sigma}:{self.best_f}:"
                f"{self.evaluations}:{self.C.flatten().tolist()}")
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def optimize(self, max_evaluations: int) -> DCMAResult:
        """
        Run DCMA optimization.

        Args:
            max_evaluations: Maximum function evaluations

        Returns:
            DCMAResult with best solution and trajectory
        """
        converged = False
        reason = "max_evals"

        while self.evaluations < max_evaluations:
            # Generate candidates
            candidates, z = self._generate_candidates()

            # Check if we'll exceed budget
            if self.evaluations + self.lambda_ > max_evaluations:
                # Evaluate remaining budget
                remaining = max_evaluations - self.evaluations
                candidates = candidates[:remaining]
                z = z[:remaining]
                f_values = self._evaluate_candidates(candidates)
                break

            # Evaluate
            f_values = self._evaluate_candidates(candidates)

            # Update distribution
            self._update_distribution(candidates, z, f_values)

            # Record receipt
            self.receipt_chain.append(self._compute_receipt_hash())

            # Check convergence
            converged, reason = self._check_convergence()
            if converged:
                break

        # Final receipt hash
        final_hash = hashlib.sha256(
            ":".join(self.receipt_chain).encode()
        ).hexdigest()

        return DCMAResult(
            x_best=self.best_x,
            f_best=self.best_f,
            evaluations=self.evaluations,
            trajectory=self.trajectory.copy(),
            converged=converged,
            reason=reason,
            receipt_hash=final_hash
        )

    def reset(self):
        """Reset optimizer state."""
        self._initialize_state()
        self.sobol.reset()
        self.evaluations = 0
        self.best_x = None
        self.best_f = float('inf')
        self.trajectory = []
        self.receipt_chain = []
