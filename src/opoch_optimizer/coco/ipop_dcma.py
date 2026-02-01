"""
Deterministic IPOP-DCMA (Increasing Population Deterministic CMA)

Complete, kernel-correct multi-restart strategy that maintains:
1. Rotational invariance (restarts sampled through search distribution, not coordinate space)
2. Full determinism and replay capability
3. Interleaved evaluation for optimal anytime performance
4. Two-stage budget allocation (probe then exploit)

Key insight: Multi-start in coordinate space (Sobol in x) reintroduces axis privilege.
Instead, sample restart means as: m_0^(s) = m_base + σ_0 * A_0 * u_s
where u_s is Sobol→Gaussian in the isotropic reference space.
"""

import numpy as np
from scipy.stats import norm, qmc
from scipy.linalg import cholesky, solve_triangular
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Optional, Dict
import hashlib


@dataclass
class IPOPDCMAConfig:
    """Configuration for IPOP-DCMA."""
    # Initial step size (relative to domain)
    sigma0: float = 0.3

    # Maximum number of restarts
    max_restarts: int = 9

    # Probe budget per restart (multiplier of dimension)
    probe_budget_factor: int = 50

    # Top K restarts to exploit (fraction of total restarts)
    top_k_fraction: float = 0.5

    # Population increase factor on restart
    lambda_increase_factor: float = 2.0

    # Seed for determinism
    seed: int = 42

    # Convergence tolerance
    tol_fun: float = 1e-12
    tol_x: float = 1e-12


@dataclass
class IPOPDCMAResult:
    """Result of IPOP-DCMA optimization."""
    x_best: np.ndarray
    f_best: float
    evaluations: int
    trajectory: List[Tuple[int, float]]
    n_restarts: int
    restart_history: List[Dict]
    receipt_hash: str


class RestartState:
    """State for a single DCMA restart."""

    def __init__(
        self,
        dim: int,
        mean: np.ndarray,
        sigma: float,
        lambda_: int,
        restart_id: int,
        seed: int
    ):
        self.dim = dim
        self.restart_id = restart_id
        self.seed = seed

        # Distribution parameters
        self.mean = mean.copy()
        self.sigma = sigma
        self.lambda_ = lambda_
        self.mu = lambda_ // 2

        # Covariance (start with identity - rotationally invariant)
        self.C = np.eye(dim)
        self.A = np.eye(dim)  # Cholesky factor

        # Evolution paths
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)

        # CMA parameters
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)
        self.mueff = 1.0 / np.sum(self.weights ** 2)

        self.cc = (4 + self.mueff / dim) / (dim + 4 + 2 * self.mueff / dim)
        self.cs = (self.mueff + 2) / (dim + self.mueff + 5)
        self.c1 = 2 / ((dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1,
                       2 * (self.mueff - 2 + 1 / self.mueff) / ((dim + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (dim + 1)) - 1) + self.cs
        self.chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Sobol generator for this restart
        try:
            self.sobol = qmc.Sobol(d=dim, scramble=True, seed=seed + restart_id)
        except:
            self.sobol = None
        self.sobol_index = 0

        # Tracking
        self.evaluations = 0
        self.generations = 0
        self.best_f = float('inf')
        self.best_x = None
        self.stagnation = 0
        self.last_best = float('inf')

    def generate_candidates(self, lb: np.ndarray, ub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate candidates using Sobol→Gaussian through search distribution."""
        half_lambda = self.lambda_ // 2

        # Generate Sobol points in [0,1]^d
        if self.sobol is not None:
            u = self.sobol.random(half_lambda)
            u = np.clip(u, 1e-10, 1 - 1e-10)
            z_half = norm.ppf(u)
        else:
            # Fallback: deterministic Halton-like
            z_half = self._halton_gaussian(half_lambda)

        # Antithetic sampling
        z = np.vstack([z_half, -z_half])

        # Transform through search distribution (rotationally invariant)
        # x = mean + sigma * A * z
        candidates = np.zeros((self.lambda_, self.dim))
        for i in range(self.lambda_):
            y = self.A @ z[i]
            x = self.mean + self.sigma * y
            candidates[i] = np.clip(x, lb, ub)

        self.sobol_index += half_lambda
        return candidates, z

    def _halton_gaussian(self, n: int) -> np.ndarray:
        """Fallback Halton sequence mapped to Gaussian."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        samples = np.zeros((n, self.dim))

        for j in range(self.dim):
            base = primes[j % len(primes)]
            for i in range(n):
                idx = self.sobol_index + i + 1
                f = 1.0 / base
                r = 0.0
                while idx > 0:
                    r += f * (idx % base)
                    idx //= base
                    f /= base
                samples[i, j] = r

        samples = np.clip(samples, 1e-10, 1 - 1e-10)
        return norm.ppf(samples)

    def update_distribution(
        self,
        candidates: np.ndarray,
        z: np.ndarray,
        f_values: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray
    ):
        """Update distribution based on ranked results."""
        n_candidates = len(f_values)
        if n_candidates < 2:
            return

        # Sort by fitness
        idx = np.argsort(f_values)

        # Select best mu (or available)
        mu_eff = min(self.mu, n_candidates)
        x_sel = candidates[idx[:mu_eff]]
        z_sel = z[idx[:mu_eff]]

        # Recompute weights if needed
        if mu_eff < self.mu:
            weights = np.log(mu_eff + 0.5) - np.log(np.arange(1, mu_eff + 1))
            weights = weights / np.sum(weights)
            mueff_local = 1.0 / np.sum(weights ** 2)
        else:
            weights = self.weights
            mueff_local = self.mueff

        # Old mean
        mean_old = self.mean.copy()

        # Update mean
        self.mean = np.sum(weights[:, None] * x_sel, axis=0)

        # Evolution path updates
        y_mean = (self.mean - mean_old) / self.sigma

        try:
            z_mean = solve_triangular(self.A, y_mean, lower=True)
        except:
            z_mean = np.linalg.lstsq(self.A, y_mean, rcond=None)[0]

        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * mueff_local) * z_mean

        hsig = (np.linalg.norm(self.ps) /
                np.sqrt(1 - (1 - self.cs) ** (2 * (self.evaluations + 1) / self.lambda_)) /
                self.chiN < 1.4 + 2 / (self.dim + 1))

        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * mueff_local) * y_mean

        # Covariance update
        C_rank1 = np.outer(self.pc, self.pc)

        C_rankmu = np.zeros((self.dim, self.dim))
        for i in range(mu_eff):
            y_i = (x_sel[i] - mean_old) / self.sigma
            C_rankmu += weights[i] * np.outer(y_i, y_i)

        self.C = ((1 - self.c1 - self.cmu + (1 - hsig) * self.c1 * self.cc * (2 - self.cc)) * self.C +
                  self.c1 * C_rank1 +
                  self.cmu * C_rankmu)

        self.C = (self.C + self.C.T) / 2

        # Update Cholesky
        try:
            self.A = cholesky(self.C, lower=True)
        except np.linalg.LinAlgError:
            eigvals = np.linalg.eigvalsh(self.C)
            min_eig = np.min(eigvals)
            if min_eig < 1e-10:
                self.C += (1e-10 - min_eig) * np.eye(self.dim)
            self.A = cholesky(self.C, lower=True)

        # Step size update
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.sigma = np.clip(self.sigma, 1e-20, 1e10)

        self.generations += 1

    def check_stagnation(self, tol_fun: float, max_stag: int = 50) -> bool:
        """Check if restart has stagnated."""
        if abs(self.best_f - self.last_best) < tol_fun:
            self.stagnation += 1
        else:
            self.stagnation = 0
        self.last_best = self.best_f

        return self.stagnation >= max_stag


class IPOPDCMA:
    """
    Deterministic IPOP-DCMA Optimizer.

    Implements rotation-invariant multi-restart with:
    1. Restart means sampled through search distribution (not coordinate space)
    2. Interleaved evaluation across restarts
    3. Two-stage budget allocation (probe then exploit)
    4. IPOP population increase on restart
    """

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        dim: int,
        bounds: List[Tuple[float, float]],
        config: IPOPDCMAConfig = None
    ):
        self.objective = objective
        self.dim = dim
        self.bounds = bounds
        self.lb = np.array([b[0] for b in bounds])
        self.ub = np.array([b[1] for b in bounds])
        self.config = config or IPOPDCMAConfig()

        # Global tracking
        self.evaluations = 0
        self.best_x = None
        self.best_f = float('inf')
        self.trajectory = []

        # Restart tracking
        self.restarts: List[RestartState] = []
        self.restart_history = []

        # Base parameters
        self.base_lambda = 4 + int(3 * np.log(dim))
        if self.base_lambda % 2 == 1:
            self.base_lambda += 1

        self.base_sigma = self.config.sigma0 * np.mean(self.ub - self.lb)
        self.base_mean = (self.lb + self.ub) / 2

        # Generate restart directions using Sobol (in isotropic space)
        self._generate_restart_directions()

    def _generate_restart_directions(self):
        """
        Generate deterministic restart directions in isotropic space.

        These will be transformed through the search distribution
        to maintain rotational invariance.
        """
        try:
            sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.config.seed)
            u = sampler.random(self.config.max_restarts + 1)
            u = np.clip(u, 1e-10, 1 - 1e-10)
            self.restart_directions = norm.ppf(u)
        except:
            # Fallback
            np.random.seed(self.config.seed)
            self.restart_directions = np.random.randn(self.config.max_restarts + 1, self.dim)

    def _create_restart(
        self,
        restart_id: int,
        lambda_: int,
        base_A: np.ndarray = None
    ) -> RestartState:
        """
        Create a new restart with mean sampled through search distribution.

        Key: m_0^(s) = m_base + σ_0 * A_0 * u_s
        This maintains rotational invariance.
        """
        if base_A is None:
            base_A = np.eye(self.dim)

        # Sample restart mean through search distribution
        u_s = self.restart_directions[restart_id]

        if restart_id == 0:
            # First restart: start at center
            mean = self.base_mean.copy()
        else:
            # Subsequent restarts: offset through search distribution
            offset = self.base_sigma * (base_A @ u_s)
            mean = np.clip(self.base_mean + offset, self.lb, self.ub)

        return RestartState(
            dim=self.dim,
            mean=mean,
            sigma=self.base_sigma,
            lambda_=lambda_,
            restart_id=restart_id,
            seed=self.config.seed + restart_id * 1000
        )

    def _evaluate(self, x: np.ndarray) -> float:
        """Evaluate and update global best."""
        f = self.objective(x)
        self.evaluations += 1

        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
            self.trajectory.append((self.evaluations, self.best_f))

        return f

    def _run_generation(self, restart: RestartState) -> List[float]:
        """Run one generation of a restart."""
        candidates, z = restart.generate_candidates(self.lb, self.ub)

        f_values = []
        for i, x in enumerate(candidates):
            f = self._evaluate(x)
            f_values.append(f)
            restart.evaluations += 1

            if f < restart.best_f:
                restart.best_f = f
                restart.best_x = x.copy()

        f_values = np.array(f_values)
        restart.update_distribution(candidates, z, f_values, self.lb, self.ub)

        return f_values

    def optimize(self, max_evaluations: int) -> IPOPDCMAResult:
        """
        Run IPOP-DCMA optimization.

        Strategy:
        1. Stage 1 (Probe): Create restarts, run each for probe_budget
        2. Stage 2 (Exploit): Select top K, allocate remaining budget
        3. Interleave generations across active restarts
        """
        probe_budget = self.config.probe_budget_factor * self.dim

        # Determine number of restarts based on budget
        n_restarts = min(
            self.config.max_restarts,
            max(1, max_evaluations // (2 * probe_budget))
        )

        # ========== STAGE 1: PROBE ==========
        # Create and probe all restarts
        current_lambda = self.base_lambda
        base_A = np.eye(self.dim)

        for r in range(n_restarts):
            if self.evaluations >= max_evaluations:
                break

            restart = self._create_restart(r, current_lambda, base_A)
            self.restarts.append(restart)

            # Run probe phase for this restart
            restart_probe_budget = min(probe_budget, max_evaluations - self.evaluations)

            while restart.evaluations < restart_probe_budget:
                if self.evaluations >= max_evaluations:
                    break

                remaining = restart_probe_budget - restart.evaluations
                if remaining < restart.lambda_:
                    break

                self._run_generation(restart)

                # Check early termination
                if self.best_f < self.config.tol_fun:
                    break

            self.restart_history.append({
                'restart_id': r,
                'lambda': current_lambda,
                'probe_evals': restart.evaluations,
                'probe_best_f': restart.best_f
            })

            # IPOP: increase population for next restart
            current_lambda = int(current_lambda * self.config.lambda_increase_factor)
            if current_lambda % 2 == 1:
                current_lambda += 1

            # Update base_A from best restart's covariance (adaptation)
            if restart.best_f < float('inf'):
                base_A = restart.A.copy()

            if self.best_f < self.config.tol_fun:
                break

        # ========== STAGE 2: EXPLOIT ==========
        if self.evaluations < max_evaluations and self.best_f >= self.config.tol_fun:
            # Select top K restarts by best_f
            active_restarts = [r for r in self.restarts if r.best_f < float('inf')]

            if len(active_restarts) > 0:
                # Sort by best_f (ascending)
                active_restarts.sort(key=lambda r: r.best_f)

                # Select top K
                top_k = max(1, int(len(active_restarts) * self.config.top_k_fraction))
                selected = active_restarts[:top_k]

                # Remaining budget
                remaining_budget = max_evaluations - self.evaluations

                # Allocate proportionally (inverse of best_f as weight)
                if len(selected) > 1:
                    # Use rank-based allocation (deterministic)
                    ranks = np.arange(1, len(selected) + 1)
                    weights = 1.0 / ranks
                    weights = weights / np.sum(weights)
                    budgets = (weights * remaining_budget).astype(int)
                    budgets[-1] = remaining_budget - np.sum(budgets[:-1])
                else:
                    budgets = [remaining_budget]

                # Interleaved exploitation
                restart_budgets = {r.restart_id: b for r, b in zip(selected, budgets)}
                restart_evals = {r.restart_id: 0 for r in selected}

                while self.evaluations < max_evaluations and self.best_f >= self.config.tol_fun:
                    made_progress = False

                    for restart in selected:
                        if self.evaluations >= max_evaluations:
                            break
                        if self.best_f < self.config.tol_fun:
                            break

                        # Check if this restart has budget left
                        if restart_evals[restart.restart_id] >= restart_budgets[restart.restart_id]:
                            continue

                        # Check stagnation
                        if restart.check_stagnation(self.config.tol_fun):
                            continue

                        # Run one generation
                        remaining = restart_budgets[restart.restart_id] - restart_evals[restart.restart_id]
                        if remaining < restart.lambda_:
                            continue

                        gen_evals_before = restart.evaluations
                        self._run_generation(restart)
                        gen_evals = restart.evaluations - gen_evals_before
                        restart_evals[restart.restart_id] += gen_evals
                        made_progress = True

                    if not made_progress:
                        break

        # Compute receipt hash
        receipt_data = f"{self.best_x.tolist() if self.best_x is not None else 'None'}:{self.best_f}:{self.evaluations}"
        receipt_hash = hashlib.sha256(receipt_data.encode()).hexdigest()[:16]

        return IPOPDCMAResult(
            x_best=self.best_x,
            f_best=self.best_f,
            evaluations=self.evaluations,
            trajectory=self.trajectory.copy(),
            n_restarts=len(self.restarts),
            restart_history=self.restart_history.copy(),
            receipt_hash=receipt_hash
        )

    def reset(self):
        """Reset optimizer state."""
        self.evaluations = 0
        self.best_x = None
        self.best_f = float('inf')
        self.trajectory = []
        self.restarts = []
        self.restart_history = []


class FullIPOPDCMA:
    """
    Full IPOP-DCMA with quadratic detection.

    Phase 0: Quick quadratic check
    Phase 1: If quadratic, solve exactly
    Phase 2: Otherwise, run IPOP-DCMA
    """

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        dim: int,
        bounds: List[Tuple[float, float]],
        config: IPOPDCMAConfig = None
    ):
        self.objective = objective
        self.dim = dim
        self.bounds = bounds
        self.lb = np.array([b[0] for b in bounds])
        self.ub = np.array([b[1] for b in bounds])
        self.config = config or IPOPDCMAConfig()

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

    def optimize(self, max_evaluations: int) -> IPOPDCMAResult:
        """Run full optimization pipeline."""
        from .quadratic_id import QuadraticIdentifier

        # Phase 0: Quick quadratic check (if budget allows and dim is small)
        if max_evaluations > 5 * self.dim ** 2 and self.dim <= 20:
            quad_id = QuadraticIdentifier(
                self._eval,
                self.dim,
                self.bounds,
                h=1e-4
            )

            if quad_id.quick_detect(tol=0.01):
                # Phase 1: Full quadratic identification
                result = quad_id.identify_and_solve(verify=True)

                if result.is_quadratic and result.confidence > 0.95:
                    if result.f_optimal < self.config.tol_fun:
                        return IPOPDCMAResult(
                            x_best=self.best_x,
                            f_best=self.best_f,
                            evaluations=self.evaluations,
                            trajectory=self.trajectory.copy(),
                            n_restarts=0,
                            restart_history=[{'strategy': 'quadratic'}],
                            receipt_hash=result.receipt_hash
                        )

        # Phase 2: IPOP-DCMA
        remaining = max_evaluations - self.evaluations

        ipop = IPOPDCMA(
            self._eval,
            self.dim,
            self.bounds,
            self.config
        )

        # Transfer best so far
        if self.best_x is not None:
            ipop.best_x = self.best_x.copy()
            ipop.best_f = self.best_f

        ipop_result = ipop.optimize(remaining)

        return IPOPDCMAResult(
            x_best=self.best_x,
            f_best=self.best_f,
            evaluations=self.evaluations,
            trajectory=self.trajectory.copy(),
            n_restarts=ipop_result.n_restarts,
            restart_history=ipop_result.restart_history,
            receipt_hash=ipop_result.receipt_hash
        )

    def reset(self):
        """Reset optimizer state."""
        self.evaluations = 0
        self.best_x = None
        self.best_f = float('inf')
        self.trajectory = []
        self.f_cache = {}
