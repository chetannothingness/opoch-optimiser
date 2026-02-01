"""
Sobol Sequence Generator

Deterministic low-discrepancy quasi-random sequence for global sampling.
Provides uniform coverage of the search space with explicit indexing.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SobolPoint:
    """A point in the Sobol sequence."""
    x: np.ndarray
    index: int
    region_hash: Optional[str] = None


class SobolGenerator:
    """
    Sobol sequence generator for deterministic sampling.

    Uses scipy's Sobol engine if available, falls back to custom implementation.
    All sequences are reproducible given the same seed/index.
    """

    def __init__(self, dimension: int, bounds: List[Tuple[float, float]], seed: int = 42):
        """
        Initialize Sobol generator.

        Args:
            dimension: Number of dimensions
            bounds: List of (lower, upper) bounds for each dimension
            seed: Random seed for scrambling
        """
        self.dimension = dimension
        self.bounds = bounds
        self.lb = np.array([b[0] for b in bounds])
        self.ub = np.array([b[1] for b in bounds])
        self.seed = seed
        self._index = 0
        self._engine = None

        # Try to use scipy's Sobol engine
        try:
            from scipy.stats import qmc
            self._engine = qmc.Sobol(d=dimension, scramble=True, seed=seed)
        except ImportError:
            pass

    def generate(self, n_points: int) -> List[SobolPoint]:
        """
        Generate n_points from the Sobol sequence.

        Args:
            n_points: Number of points to generate

        Returns:
            List of SobolPoint objects
        """
        if self._engine is not None:
            # Use scipy Sobol
            samples = self._engine.random(n_points)
            points = []
            for i, s in enumerate(samples):
                x = self.lb + s * (self.ub - self.lb)
                points.append(SobolPoint(x=x, index=self._index + i))
            self._index += n_points
            return points
        else:
            # Fallback: use deterministic Halton-like sequence
            return self._fallback_generate(n_points)

    def generate_in_region(
        self,
        n_points: int,
        region_lb: np.ndarray,
        region_ub: np.ndarray,
        region_hash: str
    ) -> List[SobolPoint]:
        """
        Generate Sobol points within a specific region.

        Args:
            n_points: Number of points
            region_lb: Lower bounds of region
            region_ub: Upper bounds of region
            region_hash: Hash for deterministic offset

        Returns:
            List of SobolPoint objects in the region
        """
        # Compute offset from hash for determinism
        offset = int(region_hash[:8], 16) % 1000000

        if self._engine is not None:
            try:
                from scipy.stats import qmc
                # Create new engine with offset seed
                engine = qmc.Sobol(d=self.dimension, scramble=True, seed=self.seed + offset)
                # Skip some points for variety
                _ = engine.random(offset % 100)
                samples = engine.random(n_points)
            except:
                samples = self._fallback_samples(n_points)
        else:
            samples = self._fallback_samples(n_points)

        points = []
        for i, s in enumerate(samples):
            x = region_lb + s * (region_ub - region_lb)
            points.append(SobolPoint(x=x, index=offset + i, region_hash=region_hash))

        return points

    def _fallback_generate(self, n_points: int) -> List[SobolPoint]:
        """Fallback generator using Halton-like sequence."""
        samples = self._fallback_samples(n_points)
        points = []
        for i, s in enumerate(samples):
            x = self.lb + s * (self.ub - self.lb)
            points.append(SobolPoint(x=x, index=self._index + i))
        self._index += n_points
        return points

    def _fallback_samples(self, n_points: int) -> np.ndarray:
        """Generate Halton-like samples in [0,1]^d."""
        # Use prime bases for each dimension
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

        samples = np.zeros((n_points, self.dimension))
        for j in range(self.dimension):
            base = primes[j % len(primes)]
            for i in range(n_points):
                n = self._index + i + 1
                f = 1.0 / base
                r = 0.0
                while n > 0:
                    r += f * (n % base)
                    n //= base
                    f /= base
                samples[i, j] = r

        return samples

    def reset(self):
        """Reset the generator to the beginning."""
        self._index = 0
        if self._engine is not None:
            try:
                from scipy.stats import qmc
                self._engine = qmc.Sobol(d=self.dimension, scramble=True, seed=self.seed)
            except:
                pass
