"""
PhaseProbe: DFT-based Phase Extraction for Periodic Functions

Implements the Delta*-closure for shifted multimodal functions.
For functions with periodic structure (like shifted Rastrigin),
PhaseProbe can identify the latent shift parameter in O(d*M) evaluations.

Mathematical Foundation:
For f(x) = g(x - s) where g is periodic:
1. Sample f along each coordinate axis
2. Apply DFT to extract phase information
3. Recover shift s from phase angles
4. Refine candidate solution locally

This collapses O(n^n) search to O(d*M) identification.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class PhaseProbeResult:
    """Result of phase probe identification."""
    candidate_x: np.ndarray
    estimated_shift: np.ndarray
    confidence: float
    total_evals: int
    periodic_dimensions: List[int]


class PhaseProbe:
    """
    DFT-based phase extraction for periodic function identification.

    Uses the Discrete Fourier Transform to identify the latent shift
    parameter in shifted periodic functions.
    """

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        dimension: int,
        bounds: List[Tuple[float, float]],
        expected_period: float = 1.0
    ):
        """
        Initialize PhaseProbe.

        Args:
            objective: Function to evaluate
            dimension: Number of dimensions
            bounds: Variable bounds
            expected_period: Expected period of the function (e.g., 1 for Rastrigin)
        """
        self.objective = objective
        self.dimension = dimension
        self.bounds = bounds
        self.lb = np.array([b[0] for b in bounds])
        self.ub = np.array([b[1] for b in bounds])
        self.expected_period = expected_period
        self.total_evals = 0

    def detect_periodicity(
        self,
        x0: np.ndarray,
        dim: int,
        M: int = 32
    ) -> Tuple[bool, float]:
        """
        Detect if function has periodic structure along dimension dim.

        Args:
            x0: Base point
            dim: Dimension to probe
            M: Number of DFT samples

        Returns:
            (is_periodic, energy_ratio): True if periodic, with confidence
        """
        # Sample along dimension
        width = self.ub[dim] - self.lb[dim]
        samples = np.zeros(M)

        for i in range(M):
            x = x0.copy()
            x[dim] = self.lb[dim] + i * width / M
            samples[i] = self.objective(x)
            self.total_evals += 1

        # Apply DFT
        fft = np.fft.fft(samples)
        magnitudes = np.abs(fft)

        # Check for strong periodic component
        # DC component at index 0, fundamental at index 1, etc.
        dc_energy = magnitudes[0] ** 2
        ac_energy = np.sum(magnitudes[1:M//2] ** 2)

        if dc_energy + ac_energy < 1e-10:
            return False, 0.0

        # Find dominant frequency
        dominant_idx = np.argmax(magnitudes[1:M//2]) + 1
        dominant_energy = magnitudes[dominant_idx] ** 2
        total_ac_energy = ac_energy

        # Periodic if dominant frequency contains most AC energy
        energy_ratio = dominant_energy / (total_ac_energy + 1e-10)

        return energy_ratio > 0.3, energy_ratio

    def extract_phase(
        self,
        x0: np.ndarray,
        dim: int,
        M: int = 32
    ) -> Tuple[float, float]:
        """
        Extract phase (shift) along dimension dim.

        Args:
            x0: Base point
            dim: Dimension to probe
            M: Number of DFT samples

        Returns:
            (estimated_shift, confidence): Shift estimate and confidence
        """
        # Sample along dimension
        width = self.ub[dim] - self.lb[dim]
        samples = np.zeros(M)

        for i in range(M):
            x = x0.copy()
            x[dim] = self.lb[dim] + i * width / M
            samples[i] = self.objective(x)
            self.total_evals += 1

        # Apply DFT
        fft = np.fft.fft(samples)
        magnitudes = np.abs(fft)

        # Find dominant frequency
        dominant_idx = np.argmax(magnitudes[1:M//2]) + 1

        # Extract phase from dominant component
        phase = np.angle(fft[dominant_idx])

        # Convert phase to shift
        # Phase in [-pi, pi], frequency = dominant_idx / M cycles per width
        # shift = -phase / (2 * pi * frequency) * period
        frequency = dominant_idx / M * (width / self.expected_period)
        if frequency > 0:
            shift_fraction = -phase / (2 * np.pi * dominant_idx)
            estimated_shift = self.lb[dim] + shift_fraction * width
            # Wrap to bounds
            estimated_shift = self.lb[dim] + (estimated_shift - self.lb[dim]) % self.expected_period
        else:
            estimated_shift = (self.lb[dim] + self.ub[dim]) / 2

        confidence = magnitudes[dominant_idx] / (np.sum(magnitudes[1:M//2]) + 1e-10)

        return estimated_shift, confidence

    def identify_and_refine(self, M: int = 32) -> PhaseProbeResult:
        """
        Full phase identification and refinement.

        1. Detect periodicity in each dimension
        2. Extract phases for periodic dimensions
        3. Construct candidate solution
        4. Return result

        Args:
            M: Number of DFT samples per dimension

        Returns:
            PhaseProbeResult with candidate and metadata
        """
        center = (self.lb + self.ub) / 2
        estimated_shift = np.zeros(self.dimension)
        periodic_dims = []
        total_confidence = 0.0

        # Probe each dimension
        for d in range(self.dimension):
            is_periodic, energy_ratio = self.detect_periodicity(center, d, M)

            if is_periodic:
                shift, conf = self.extract_phase(center, d, M)
                estimated_shift[d] = shift
                periodic_dims.append(d)
                total_confidence += conf
            else:
                # Non-periodic: use center
                estimated_shift[d] = center[d]

        # Confidence is average over periodic dimensions
        avg_confidence = total_confidence / max(1, len(periodic_dims))

        # Candidate is the estimated shift
        candidate = np.clip(estimated_shift, self.lb, self.ub)

        return PhaseProbeResult(
            candidate_x=candidate,
            estimated_shift=estimated_shift,
            confidence=avg_confidence,
            total_evals=self.total_evals,
            periodic_dimensions=periodic_dims
        )

    def reset(self):
        """Reset evaluation counter."""
        self.total_evals = 0
