"""
Tie-Safe Deterministic Ordering

Implements deterministic tie-breaking via canonical fingerprints.
This ensures identical execution across runs.
"""

import hashlib
from typing import Any, Dict, List, TypeVar, Callable
from dataclasses import dataclass

from .receipts import canonical_dumps


T = TypeVar('T')


def canonical_fingerprint(obj: Any) -> str:
    """
    Compute a deterministic fingerprint for any object.

    Uses canonical JSON serialization followed by SHA-256 hash.
    """
    return hashlib.sha256(canonical_dumps(obj).encode()).hexdigest()[:16]


@dataclass
class TieSafeChoice:
    """
    Utility for deterministic selection when scores are equal.

    When multiple candidates have the same score, we break ties using:
    1. Canonical fingerprint of the candidate
    2. Lexicographic ordering

    This ensures identical choices across runs.
    """

    @staticmethod
    def select_best(
        candidates: List[T],
        score_fn: Callable[[T], float],
        fingerprint_fn: Callable[[T], str] = None,
        minimize: bool = True
    ) -> T:
        """
        Select the best candidate with deterministic tie-breaking.

        Args:
            candidates: List of candidates
            score_fn: Function to compute score
            fingerprint_fn: Function to compute fingerprint (default: canonical)
            minimize: If True, prefer lower scores

        Returns:
            The selected candidate
        """
        if not candidates:
            raise ValueError("No candidates to select from")

        if fingerprint_fn is None:
            fingerprint_fn = lambda x: canonical_fingerprint(
                x.to_canonical() if hasattr(x, 'to_canonical') else str(x)
            )

        # Compute scores and fingerprints
        scored = [
            (score_fn(c), fingerprint_fn(c), i, c)
            for i, c in enumerate(candidates)
        ]

        # Sort: by score, then fingerprint, then original index
        scored.sort(key=lambda x: (x[0] if minimize else -x[0], x[1], x[2]))

        return scored[0][3]

    @staticmethod
    def select_top_k(
        candidates: List[T],
        score_fn: Callable[[T], float],
        k: int,
        fingerprint_fn: Callable[[T], str] = None,
        minimize: bool = True
    ) -> List[T]:
        """
        Select the top k candidates with deterministic tie-breaking.

        Args:
            candidates: List of candidates
            score_fn: Function to compute score
            k: Number to select
            fingerprint_fn: Function to compute fingerprint
            minimize: If True, prefer lower scores

        Returns:
            List of k selected candidates
        """
        if fingerprint_fn is None:
            fingerprint_fn = lambda x: canonical_fingerprint(
                x.to_canonical() if hasattr(x, 'to_canonical') else str(x)
            )

        # Compute scores and fingerprints
        scored = [
            (score_fn(c), fingerprint_fn(c), i, c)
            for i, c in enumerate(candidates)
        ]

        # Sort: by score, then fingerprint, then original index
        scored.sort(key=lambda x: (x[0] if minimize else -x[0], x[1], x[2]))

        return [s[3] for s in scored[:k]]


def deterministic_argmin(
    values: List[float],
    fingerprints: List[str] = None
) -> int:
    """
    Find the index of the minimum value with deterministic tie-breaking.

    Args:
        values: List of values
        fingerprints: Optional list of fingerprints for tie-breaking

    Returns:
        Index of the minimum
    """
    if not values:
        raise ValueError("No values to search")

    if fingerprints is None:
        fingerprints = [str(i) for i in range(len(values))]

    min_val = min(values)
    candidates = [
        (fingerprints[i], i)
        for i, v in enumerate(values)
        if v == min_val
    ]

    # Sort by fingerprint, then index
    candidates.sort()

    return candidates[0][1]


def deterministic_argmax(
    values: List[float],
    fingerprints: List[str] = None
) -> int:
    """
    Find the index of the maximum value with deterministic tie-breaking.

    Args:
        values: List of values
        fingerprints: Optional list of fingerprints for tie-breaking

    Returns:
        Index of the maximum
    """
    # Convert to minimization problem
    neg_values = [-v for v in values]
    return deterministic_argmin(neg_values, fingerprints)
