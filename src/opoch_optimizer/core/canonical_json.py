"""
Canonical JSON Serialization

Provides deterministic JSON serialization with sorted keys and
SHA-256 hashing for certificates and replay verification.
"""

import json
import hashlib
from typing import Any


def canonical_dumps(obj: Any, indent: int = None) -> str:
    """
    Canonical JSON serialization with sorted keys.

    This ensures identical objects produce identical JSON strings.

    Args:
        obj: Object to serialize
        indent: Indentation level (None for compact)

    Returns:
        Canonical JSON string
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(',', ':') if indent is None else None,
        indent=indent,
        default=str
    )


def canonical_hash(obj: Any) -> str:
    """
    Compute SHA-256 hash of canonical JSON.

    Args:
        obj: Object to hash

    Returns:
        Hex digest of SHA-256 hash
    """
    return hashlib.sha256(canonical_dumps(obj).encode()).hexdigest()
