"""
Receipt Chain for Auditable Optimization

Implements canonical JSON + SHA-256 chain for replay verification.
Every solver event is recorded and hashed into a tamper-evident chain.
"""

import json
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
from pathlib import Path


def canonical_dumps(obj: Any, indent: int = None) -> str:
    """
    Canonical JSON serialization with sorted keys.

    This ensures identical objects produce identical JSON strings.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(',', ':') if indent is None else None,
        indent=indent,
        default=str
    )


def canonical_hash(obj: Any) -> str:
    """Compute SHA-256 hash of canonical JSON."""
    return hashlib.sha256(canonical_dumps(obj).encode()).hexdigest()


class ActionType(Enum):
    """Types of solver actions that generate receipts."""
    INIT = "init"
    REGION_POP = "region_pop"
    REGION_SPLIT = "region_split"
    REGION_PRUNE = "region_prune"
    REGION_EMPTY = "region_empty"
    BOUND_COMPUTE = "bound_compute"
    FEASIBLE_FOUND = "feasible_found"
    INCUMBENT_UPDATE = "incumbent_update"
    RELAXATION_SOLVE = "relaxation_solve"
    FBBT_APPLY = "fbbt_apply"
    NEWTON_APPLY = "newton_apply"
    TERMINATE = "terminate"


@dataclass
class Receipt:
    """
    A single receipt in the chain.

    Each receipt contains:
    - Sequence number
    - Action type and parameters
    - Input/output hashes for verification
    - Link to previous receipt
    - Self-hash for chain integrity
    """
    sequence: int
    action: ActionType
    params: Dict[str, Any]
    input_hash: str
    output_hash: str
    prev_hash: str
    receipt_hash: str = ""

    def __post_init__(self):
        if not self.receipt_hash:
            self.receipt_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute the hash of this receipt."""
        data = {
            "sequence": self.sequence,
            "action": self.action.value,
            "params": self.params,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "prev_hash": self.prev_hash
        }
        return canonical_hash(data)

    def verify(self) -> bool:
        """Verify the receipt hash is correct."""
        return self.receipt_hash == self._compute_hash()

    def to_canonical(self) -> Dict[str, Any]:
        """Convert to canonical dictionary form."""
        return {
            "sequence": self.sequence,
            "action": self.action.value,
            "params": self.params,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "prev_hash": self.prev_hash,
            "receipt_hash": self.receipt_hash
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Receipt':
        """Reconstruct from dictionary."""
        return cls(
            sequence=data["sequence"],
            action=ActionType(data["action"]),
            params=data["params"],
            input_hash=data["input_hash"],
            output_hash=data["output_hash"],
            prev_hash=data["prev_hash"],
            receipt_hash=data["receipt_hash"]
        )


class ReceiptChain:
    """
    A chain of receipts forming an audit trail.

    The chain is a Merkle-like structure where each receipt's hash
    depends on the previous receipt, ensuring tamper detection.
    """

    def __init__(self):
        self.receipts: List[Receipt] = []
        self._prev_hash: str = "genesis"

    def add_receipt(
        self,
        action: ActionType,
        params: Dict[str, Any],
        input_hash: str,
        output_hash: str
    ) -> Receipt:
        """
        Add a new receipt to the chain.

        Args:
            action: The action type
            params: Action parameters
            input_hash: Hash of input data
            output_hash: Hash of output data

        Returns:
            The created receipt
        """
        receipt = Receipt(
            sequence=len(self.receipts),
            action=action,
            params=params,
            input_hash=input_hash,
            output_hash=output_hash,
            prev_hash=self._prev_hash
        )

        self.receipts.append(receipt)
        self._prev_hash = receipt.receipt_hash

        return receipt

    def verify_chain(self) -> bool:
        """
        Verify the entire chain is valid.

        Checks:
        1. Each receipt's hash is correct
        2. Chain linking is correct (prev_hash matches)

        Returns:
            True if chain is valid
        """
        if not self.receipts:
            return True

        prev_hash = "genesis"
        for receipt in self.receipts:
            # Verify receipt hash
            if not receipt.verify():
                return False

            # Verify chain linking
            if receipt.prev_hash != prev_hash:
                return False

            prev_hash = receipt.receipt_hash

        return True

    @property
    def final_hash(self) -> str:
        """Get the hash of the last receipt."""
        if not self.receipts:
            return "genesis"
        return self.receipts[-1].receipt_hash

    def to_canonical(self) -> Dict[str, Any]:
        """Convert chain to canonical form."""
        return {
            "receipts": [r.to_canonical() for r in self.receipts],
            "final_hash": self.final_hash
        }

    def save_json(self, path: Path) -> None:
        """Save chain to JSON file."""
        with open(path, 'w') as f:
            f.write(canonical_dumps(self.to_canonical(), indent=2))

    @classmethod
    def load_json(cls, path: Path) -> 'ReceiptChain':
        """Load chain from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        chain = cls()
        for r_data in data["receipts"]:
            receipt = Receipt.from_dict(r_data)
            chain.receipts.append(receipt)
            chain._prev_hash = receipt.receipt_hash

        return chain
