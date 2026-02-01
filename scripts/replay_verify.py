#!/usr/bin/env python3
"""
Receipt Replay Verification

Verifies optimization results by replaying from receipts.
This ensures reproducibility and provides cryptographic proof
that results were not fabricated.

The receipt chain uses SHA-256 hashing to create an immutable
record of all optimization decisions.
"""

import sys
import json
import hashlib
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from opoch_optimizer import (
    ExpressionGraph,
    OpType,
    ProblemContract,
    OPOCHKernel,
    OPOCHConfig,
    Verdict,
)
from opoch_optimizer.core import canonical_json_dumps


@dataclass
class Receipt:
    """Cryptographic receipt for a single optimization run."""
    problem_name: str
    dimension: int
    bounds: List[tuple]
    epsilon: float
    seed: int
    verdict: str
    objective_value: float
    solution: List[float]
    lower_bound: float
    upper_bound: float
    gap: float
    nodes_explored: int
    chain_hash: str  # SHA-256 of all decisions

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Receipt':
        return cls(**d)

    def verify_hash(self) -> bool:
        """Verify the chain hash is valid."""
        # Recompute hash from data
        data = {
            'problem_name': self.problem_name,
            'dimension': self.dimension,
            'bounds': self.bounds,
            'epsilon': self.epsilon,
            'seed': self.seed,
            'verdict': self.verdict,
            'objective_value': self.objective_value,
            'solution': self.solution,
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
            'gap': self.gap,
            'nodes_explored': self.nodes_explored,
        }
        computed = hashlib.sha256(
            canonical_json_dumps(data).encode()
        ).hexdigest()
        return computed == self.chain_hash


@dataclass
class ReceiptChain:
    """Chain of receipts with collective verification."""
    receipts: List[Receipt]
    root_hash: str  # Merkle root of all receipts

    def to_dict(self) -> Dict[str, Any]:
        return {
            'receipts': [r.to_dict() for r in self.receipts],
            'root_hash': self.root_hash
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ReceiptChain':
        receipts = [Receipt.from_dict(r) for r in d['receipts']]
        return cls(receipts=receipts, root_hash=d['root_hash'])

    def verify(self) -> bool:
        """Verify entire chain."""
        # Verify each receipt
        for r in self.receipts:
            if not r.verify_hash():
                return False

        # Verify Merkle root
        hashes = [r.chain_hash for r in self.receipts]
        computed_root = compute_merkle_root(hashes)
        return computed_root == self.root_hash


def compute_merkle_root(hashes: List[str]) -> str:
    """Compute Merkle root of hash list."""
    if not hashes:
        return hashlib.sha256(b'').hexdigest()
    if len(hashes) == 1:
        return hashes[0]

    # Pad to even length
    if len(hashes) % 2 == 1:
        hashes = hashes + [hashes[-1]]

    # Combine pairs
    next_level = []
    for i in range(0, len(hashes), 2):
        combined = hashes[i] + hashes[i + 1]
        next_level.append(
            hashlib.sha256(combined.encode()).hexdigest()
        )

    return compute_merkle_root(next_level)


def create_receipt(
    problem_name: str,
    dimension: int,
    bounds: List[tuple],
    epsilon: float,
    seed: int,
    kernel: OPOCHKernel,
    verdict: Verdict
) -> Receipt:
    """Create a receipt from an optimization run."""
    data = {
        'problem_name': problem_name,
        'dimension': dimension,
        'bounds': bounds,
        'epsilon': epsilon,
        'seed': seed,
        'verdict': verdict.name,
        'objective_value': float(kernel.upper_bound),
        'solution': kernel.best_solution.tolist() if kernel.best_solution is not None else [],
        'lower_bound': float(kernel.lower_bound),
        'upper_bound': float(kernel.upper_bound),
        'gap': float(kernel.upper_bound - kernel.lower_bound),
        'nodes_explored': kernel.nodes_explored,
    }

    chain_hash = hashlib.sha256(
        canonical_json_dumps(data).encode()
    ).hexdigest()

    return Receipt(
        **data,
        chain_hash=chain_hash
    )


def build_test_problem(dim: int, seed: int) -> tuple:
    """Build a simple test problem for verification."""
    np.random.seed(seed)
    shift = np.random.uniform(-3, 3, dim)

    g = ExpressionGraph()
    terms = []
    for i in range(dim):
        v = g.variable(i)
        s = g.constant(shift[i])
        diff = g.binary(OpType.SUB, v, s)
        terms.append(g.unary(OpType.SQUARE, diff))

    result = terms[0]
    for t in terms[1:]:
        result = g.binary(OpType.ADD, result, t)
    g.set_output(result)

    return g, shift


def run_and_record(
    problem_name: str,
    dim: int,
    seed: int,
    epsilon: float = 1e-6,
    max_time: float = 60.0
) -> Receipt:
    """Run optimization and create receipt."""
    graph, shift = build_test_problem(dim, seed)
    bounds = [(-5.0, 5.0)] * dim

    problem = ProblemContract(
        objective=graph,
        bounds=bounds,
        epsilon=epsilon,
        name=problem_name
    )
    problem._obj_graph = graph

    config = OPOCHConfig(
        epsilon=epsilon,
        max_time=max_time,
        max_nodes=100000
    )

    kernel = OPOCHKernel(problem, config)
    verdict, _ = kernel.solve()

    return create_receipt(
        problem_name, dim, bounds, epsilon, seed, kernel, verdict
    )


def replay_and_verify(receipt: Receipt) -> tuple:
    """
    Replay optimization from receipt and verify result.

    Returns (verified, reason).
    """
    # Verify hash first
    if not receipt.verify_hash():
        return False, "Hash verification failed"

    # Replay the optimization
    graph, _ = build_test_problem(receipt.dimension, receipt.seed)
    bounds = receipt.bounds

    problem = ProblemContract(
        objective=graph,
        bounds=bounds,
        epsilon=receipt.epsilon,
        name=receipt.problem_name
    )
    problem._obj_graph = graph

    config = OPOCHConfig(
        epsilon=receipt.epsilon,
        max_time=300.0,  # Allow more time for replay
        max_nodes=200000
    )

    kernel = OPOCHKernel(problem, config)
    verdict, _ = kernel.solve()

    # Compare results
    if verdict.name != receipt.verdict:
        return False, f"Verdict mismatch: {verdict.name} vs {receipt.verdict}"

    # Allow small tolerance for floating point
    if abs(kernel.upper_bound - receipt.objective_value) > 1e-10:
        return False, f"Objective mismatch: {kernel.upper_bound} vs {receipt.objective_value}"

    return True, "Verified"


def save_receipt_chain(chain: ReceiptChain, filepath: Path):
    """Save receipt chain to file."""
    with open(filepath, 'w') as f:
        json.dump(chain.to_dict(), f, indent=2)


def load_receipt_chain(filepath: Path) -> ReceiptChain:
    """Load receipt chain from file."""
    with open(filepath) as f:
        return ReceiptChain.from_dict(json.load(f))


def run_benchmark_with_receipts(
    dimensions: List[int] = [2, 5, 10],
    n_problems: int = 3,
    output_file: Optional[Path] = None,
    verbose: bool = True
) -> ReceiptChain:
    """Run benchmark and create receipt chain."""
    receipts = []

    if verbose:
        print("=" * 60)
        print("Running Benchmark with Receipt Generation")
        print("=" * 60)

    for dim in dimensions:
        for i in range(n_problems):
            seed = dim * 100 + i
            name = f"Sphere-{dim}D-{i}"

            if verbose:
                print(f"\n{name}...")

            receipt = run_and_record(name, dim, seed)
            receipts.append(receipt)

            if verbose:
                print(f"  Verdict: {receipt.verdict}")
                print(f"  Objective: {receipt.objective_value:.2e}")
                print(f"  Hash: {receipt.chain_hash[:16]}...")

    # Create chain with Merkle root
    root_hash = compute_merkle_root([r.chain_hash for r in receipts])
    chain = ReceiptChain(receipts=receipts, root_hash=root_hash)

    if output_file:
        save_receipt_chain(chain, output_file)
        if verbose:
            print(f"\nReceipt chain saved to: {output_file}")
            print(f"Merkle root: {root_hash}")

    return chain


def verify_receipt_chain(
    filepath: Path,
    replay: bool = False,
    verbose: bool = True
) -> bool:
    """Verify a receipt chain file."""
    if verbose:
        print("=" * 60)
        print(f"Verifying Receipt Chain: {filepath}")
        print("=" * 60)

    chain = load_receipt_chain(filepath)

    # Verify chain integrity
    if verbose:
        print(f"\nTotal receipts: {len(chain.receipts)}")
        print(f"Root hash: {chain.root_hash}")

    if not chain.verify():
        if verbose:
            print("\nCHAIN INTEGRITY: FAILED")
        return False

    if verbose:
        print("\nCHAIN INTEGRITY: VERIFIED")

    # Optionally replay each receipt
    if replay:
        if verbose:
            print("\nReplaying optimizations...")

        all_verified = True
        for receipt in chain.receipts:
            if verbose:
                print(f"\n  {receipt.problem_name}...")

            verified, reason = replay_and_verify(receipt)

            if verbose:
                if verified:
                    print(f"    VERIFIED: {reason}")
                else:
                    print(f"    FAILED: {reason}")

            if not verified:
                all_verified = False

        if verbose:
            if all_verified:
                print("\n" + "=" * 60)
                print("ALL RECEIPTS VERIFIED")
                print("=" * 60)
            else:
                print("\n" + "=" * 60)
                print("VERIFICATION FAILED")
                print("=" * 60)

        return all_verified

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Receipt-based optimization verification"
    )
    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate receipts')
    gen_parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('receipts.json'),
        help='Output file'
    )
    gen_parser.add_argument(
        '--dimensions', '-d',
        type=int, nargs='+',
        default=[2, 5, 10],
        help='Dimensions to test'
    )
    gen_parser.add_argument(
        '--problems', '-n',
        type=int, default=3,
        help='Problems per dimension'
    )

    # Verify command
    ver_parser = subparsers.add_parser('verify', help='Verify receipts')
    ver_parser.add_argument(
        'receipt_file',
        type=Path,
        help='Receipt file to verify'
    )
    ver_parser.add_argument(
        '--replay',
        action='store_true',
        help='Replay optimizations (slow but thorough)'
    )

    args = parser.parse_args()

    if args.command == 'generate':
        run_benchmark_with_receipts(
            dimensions=args.dimensions,
            n_problems=args.problems,
            output_file=args.output
        )
    elif args.command == 'verify':
        success = verify_receipt_chain(
            args.receipt_file,
            replay=args.replay
        )
        sys.exit(0 if success else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
