"""
Replay and Verify BBOB Inversion Results

This module provides verification of inversion benchmark runs:
1. Re-runs the generator to compute x_opt
2. Verifies the logged evaluation matches
3. Checks the hash chain integrity

Usage:
    python -m opoch_optimizer.coco.inversion.replay_verify results/opoch_inversion/
"""

import os
import json
import hashlib
import numpy as np
from typing import Dict, List, Tuple, Any

from .bbob_generator import BBOBGenerator
from .bbob_inverter import BBOBInverter


def verify_receipt(
    receipt: Dict[str, Any],
    generator: BBOBGenerator,
    tol: float = 1e-10
) -> Tuple[bool, str]:
    """
    Verify a single receipt from the inversion benchmark.

    Checks:
    1. Generator state hash matches recomputed hash
    2. x_opt matches recomputed x_opt
    3. f_at_x_opt matches when we re-evaluate

    Args:
        receipt: Single receipt dict from receipt_chain.json
        generator: BBOBGenerator instance
        tol: Tolerance for numerical comparisons

    Returns:
        (passed, message) tuple
    """
    import ioh

    fid = receipt['function_id']
    iid = receipt['instance_id']
    dim = receipt['dimension']

    # Recompute generator state
    state = generator.get_state(fid, iid, dim)

    # Check 1: Generator hash matches
    if state.state_hash != receipt['generator_hash']:
        return False, f"Generator hash mismatch: {state.state_hash} != {receipt['generator_hash']}"

    # Check 2: x_opt matches
    logged_x_opt = np.array(receipt['x_opt'])
    if not np.allclose(state.x_opt, logged_x_opt, rtol=1e-12, atol=1e-12):
        return False, f"x_opt mismatch: {state.x_opt} != {logged_x_opt}"

    # Check 3: Re-evaluate and verify f_at_x_opt
    problem = ioh.get_problem(fid, iid, dim, ioh.ProblemClass.BBOB)
    f_recomputed = problem(state.x_opt)

    if abs(f_recomputed - receipt['f_at_x_opt']) > tol:
        return False, f"f_at_x_opt mismatch: {f_recomputed} != {receipt['f_at_x_opt']}"

    # Check 4: Verify f_opt matches
    if abs(state.f_opt - receipt['f_opt']) > tol:
        return False, f"f_opt mismatch: {state.f_opt} != {receipt['f_opt']}"

    return True, "Verified"


def verify_chain_integrity(receipt_chain: List[Dict], chain_hash: str) -> Tuple[bool, str]:
    """
    Verify the integrity of the receipt chain.

    The chain hash should match SHA256(json.dumps(receipt_chain, sort_keys=True)).

    Args:
        receipt_chain: List of receipt dicts
        chain_hash: The stored chain hash

    Returns:
        (passed, message) tuple
    """
    chain_data = json.dumps(receipt_chain, sort_keys=True)
    recomputed_hash = hashlib.sha256(chain_data.encode()).hexdigest()

    if recomputed_hash != chain_hash:
        return False, f"Chain hash mismatch: {recomputed_hash} != {chain_hash}"

    return True, "Chain integrity verified"


def replay_and_verify(
    results_dir: str,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Replay and verify an entire inversion benchmark run.

    Args:
        results_dir: Directory containing summary.json and receipt_chain.json
        verbose: Print progress

    Returns:
        Verification summary dict
    """
    # Load results
    summary_path = os.path.join(results_dir, 'summary.json')
    chain_path = os.path.join(results_dir, 'receipt_chain.json')

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary not found: {summary_path}")
    if not os.path.exists(chain_path):
        raise FileNotFoundError(f"Receipt chain not found: {chain_path}")

    with open(summary_path, 'r') as f:
        summary = json.load(f)

    with open(chain_path, 'r') as f:
        receipt_chain = json.load(f)

    if verbose:
        print("\n" + "=" * 70)
        print("REPLAY VERIFICATION")
        print("=" * 70)
        print(f"Results directory: {results_dir}")
        print(f"Total receipts: {len(receipt_chain)}")
        print(f"Original chain hash: {summary['chain_hash'][:16]}...")
        print("=" * 70 + "\n")

    # Initialize generator
    generator = BBOBGenerator()

    # Verify chain integrity first
    chain_ok, chain_msg = verify_chain_integrity(
        receipt_chain,
        summary['chain_hash']
    )

    if verbose:
        status = "PASS" if chain_ok else "FAIL"
        print(f"Chain integrity: [{status}] {chain_msg}")

    # Verify each receipt
    passed = 0
    failed = 0
    failures = []

    for i, receipt in enumerate(receipt_chain):
        ok, msg = verify_receipt(receipt, generator)

        if ok:
            passed += 1
        else:
            failed += 1
            failures.append({
                'index': i,
                'receipt': receipt,
                'message': msg
            })

        if verbose and (i + 1) % 100 == 0:
            print(f"  Verified {i + 1}/{len(receipt_chain)} receipts...")

    # Summary
    verification_result = {
        'results_dir': results_dir,
        'total_receipts': len(receipt_chain),
        'passed': passed,
        'failed': failed,
        'chain_integrity': chain_ok,
        'all_verified': chain_ok and failed == 0,
        'failures': failures
    }

    if verbose:
        print("\n" + "=" * 70)
        print("VERIFICATION SUMMARY")
        print("=" * 70)
        print(f"Receipts verified: {passed}/{len(receipt_chain)}")
        print(f"Chain integrity: {'PASS' if chain_ok else 'FAIL'}")

        if verification_result['all_verified']:
            print("\n*** ALL VERIFICATIONS PASSED ***")
            print("The inversion benchmark results are valid and reproducible.")
        else:
            print(f"\n*** {failed} VERIFICATION FAILURES ***")
            for f in failures[:5]:
                print(f"  - Receipt {f['index']}: {f['message']}")
            if len(failures) > 5:
                print(f"  ... and {len(failures) - 5} more")

        print("=" * 70)

    return verification_result


def main():
    """Main entry point for verification."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify COCO/BBOB inversion benchmark results"
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Directory containing inversion results"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    result = replay_and_verify(args.results_dir, verbose=not args.quiet)

    # Exit with error code if verification failed
    if not result['all_verified']:
        exit(1)


if __name__ == "__main__":
    main()
