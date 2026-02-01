"""
GLOBALLib Benchmark Runner

Runs all GLOBALLib problems with certified mathematical verification.

The contract:
- UNIQUE-OPT: Certified via gap closure (UB - LB ≤ ε)
- UNSAT: Certified via refutation cover
- OMEGA-GAP: Budget exhausted before certification

No reference to external optimal values. Pure mathematics.
"""

import time
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np

from opoch_optimizer.solver.opoch_kernel import OPOCHKernel, OPOCHConfig
from opoch_optimizer.core.output_gate import Verdict

from .globallib_problems import (
    GLOBALLibProblem,
    get_all_problems,
    get_problem,
)


class CertificationStatus(Enum):
    """Certification status for a problem."""
    UNIQUE_OPT = "UNIQUE-OPT"  # Certified optimal via gap closure
    UNSAT = "UNSAT"            # Certified infeasible via refutation
    OMEGA_GAP = "Ω-GAP"        # Budget exhausted
    ERROR = "ERROR"            # Solver error


@dataclass
class GLOBALLibResult:
    """Result for a single GLOBALLib problem."""
    problem_name: str
    status: CertificationStatus
    upper_bound: float
    lower_bound: float
    gap: float
    epsilon: float
    gap_closed: bool
    solution: Optional[np.ndarray]
    objective_value: Optional[float]
    nodes_explored: int
    time_seconds: float
    certificate_hash: str
    error_message: Optional[str] = None


def run_single_problem(
    problem: GLOBALLibProblem,
    epsilon: float = 1e-4,
    max_time: float = 120.0,
    max_nodes: int = 50000,
    verbose: bool = True
) -> GLOBALLibResult:
    """
    Run a single GLOBALLib problem with certified verification.

    Args:
        problem: The problem to solve
        epsilon: Optimality gap tolerance
        max_time: Maximum time in seconds
        max_nodes: Maximum nodes to explore
        verbose: Print progress

    Returns:
        GLOBALLibResult with certification status
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Problem: {problem.name}")
        print(f"Description: {problem.description}")
        print(f"n_vars: {problem.n_vars}")
        print(f"Constraints: {len(problem.eq_constraints)} eq, {len(problem.ineq_constraints)} ineq")
        print(f"{'='*60}")

    try:
        # Convert to problem contract
        contract = problem.to_problem_contract()

        # Configure solver
        config = OPOCHConfig(
            epsilon=epsilon,
            max_time=max_time,
            max_nodes=max_nodes,
            log_frequency=1000 if verbose else 10000
        )

        # Solve
        start_time = time.time()
        kernel = OPOCHKernel(contract, config)
        verdict, result = kernel.solve()
        elapsed = time.time() - start_time

        # Extract results
        ub = result.upper_bound if hasattr(result, 'upper_bound') else float('inf')
        lb = result.lower_bound if hasattr(result, 'lower_bound') else float('-inf')
        gap = ub - lb
        gap_closed = gap <= epsilon

        # Solution
        solution = None
        obj_val = None
        if hasattr(result, 'x_optimal') and result.x_optimal is not None:
            solution = result.x_optimal
            obj_val = ub
        elif hasattr(result, 'x_best') and result.x_best is not None:
            solution = result.x_best
            obj_val = ub

        # Determine certification status
        if verdict == Verdict.UNIQUE_OPT:
            status = CertificationStatus.UNIQUE_OPT
        elif verdict == Verdict.UNSAT:
            status = CertificationStatus.UNSAT
        else:
            status = CertificationStatus.OMEGA_GAP

        # Compute certificate hash
        cert_data = f"{problem.name}:{status.value}:{ub}:{lb}:{gap}"
        cert_hash = hashlib.sha256(cert_data.encode()).hexdigest()[:16]

        result_obj = GLOBALLibResult(
            problem_name=problem.name,
            status=status,
            upper_bound=ub,
            lower_bound=lb,
            gap=gap,
            epsilon=epsilon,
            gap_closed=gap_closed,
            solution=solution,
            objective_value=obj_val,
            nodes_explored=result.nodes_explored if hasattr(result, 'nodes_explored') else 0,
            time_seconds=elapsed,
            certificate_hash=cert_hash
        )

        if verbose:
            print(f"\nResult: {status.value}")
            print(f"Upper Bound: {ub:.6g}")
            print(f"Lower Bound: {lb:.6g}")
            print(f"Gap: {gap:.6g} (ε = {epsilon})")
            print(f"Gap Closed: {gap_closed}")
            print(f"Nodes: {result_obj.nodes_explored}")
            print(f"Time: {elapsed:.2f}s")
            print(f"Certificate: {cert_hash}")

        return result_obj

    except Exception as e:
        if verbose:
            print(f"ERROR: {e}")

        return GLOBALLibResult(
            problem_name=problem.name,
            status=CertificationStatus.ERROR,
            upper_bound=float('inf'),
            lower_bound=float('-inf'),
            gap=float('inf'),
            epsilon=epsilon,
            gap_closed=False,
            solution=None,
            objective_value=None,
            nodes_explored=0,
            time_seconds=0.0,
            certificate_hash="error",
            error_message=str(e)
        )


def run_globallib_benchmark(
    problem_names: Optional[List[str]] = None,
    epsilon: float = 1e-4,
    max_time: float = 120.0,
    max_nodes: int = 50000,
    verbose: bool = True
) -> Dict[str, GLOBALLibResult]:
    """
    Run GLOBALLib benchmark suite.

    Args:
        problem_names: List of problem names to run (None = all)
        epsilon: Optimality gap tolerance
        max_time: Maximum time per problem
        max_nodes: Maximum nodes per problem
        verbose: Print progress

    Returns:
        Dictionary mapping problem names to results
    """
    if problem_names is None:
        problems = get_all_problems()
    else:
        problems = [get_problem(name) for name in problem_names]

    results = {}
    total_start = time.time()

    print("\n" + "=" * 80)
    print("GLOBALLib BENCHMARK - MATHEMATICAL CERTIFICATION")
    print("No shortcuts. No reference values. Pure gap closure.")
    print("=" * 80)
    print(f"Problems: {len(problems)}")
    print(f"Epsilon: {epsilon}")
    print(f"Max time per problem: {max_time}s")
    print(f"Max nodes per problem: {max_nodes}")
    print("=" * 80)

    for i, problem in enumerate(problems, 1):
        if verbose:
            print(f"\n[{i}/{len(problems)}] Running {problem.name}...")

        result = run_single_problem(
            problem,
            epsilon=epsilon,
            max_time=max_time,
            max_nodes=max_nodes,
            verbose=verbose
        )
        results[problem.name] = result

    total_elapsed = time.time() - total_start

    # Print summary
    _print_summary(results, total_elapsed, epsilon)

    return results


def _print_summary(results: Dict[str, GLOBALLibResult], total_time: float, epsilon: float):
    """Print benchmark summary."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    n_total = len(results)
    n_certified = sum(1 for r in results.values() if r.status == CertificationStatus.UNIQUE_OPT)
    n_unsat = sum(1 for r in results.values() if r.status == CertificationStatus.UNSAT)
    n_omega = sum(1 for r in results.values() if r.status == CertificationStatus.OMEGA_GAP)
    n_error = sum(1 for r in results.values() if r.status == CertificationStatus.ERROR)

    print(f"\nTotals (ε = {epsilon}):")
    print(f"  UNIQUE-OPT: {n_certified}/{n_total} ({100*n_certified/n_total:.1f}%)")
    print(f"  UNSAT:      {n_unsat}/{n_total} ({100*n_unsat/n_total:.1f}%)")
    print(f"  Ω-GAP:      {n_omega}/{n_total} ({100*n_omega/n_total:.1f}%)")
    print(f"  ERROR:      {n_error}/{n_total} ({100*n_error/n_total:.1f}%)")
    print(f"\nTotal time: {total_time:.2f}s")

    # Scoreboard
    print("\n" + "-" * 80)
    print(f"{'Problem':<30} {'Status':<12} {'Gap':<12} {'Nodes':<10} {'Time':<8}")
    print("-" * 80)

    for name, result in sorted(results.items()):
        gap_str = f"{result.gap:.2e}" if result.gap < float('inf') else "∞"
        print(f"{name:<30} {result.status.value:<12} {gap_str:<12} {result.nodes_explored:<10} {result.time_seconds:.2f}s")

    print("-" * 80)

    # Certification statement
    certified_pct = 100 * (n_certified + n_unsat) / n_total
    print(f"\n*** CERTIFICATION RATE: {certified_pct:.1f}% ***")
    print("(via pure mathematical gap closure, no reference values)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run GLOBALLib benchmark")
    parser.add_argument("--epsilon", type=float, default=1e-4, help="Gap tolerance")
    parser.add_argument("--max-time", type=float, default=120.0, help="Max time per problem")
    parser.add_argument("--max-nodes", type=int, default=50000, help="Max nodes per problem")
    parser.add_argument("--problems", nargs="+", help="Specific problems to run")
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")

    args = parser.parse_args()

    results = run_globallib_benchmark(
        problem_names=args.problems,
        epsilon=args.epsilon,
        max_time=args.max_time,
        max_nodes=args.max_nodes,
        verbose=not args.quiet
    )
