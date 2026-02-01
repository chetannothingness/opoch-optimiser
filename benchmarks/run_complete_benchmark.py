"""
Complete GLOBALLib Benchmark with Baseline Comparison

Runs all problems and compares:
1. OPOCH (certified gap closure)
2. SciPy SLSQP (local solver)
3. SciPy differential_evolution (global heuristic)
4. SciPy basinhopping (global heuristic)

OPOCH certification is via gap = UB - LB ≤ ε.
Baselines are compared on solution quality only (no certification).
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import warnings

from opoch_optimizer.solver.opoch_kernel import OPOCHKernel, OPOCHConfig
from opoch_optimizer.core.output_gate import Verdict

from benchmarks.globallib_complete import get_all_problems, GLOBALLibProblem


class Status(Enum):
    SOLVED = "SOLVED"       # Gap closed, certified optimal
    OMEGA = "Ω-GAP"         # Budget exhausted
    ERROR = "ERROR"


@dataclass
class SolverResult:
    """Result from a solver."""
    solver_name: str
    problem_name: str
    status: str
    objective: float
    solution: Optional[np.ndarray]
    gap: float  # For OPOCH: UB - LB, for baselines: |f - f*|
    time_seconds: float
    certified: bool  # True only for OPOCH with closed gap
    nodes: int = 0
    evals: int = 0


def run_opoch(problem: GLOBALLibProblem, epsilon: float, max_time: float, max_nodes: int) -> SolverResult:
    """Run OPOCH solver with certified gap closure."""
    try:
        contract = problem.to_problem_contract()
        config = OPOCHConfig(
            epsilon=epsilon,
            max_time=max_time,
            max_nodes=max_nodes,
            log_frequency=100000  # Quiet
        )

        start = time.time()
        kernel = OPOCHKernel(contract, config)
        verdict, result = kernel.solve()
        elapsed = time.time() - start

        ub = result.upper_bound if hasattr(result, 'upper_bound') else float('inf')
        lb = result.lower_bound if hasattr(result, 'lower_bound') else float('-inf')
        gap = ub - lb

        solution = None
        if hasattr(result, 'x_optimal') and result.x_optimal is not None:
            solution = result.x_optimal
        elif hasattr(result, 'x_best') and result.x_best is not None:
            solution = result.x_best

        certified = verdict == Verdict.UNIQUE_OPT and gap <= epsilon
        status = "SOLVED" if certified else ("Ω-GAP" if verdict == Verdict.OMEGA_GAP else str(verdict))

        return SolverResult(
            solver_name="OPOCH",
            problem_name=problem.name,
            status=status,
            objective=ub,
            solution=solution,
            gap=gap,
            time_seconds=elapsed,
            certified=certified,
            nodes=result.nodes_explored if hasattr(result, 'nodes_explored') else 0
        )

    except Exception as e:
        return SolverResult(
            solver_name="OPOCH",
            problem_name=problem.name,
            status=f"ERROR: {e}",
            objective=float('inf'),
            solution=None,
            gap=float('inf'),
            time_seconds=0.0,
            certified=False
        )


def run_scipy_slsqp(problem: GLOBALLibProblem, max_time: float) -> SolverResult:
    """Run SciPy SLSQP (local solver)."""
    from scipy.optimize import minimize

    try:
        x0 = np.array([(b[0] + b[1])/2 for b in problem.bounds])
        bounds = problem.bounds

        constraints = []
        for g in problem.ineq_constraints:
            constraints.append({'type': 'ineq', 'fun': lambda x, g=g: -g(x)})
        for h in problem.eq_constraints:
            constraints.append({'type': 'eq', 'fun': h})

        start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                problem.objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
        elapsed = time.time() - start

        obj = result.fun if result.success else float('inf')
        gap = abs(obj - problem.known_optimal)

        return SolverResult(
            solver_name="SLSQP",
            problem_name=problem.name,
            status="SOLVED" if result.success else "FAILED",
            objective=obj,
            solution=result.x if result.success else None,
            gap=gap,
            time_seconds=elapsed,
            certified=False,  # Local solver, no certification
            evals=result.nfev if hasattr(result, 'nfev') else 0
        )

    except Exception as e:
        return SolverResult(
            solver_name="SLSQP",
            problem_name=problem.name,
            status=f"ERROR",
            objective=float('inf'),
            solution=None,
            gap=float('inf'),
            time_seconds=0.0,
            certified=False
        )


def run_scipy_de(problem: GLOBALLibProblem, max_time: float) -> SolverResult:
    """Run SciPy differential_evolution (global heuristic)."""
    from scipy.optimize import differential_evolution

    # Skip if has equality constraints (DE doesn't handle them well)
    if problem.eq_constraints:
        return SolverResult(
            solver_name="DE",
            problem_name=problem.name,
            status="SKIP",
            objective=float('inf'),
            solution=None,
            gap=float('inf'),
            time_seconds=0.0,
            certified=False
        )

    try:
        bounds = problem.bounds

        constraints = []
        for g in problem.ineq_constraints:
            from scipy.optimize import NonlinearConstraint
            constraints.append(NonlinearConstraint(g, -np.inf, 0))

        start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = differential_evolution(
                problem.objective,
                bounds,
                constraints=constraints if constraints else (),
                maxiter=1000,
                tol=1e-8,
                seed=42
            )
        elapsed = time.time() - start

        obj = result.fun
        gap = abs(obj - problem.known_optimal)

        return SolverResult(
            solver_name="DE",
            problem_name=problem.name,
            status="SOLVED" if result.success else "FAILED",
            objective=obj,
            solution=result.x,
            gap=gap,
            time_seconds=elapsed,
            certified=False,
            evals=result.nfev
        )

    except Exception as e:
        return SolverResult(
            solver_name="DE",
            problem_name=problem.name,
            status="ERROR",
            objective=float('inf'),
            solution=None,
            gap=float('inf'),
            time_seconds=0.0,
            certified=False
        )


def run_scipy_basinhopping(problem: GLOBALLibProblem, max_time: float) -> SolverResult:
    """Run SciPy basinhopping (global heuristic)."""
    from scipy.optimize import basinhopping

    # Skip if has constraints (basinhopping doesn't handle them directly)
    if problem.eq_constraints or problem.ineq_constraints:
        return SolverResult(
            solver_name="BH",
            problem_name=problem.name,
            status="SKIP",
            objective=float('inf'),
            solution=None,
            gap=float('inf'),
            time_seconds=0.0,
            certified=False
        )

    try:
        x0 = np.array([(b[0] + b[1])/2 for b in problem.bounds])

        class BoundsChecker:
            def __init__(self, bounds):
                self.bounds = bounds
            def __call__(self, **kwargs):
                x = kwargs["x_new"]
                for i, (lo, hi) in enumerate(self.bounds):
                    if x[i] < lo or x[i] > hi:
                        return False
                return True

        start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = basinhopping(
                problem.objective,
                x0,
                niter=100,
                accept_test=BoundsChecker(problem.bounds),
                seed=42
            )
        elapsed = time.time() - start

        obj = result.fun
        gap = abs(obj - problem.known_optimal)

        return SolverResult(
            solver_name="BH",
            problem_name=problem.name,
            status="SOLVED",
            objective=obj,
            solution=result.x,
            gap=gap,
            time_seconds=elapsed,
            certified=False,
            evals=result.nfev if hasattr(result, 'nfev') else 0
        )

    except Exception as e:
        return SolverResult(
            solver_name="BH",
            problem_name=problem.name,
            status="ERROR",
            objective=float('inf'),
            solution=None,
            gap=float('inf'),
            time_seconds=0.0,
            certified=False
        )


def run_complete_benchmark(
    epsilon: float = 1e-4,
    max_time: float = 60.0,
    max_nodes: int = 50000,
    run_baselines: bool = True
):
    """Run complete benchmark with all solvers."""

    problems = get_all_problems()

    print("=" * 100)
    print("COMPLETE GLOBALLib BENCHMARK")
    print("OPOCH: Mathematical certification via gap closure (UB - LB ≤ ε)")
    print("Baselines: Solution quality comparison (|f - f*| for reference only)")
    print("=" * 100)
    print(f"Problems: {len(problems)}")
    print(f"OPOCH epsilon: {epsilon}")
    print(f"Max time per problem: {max_time}s")
    print("=" * 100)

    all_results: Dict[str, List[SolverResult]] = {
        "OPOCH": [],
        "SLSQP": [],
        "DE": [],
        "BH": []
    }

    for i, problem in enumerate(problems, 1):
        print(f"\n[{i}/{len(problems)}] {problem.name}: {problem.description}")

        # Run OPOCH
        opoch_result = run_opoch(problem, epsilon, max_time, max_nodes)
        all_results["OPOCH"].append(opoch_result)
        status_str = "✓" if opoch_result.certified else "✗"
        print(f"  OPOCH: {status_str} obj={opoch_result.objective:.6g}, gap={opoch_result.gap:.2e}, time={opoch_result.time_seconds:.2f}s")

        if run_baselines:
            # Run SLSQP
            slsqp_result = run_scipy_slsqp(problem, max_time)
            all_results["SLSQP"].append(slsqp_result)

            # Run DE
            de_result = run_scipy_de(problem, max_time)
            all_results["DE"].append(de_result)

            # Run Basinhopping
            bh_result = run_scipy_basinhopping(problem, max_time)
            all_results["BH"].append(bh_result)

    # Print summary
    _print_summary(problems, all_results, epsilon, run_baselines)

    return all_results


def _print_summary(problems, all_results, epsilon, run_baselines):
    """Print comprehensive summary."""

    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)

    n = len(problems)

    # OPOCH stats
    opoch = all_results["OPOCH"]
    n_certified = sum(1 for r in opoch if r.certified)
    n_solved = sum(1 for r in opoch if "SOLVED" in r.status)
    avg_time = np.mean([r.time_seconds for r in opoch])
    avg_gap = np.mean([r.gap for r in opoch if r.gap < float('inf')])

    print(f"\nOPOCH (ε = {epsilon}):")
    print(f"  Certified (gap ≤ ε): {n_certified}/{n} ({100*n_certified/n:.1f}%)")
    print(f"  Average gap: {avg_gap:.2e}")
    print(f"  Average time: {avg_time:.2f}s")

    if run_baselines:
        for solver in ["SLSQP", "DE", "BH"]:
            results = all_results[solver]
            n_solved = sum(1 for r in results if r.status == "SOLVED")
            n_skip = sum(1 for r in results if r.status == "SKIP")
            avg_gap = np.mean([r.gap for r in results if r.gap < float('inf') and r.status == "SOLVED"] or [float('inf')])
            avg_time = np.mean([r.time_seconds for r in results if r.status != "SKIP"])

            print(f"\n{solver}:")
            print(f"  Solved: {n_solved}/{n-n_skip} ({100*n_solved/(n-n_skip):.1f}% of applicable)")
            print(f"  Skipped: {n_skip} (constraints not supported)")
            print(f"  Average gap to known optimal: {avg_gap:.2e}")
            print(f"  Average time: {avg_time:.2f}s")

    # Detailed table
    print("\n" + "-" * 100)
    if run_baselines:
        print(f"{'Problem':<25} {'OPOCH':<15} {'SLSQP':<15} {'DE':<15} {'BH':<15}")
    else:
        print(f"{'Problem':<25} {'Status':<12} {'Objective':<15} {'Gap':<12} {'Time':<10} {'Nodes':<10}")
    print("-" * 100)

    for i, problem in enumerate(problems):
        opoch = all_results["OPOCH"][i]

        if run_baselines:
            slsqp = all_results["SLSQP"][i]
            de = all_results["DE"][i]
            bh = all_results["BH"][i]

            def fmt(r):
                if r.status == "SKIP":
                    return "SKIP"
                elif r.certified:
                    return f"✓{r.gap:.1e}"
                elif r.gap < 1e-4:
                    return f"{r.gap:.1e}"
                else:
                    return f"{r.gap:.1e}"

            print(f"{problem.name:<25} {fmt(opoch):<15} {fmt(slsqp):<15} {fmt(de):<15} {fmt(bh):<15}")
        else:
            status_icon = "✓" if opoch.certified else "✗"
            gap_str = f"{opoch.gap:.2e}" if opoch.gap < float('inf') else "∞"
            print(f"{problem.name:<25} {status_icon} {opoch.status:<10} {opoch.objective:<15.6g} {gap_str:<12} {opoch.time_seconds:<10.2f} {opoch.nodes:<10}")

    print("-" * 100)

    # Final certification rate
    n_certified = sum(1 for r in all_results["OPOCH"] if r.certified)
    print(f"\n*** OPOCH CERTIFICATION RATE: {n_certified}/{n} = {100*n_certified/n:.1f}% ***")
    print("(Certification = mathematical proof that UB - LB ≤ ε)")

    if run_baselines:
        print("\n*** BASELINE COMPARISON ***")
        print("(Baselines show |f - f*| where f* is known optimal - NOT certified)")
        print("OPOCH is the ONLY solver that provides mathematical certification.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Complete GLOBALLib benchmark")
    parser.add_argument("--epsilon", type=float, default=1e-4, help="Gap tolerance")
    parser.add_argument("--max-time", type=float, default=60.0, help="Max time per problem")
    parser.add_argument("--max-nodes", type=int, default=50000, help="Max nodes per problem")
    parser.add_argument("--no-baselines", action="store_true", help="Skip baseline solvers")

    args = parser.parse_args()

    run_complete_benchmark(
        epsilon=args.epsilon,
        max_time=args.max_time,
        max_nodes=args.max_nodes,
        run_baselines=not args.no_baselines
    )
