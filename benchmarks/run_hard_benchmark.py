"""
Run GLOBALLib HARD Benchmark Suite

Complete, honest benchmark with mathematical certification.
NO SHORTCUTS. Pure math via gap closure (UB - LB <= epsilon).
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
import warnings
from collections import defaultdict

from opoch_optimizer.solver.opoch_kernel import OPOCHKernel, OPOCHConfig
from opoch_optimizer.core.output_gate import Verdict

from benchmarks.globallib_hard import get_all, HardProblem


@dataclass
class BenchmarkResult:
    """Result from a solver."""
    solver: str
    problem: str
    category: str
    difficulty: str
    status: str
    objective: float
    gap: float
    time_seconds: float
    certified: bool
    nodes: int = 0


def run_opoch(problem: HardProblem, epsilon: float, max_time: float, max_nodes: int) -> BenchmarkResult:
    """Run OPOCH with certified gap closure."""
    try:
        contract = problem.to_problem_contract()
        config = OPOCHConfig(
            epsilon=epsilon,
            max_time=max_time,
            max_nodes=max_nodes,
            log_frequency=1000000  # Quiet
        )

        start = time.time()
        kernel = OPOCHKernel(contract, config)
        verdict, result = kernel.solve()
        elapsed = time.time() - start

        ub = getattr(result, 'upper_bound', float('inf'))
        lb = getattr(result, 'lower_bound', float('-inf'))
        gap = ub - lb

        certified = verdict == Verdict.UNIQUE_OPT and gap <= epsilon
        status = "CERTIFIED" if certified else (
            "OMEGA-GAP" if verdict == Verdict.OMEGA_GAP else str(verdict)
        )

        return BenchmarkResult(
            solver="OPOCH",
            problem=problem.name,
            category=problem.category,
            difficulty=problem.difficulty,
            status=status,
            objective=ub,
            gap=gap,
            time_seconds=elapsed,
            certified=certified,
            nodes=getattr(result, 'nodes_explored', 0)
        )

    except Exception as e:
        return BenchmarkResult(
            solver="OPOCH",
            problem=problem.name,
            category=problem.category,
            difficulty=problem.difficulty,
            status=f"ERROR: {str(e)[:50]}",
            objective=float('inf'),
            gap=float('inf'),
            time_seconds=0.0,
            certified=False
        )


def run_scipy_slsqp(problem: HardProblem, max_time: float) -> BenchmarkResult:
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
        gap = abs(obj - problem.known_optimal) if np.isfinite(problem.known_optimal) else float('inf')

        return BenchmarkResult(
            solver="SLSQP",
            problem=problem.name,
            category=problem.category,
            difficulty=problem.difficulty,
            status="SOLVED" if result.success else "FAILED",
            objective=obj,
            gap=gap,
            time_seconds=elapsed,
            certified=False
        )

    except Exception as e:
        return BenchmarkResult(
            solver="SLSQP",
            problem=problem.name,
            category=problem.category,
            difficulty=problem.difficulty,
            status="ERROR",
            objective=float('inf'),
            gap=float('inf'),
            time_seconds=0.0,
            certified=False
        )


def run_scipy_de(problem: HardProblem, max_time: float) -> BenchmarkResult:
    """Run SciPy differential_evolution."""
    from scipy.optimize import differential_evolution, NonlinearConstraint

    # Skip if has equality constraints
    if problem.eq_constraints:
        return BenchmarkResult(
            solver="DE",
            problem=problem.name,
            category=problem.category,
            difficulty=problem.difficulty,
            status="SKIP-EQ",
            objective=float('inf'),
            gap=float('inf'),
            time_seconds=0.0,
            certified=False
        )

    try:
        bounds = problem.bounds
        constraints = []
        for g in problem.ineq_constraints:
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
        gap = abs(obj - problem.known_optimal) if np.isfinite(problem.known_optimal) else float('inf')

        return BenchmarkResult(
            solver="DE",
            problem=problem.name,
            category=problem.category,
            difficulty=problem.difficulty,
            status="SOLVED" if result.success else "FAILED",
            objective=obj,
            gap=gap,
            time_seconds=elapsed,
            certified=False
        )

    except Exception as e:
        return BenchmarkResult(
            solver="DE",
            problem=problem.name,
            category=problem.category,
            difficulty=problem.difficulty,
            status="ERROR",
            objective=float('inf'),
            gap=float('inf'),
            time_seconds=0.0,
            certified=False
        )


def run_scipy_bh(problem: HardProblem, max_time: float) -> BenchmarkResult:
    """Run SciPy basinhopping."""
    from scipy.optimize import basinhopping

    # Skip if has constraints
    if problem.eq_constraints or problem.ineq_constraints:
        return BenchmarkResult(
            solver="BH",
            problem=problem.name,
            category=problem.category,
            difficulty=problem.difficulty,
            status="SKIP-CONSTR",
            objective=float('inf'),
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
        gap = abs(obj - problem.known_optimal) if np.isfinite(problem.known_optimal) else float('inf')

        return BenchmarkResult(
            solver="BH",
            problem=problem.name,
            category=problem.category,
            difficulty=problem.difficulty,
            status="SOLVED",
            objective=obj,
            gap=gap,
            time_seconds=elapsed,
            certified=False
        )

    except Exception as e:
        return BenchmarkResult(
            solver="BH",
            problem=problem.name,
            category=problem.category,
            difficulty=problem.difficulty,
            status="ERROR",
            objective=float('inf'),
            gap=float('inf'),
            time_seconds=0.0,
            certified=False
        )


def run_benchmark(
    epsilon: float = 1e-4,
    max_time: float = 60.0,
    max_nodes: int = 50000,
    run_baselines: bool = True
):
    """Run complete hard benchmark."""

    problems = get_all()

    # Count by category and difficulty
    by_category = defaultdict(list)
    by_difficulty = defaultdict(list)
    for p in problems:
        by_category[p.category].append(p)
        by_difficulty[p.difficulty].append(p)

    print("=" * 100)
    print("GLOBALLib HARD BENCHMARK - Pure Mathematics, No Shortcuts")
    print("=" * 100)
    print(f"Total problems: {len(problems)}")
    print()
    print("By Category:")
    for cat, ps in sorted(by_category.items()):
        diff_counts = defaultdict(int)
        for p in ps:
            diff_counts[p.difficulty] += 1
        diff_str = ", ".join(f"{d}:{c}" for d, c in sorted(diff_counts.items()))
        print(f"  {cat}: {len(ps)} problems ({diff_str})")
    print()
    print("By Difficulty:")
    for diff, ps in sorted(by_difficulty.items()):
        print(f"  {diff}: {len(ps)} problems")
    print()
    print(f"Certification: gap = UB - LB <= {epsilon}")
    print(f"Max time per problem: {max_time}s")
    print(f"Max nodes per problem: {max_nodes}")
    print("=" * 100)

    results: Dict[str, List[BenchmarkResult]] = {
        "OPOCH": [],
        "SLSQP": [],
        "DE": [],
        "BH": []
    }

    for i, problem in enumerate(problems, 1):
        print(f"\n[{i:02d}/{len(problems)}] {problem.name} ({problem.category}, {problem.difficulty})")

        # OPOCH
        opoch = run_opoch(problem, epsilon, max_time, max_nodes)
        results["OPOCH"].append(opoch)
        cert_mark = "CERTIFIED" if opoch.certified else opoch.status
        print(f"  OPOCH: {cert_mark}, obj={opoch.objective:.6g}, gap={opoch.gap:.2e}, t={opoch.time_seconds:.2f}s")

        if run_baselines:
            # SLSQP
            slsqp = run_scipy_slsqp(problem, max_time)
            results["SLSQP"].append(slsqp)

            # DE
            de = run_scipy_de(problem, max_time)
            results["DE"].append(de)

            # BH
            bh = run_scipy_bh(problem, max_time)
            results["BH"].append(bh)

    # Print summary
    print_summary(problems, results, epsilon, run_baselines)
    return results


def print_summary(problems, results, epsilon, run_baselines):
    """Print comprehensive summary."""

    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)

    n = len(problems)
    opoch_results = results["OPOCH"]

    # OPOCH summary
    n_certified = sum(1 for r in opoch_results if r.certified)
    avg_time = np.mean([r.time_seconds for r in opoch_results])
    gaps = [r.gap for r in opoch_results if r.gap < float('inf')]
    avg_gap = np.mean(gaps) if gaps else float('inf')

    print(f"\nOPOCH (epsilon = {epsilon}):")
    print(f"  CERTIFIED: {n_certified}/{n} ({100*n_certified/n:.1f}%)")
    print(f"  Average gap: {avg_gap:.2e}")
    print(f"  Average time: {avg_time:.2f}s")

    # By category
    print("\n  By Category:")
    for cat in ["unconstrained", "inequality", "equality", "mixed"]:
        cat_results = [r for r in opoch_results if r.category == cat]
        if cat_results:
            cat_cert = sum(1 for r in cat_results if r.certified)
            print(f"    {cat}: {cat_cert}/{len(cat_results)} certified")

    # By difficulty
    print("\n  By Difficulty:")
    for diff in ["easy", "medium", "hard", "extreme"]:
        diff_results = [r for r in opoch_results if r.difficulty == diff]
        if diff_results:
            diff_cert = sum(1 for r in diff_results if r.certified)
            print(f"    {diff}: {diff_cert}/{len(diff_results)} certified")

    if run_baselines:
        for solver in ["SLSQP", "DE", "BH"]:
            solver_results = results[solver]
            n_skip = sum(1 for r in solver_results if r.status.startswith("SKIP"))
            n_solved = sum(1 for r in solver_results if r.status == "SOLVED")
            n_applicable = n - n_skip
            gaps = [r.gap for r in solver_results if r.status == "SOLVED" and r.gap < float('inf')]
            avg_gap = np.mean(gaps) if gaps else float('inf')

            print(f"\n{solver}:")
            if n_applicable > 0:
                print(f"  Solved: {n_solved}/{n_applicable} ({100*n_solved/n_applicable:.1f}% of applicable)")
            else:
                print(f"  Solved: 0/0 (no applicable problems)")
            print(f"  Skipped: {n_skip}")
            print(f"  Average gap to known: {avg_gap:.2e}")

    # Detailed table
    print("\n" + "-" * 100)
    header = f"{'Problem':<22} {'Cat':<12} {'Diff':<8} {'OPOCH':<12}"
    if run_baselines:
        header += f" {'SLSQP':<10} {'DE':<10} {'BH':<10}"
    print(header)
    print("-" * 100)

    for i, problem in enumerate(problems):
        opoch = results["OPOCH"][i]

        def fmt(r, is_opoch=False):
            if r.status.startswith("SKIP"):
                return "SKIP"
            elif r.certified if is_opoch else False:
                return f"C:{r.gap:.1e}"
            elif r.gap < 1e-2:
                return f"{r.gap:.1e}"
            elif r.gap < float('inf'):
                return f"{r.gap:.1e}"
            else:
                return "FAIL"

        row = f"{problem.name:<22} {problem.category:<12} {problem.difficulty:<8} {fmt(opoch, True):<12}"
        if run_baselines:
            slsqp = results["SLSQP"][i]
            de = results["DE"][i]
            bh = results["BH"][i]
            row += f" {fmt(slsqp):<10} {fmt(de):<10} {fmt(bh):<10}"
        print(row)

    print("-" * 100)

    # Final certification
    n_certified = sum(1 for r in results["OPOCH"] if r.certified)
    print(f"\n{'='*100}")
    print(f"OPOCH MATHEMATICAL CERTIFICATION RATE: {n_certified}/{n} = {100*n_certified/n:.1f}%")
    print(f"{'='*100}")
    print("Certification = mathematical PROOF that UB - LB <= epsilon")
    print("OPOCH is the ONLY solver providing certified global optimization.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GLOBALLib HARD Benchmark")
    parser.add_argument("--epsilon", type=float, default=1e-4, help="Gap tolerance")
    parser.add_argument("--max-time", type=float, default=60.0, help="Max time per problem")
    parser.add_argument("--max-nodes", type=int, default=50000, help="Max nodes per problem")
    parser.add_argument("--no-baselines", action="store_true", help="Skip baseline solvers")

    args = parser.parse_args()

    run_benchmark(
        epsilon=args.epsilon,
        max_time=args.max_time,
        max_nodes=args.max_nodes,
        run_baselines=not args.no_baselines
    )
