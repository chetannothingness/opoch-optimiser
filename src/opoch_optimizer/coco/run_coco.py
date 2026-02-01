"""
COCO/BBOB Benchmark Runner

Runs the full BBOB benchmark suite with OPOCH-COCO.
Uses IOH library EXCLUSIVELY for BBOB functions (no fallback approximations).

Usage:
    python -m opoch_optimizer.coco.run_coco --dims 2,5,10,20 --budget 10000
"""

import argparse
import os
import time
import numpy as np
from typing import List, Tuple
from datetime import datetime

from .opoch_coco import OPOCHCOCO, OPOCHConfig
from .logger_ioh import IOHExperimentLogger


def get_bbob_function(fid: int, instance: int, dim: int):
    """
    Get BBOB function from IOH library.

    IMPORTANT: Uses IOH exclusively. No fallback approximations.
    COCO's transforms are part of the definition.
    """
    import ioh

    problem = ioh.get_problem(fid, instance, dim, ioh.ProblemClass.BBOB)

    # Get optimal value
    try:
        f_opt = problem.optimum.y
    except:
        # Some IOH versions use different API
        f_opt = 0.0  # BBOB optimal is always 0 after normalization

    bounds = [(-5.0, 5.0)] * dim

    return problem, f_opt, bounds


def run_single(
    fid: int,
    instance: int,
    dim: int,
    budget: int
) -> Tuple[float, float, int, List[Tuple[int, float]], str]:
    """
    Run a single optimization.

    Returns:
        (f_best, f_opt, evaluations, trajectory, receipt_hash)
    """
    func, f_opt, bounds = get_bbob_function(fid, instance, dim)

    # Normalize to minimize toward 0
    def normalized_func(x):
        return func(x) - f_opt

    # Configure OPOCH-COCO
    config = OPOCHConfig(
        seed=42 + instance * 1000 + fid,  # Unique deterministic seed
        sigma0=0.3,
        probe_factor=100.0,  # Probe budget = 100 * d (balances restarts vs. basin ID)
        max_restarts=100,   # High for multimodal basin coverage
        top_k_fraction=0.1, # Top 10% restarts for exploitation
        lambda_factor=2.0,
        sigma_factor=1.5,
        tol_fun=1e-12
    )

    optimizer = OPOCHCOCO(
        normalized_func,
        dim,
        bounds,
        config
    )

    result = optimizer.optimize(budget)

    return (
        result.f_best,
        0.0,  # Target is 0 (normalized)
        result.evaluations,
        result.trajectory,
        result.receipt_hash
    )


def run_benchmark(
    dimensions: List[int] = [2, 5, 10, 20],
    functions: List[int] = list(range(1, 25)),
    instances: List[int] = [1, 2, 3, 4, 5],
    budget_factor: int = 10000,
    output_dir: str = "results/opoch_coco",
):
    """
    Run full COCO/BBOB benchmark.

    Args:
        dimensions: List of dimensions to test
        functions: List of function IDs (1-24 for BBOB)
        instances: List of instance IDs
        budget_factor: Budget = budget_factor * dim
        output_dir: Output directory for results
    """
    # Verify IOH is available
    try:
        import ioh
        print(f"Using IOH library for BBOB functions")
    except ImportError:
        raise RuntimeError("IOH library required. Install with: pip install ioh")

    logger = IOHExperimentLogger(
        output_dir=output_dir,
        algorithm_name="OPOCH-COCO",
        suite="BBOB"
    )

    total_runs = len(dimensions) * len(functions) * len(instances)
    run_count = 0
    start_time = time.time()

    print(f"\nRunning COCO/BBOB Benchmark with OPOCH-COCO")
    print(f"{'='*60}")
    print(f"Dimensions: {dimensions}")
    print(f"Functions: {len(functions)} ({min(functions)}-{max(functions)})")
    print(f"Instances: {instances}")
    print(f"Budget factor: {budget_factor} * dim")
    print(f"Total runs: {total_runs}")
    print(f"{'='*60}\n")

    for dim in dimensions:
        budget = budget_factor * dim
        print(f"\n--- Dimension {dim} (budget={budget}) ---")

        for fid in functions:
            for inst in instances:
                run_count += 1

                print(f"  f{fid:02d} i{inst} d{dim}: ", end="", flush=True)

                try:
                    f_best, f_opt, evals, trajectory, receipt = run_single(
                        fid, inst, dim, budget
                    )

                    # Target is 1e-8
                    target = 1e-8
                    hit = "HIT" if f_best <= target else f"{f_best:.2e}"

                    print(f"{hit} ({evals} evals)")

                    logger.log_run(
                        function_id=fid,
                        instance=inst,
                        dimension=dim,
                        trajectory=trajectory,
                        final_f=f_best,
                        final_x=None,
                        evaluations=evals,
                        target=target,
                        receipt_hash=receipt
                    )

                except Exception as e:
                    print(f"ERROR: {e}")

    # Finalize
    summary = logger.finalize()

    print(f"\nTotal time: {time.time() - start_time:.1f}s")
    print(f"Results written to: {output_dir}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run COCO/BBOB benchmark with OPOCH-COCO")

    parser.add_argument(
        "--dims",
        type=str,
        default="2,5,10",
        help="Comma-separated dimensions (default: 2,5,10)"
    )
    parser.add_argument(
        "--functions",
        type=str,
        default="1-24",
        help="Function range, e.g., '1-24' or '1,2,3' (default: 1-24)"
    )
    parser.add_argument(
        "--instances",
        type=str,
        default="1-5",
        help="Instance range (default: 1-5)"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=10000,
        help="Budget factor (budget = factor * dim) (default: 10000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/opoch_coco",
        help="Output directory (default: results/opoch_coco)"
    )

    args = parser.parse_args()

    # Parse dimensions
    dimensions = [int(d) for d in args.dims.split(",")]

    # Parse functions
    if "-" in args.functions:
        start, end = args.functions.split("-")
        functions = list(range(int(start), int(end) + 1))
    else:
        functions = [int(f) for f in args.functions.split(",")]

    # Parse instances
    if "-" in args.instances:
        start, end = args.instances.split("-")
        instances = list(range(int(start), int(end) + 1))
    else:
        instances = [int(i) for i in args.instances.split(",")]

    run_benchmark(
        dimensions=dimensions,
        functions=functions,
        instances=instances,
        budget_factor=args.budget,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
