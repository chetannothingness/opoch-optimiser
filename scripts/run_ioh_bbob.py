#!/usr/bin/env python3
"""
OPOCH IOHprofiler BBOB Benchmark Runner

Produces IOH-analyzer compatible output for standard BBOB functions.
Tests dimensions d=2,10,20 with 100k evaluation budget.

Output format matches IOHprofiler v0.3.18+ specification.

NO SHORTCUTS. Real math. Real results.
"""

import sys
import time
import json
import zipfile
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
import argparse

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from opoch_optimizer import (
    ExpressionGraph,
    OpType,
    ProblemContract,
    OPOCHKernel,
    OPOCHConfig,
    Verdict,
)
from opoch_optimizer.primal import PhaseProbe


@dataclass
class EvalTracker:
    """Tracks function evaluations with best-so-far trajectory."""
    evals: int = 0
    best_y: float = float('inf')
    trajectory: List[Tuple[int, float]] = field(default_factory=list)

    def track(self, y: float) -> float:
        self.evals += 1
        if y < self.best_y:
            self.best_y = y
            self.trajectory.append((self.evals, y))
        return y

    def reset(self):
        self.evals = 0
        self.best_y = float('inf')
        self.trajectory = []


@dataclass
class RunResult:
    """Result of a single benchmark run."""
    function: str
    dim: int
    instance: int
    certified: bool
    time: float
    nodes: int
    total_evals: int
    trajectory: List[Tuple[int, float]]
    best_x: Optional[np.ndarray]
    best_y: float
    gap: float
    error_from_optimum: Optional[float]


# ==============================================================================
# BBOB Function Builders (Expression Graph)
# ==============================================================================

def build_sphere(n: int, shift: np.ndarray) -> ExpressionGraph:
    """
    Shifted Sphere: f(x) = sum((x_i - shift_i)^2)
    Optimum at x = shift with f* = 0
    """
    g = ExpressionGraph()
    terms = []
    for i in range(n):
        v = g.variable(i)
        s = g.constant(shift[i])
        diff = g.binary(OpType.SUB, v, s)
        terms.append(g.unary(OpType.SQUARE, diff))

    result = terms[0]
    for t in terms[1:]:
        result = g.binary(OpType.ADD, result, t)
    g.set_output(result)
    return g


def build_ellipsoid(n: int, shift: np.ndarray) -> ExpressionGraph:
    """
    Shifted Ellipsoid: f(x) = sum(10^(6*(i-1)/(n-1)) * (x_i - shift_i)^2)
    Optimum at x = shift with f* = 0
    """
    g = ExpressionGraph()
    terms = []
    for i in range(n):
        v = g.variable(i)
        s = g.constant(shift[i])
        diff = g.binary(OpType.SUB, v, s)
        sq = g.unary(OpType.SQUARE, diff)

        if n > 1:
            coef = 10 ** (6 * i / (n - 1))
        else:
            coef = 1.0
        c = g.constant(coef)
        terms.append(g.binary(OpType.MUL, c, sq))

    result = terms[0]
    for t in terms[1:]:
        result = g.binary(OpType.ADD, result, t)
    g.set_output(result)
    return g


def build_rastrigin(n: int, shift: np.ndarray) -> ExpressionGraph:
    """
    Shifted Rastrigin: f(x) = 10n + sum((x_i-s_i)^2 - 10*cos(2*pi*(x_i-s_i)))
    Optimum at x = shift with f* = 0
    """
    g = ExpressionGraph()
    ten = g.constant(10.0)
    two_pi = g.constant(2.0 * np.pi)
    base = g.constant(10.0 * n)

    result = base
    for i in range(n):
        v = g.variable(i)
        s = g.constant(shift[i])
        diff = g.binary(OpType.SUB, v, s)

        diff_sq = g.unary(OpType.SQUARE, diff)
        angle = g.binary(OpType.MUL, two_pi, diff)
        cos_term = g.unary(OpType.COS, angle)
        scaled_cos = g.binary(OpType.MUL, ten, cos_term)
        term = g.binary(OpType.SUB, diff_sq, scaled_cos)
        result = g.binary(OpType.ADD, result, term)

    g.set_output(result)
    return g


def build_rosenbrock(n: int, shift: np.ndarray) -> ExpressionGraph:
    """
    Shifted Rosenbrock: f(x) = sum(100*(x_{i+1}-s_{i+1}-(x_i-s_i)^2)^2 + (1-(x_i-s_i))^2)
    Optimum at x = shift + 1 with f* = 0
    """
    g = ExpressionGraph()
    hundred = g.constant(100.0)
    one = g.constant(1.0)

    terms = []
    for i in range(n - 1):
        vi = g.variable(i)
        vi1 = g.variable(i + 1)
        si = g.constant(shift[i])
        si1 = g.constant(shift[i + 1])

        # (x_i - s_i)
        diff_i = g.binary(OpType.SUB, vi, si)
        # (x_{i+1} - s_{i+1})
        diff_i1 = g.binary(OpType.SUB, vi1, si1)

        # (x_i - s_i)^2
        diff_i_sq = g.unary(OpType.SQUARE, diff_i)
        # x_{i+1} - s_{i+1} - (x_i - s_i)^2
        inner = g.binary(OpType.SUB, diff_i1, diff_i_sq)
        # 100 * (...)^2
        term1 = g.binary(OpType.MUL, hundred, g.unary(OpType.SQUARE, inner))

        # 1 - (x_i - s_i)
        one_minus = g.binary(OpType.SUB, one, diff_i)
        # (1 - (x_i - s_i))^2
        term2 = g.unary(OpType.SQUARE, one_minus)

        terms.append(g.binary(OpType.ADD, term1, term2))

    if not terms:
        g.set_output(g.constant(0.0))
    else:
        result = terms[0]
        for t in terms[1:]:
            result = g.binary(OpType.ADD, result, t)
        g.set_output(result)

    return g


def build_schwefel(n: int, shift: np.ndarray) -> ExpressionGraph:
    """
    Shifted Schwefel: f(x) = 418.9829*n - sum((x_i-s_i)*sin(sqrt(|x_i-s_i|)))
    Global optimum at x = shift + 420.9687... with f* ≈ 0
    """
    g = ExpressionGraph()
    base = g.constant(418.9829 * n)

    result = base
    for i in range(n):
        v = g.variable(i)
        s = g.constant(shift[i])
        diff = g.binary(OpType.SUB, v, s)

        abs_diff = g.unary(OpType.ABS, diff)
        sqrt_abs = g.unary(OpType.SQRT, abs_diff)
        sin_term = g.unary(OpType.SIN, sqrt_abs)
        product = g.binary(OpType.MUL, diff, sin_term)
        result = g.binary(OpType.SUB, result, product)

    g.set_output(result)
    return g


# ==============================================================================
# BBOB Function Definitions
# ==============================================================================

BBOB_FUNCTIONS = {
    1: {
        'name': 'Sphere',
        'builder': build_sphere,
        'bounds': (-5, 5),
        'optimum_offset': 0.0,  # Optimum at shift
    },
    2: {
        'name': 'Ellipsoid',
        'builder': build_ellipsoid,
        'bounds': (-5, 5),
        'optimum_offset': 0.0,
    },
    3: {
        'name': 'Rastrigin',
        'builder': build_rastrigin,
        'bounds': (-5.12, 5.12),
        'optimum_offset': 0.0,
    },
    8: {
        'name': 'Rosenbrock',
        'builder': build_rosenbrock,
        'bounds': (-5, 10),
        'optimum_offset': 1.0,  # Optimum at shift + 1
    },
}


# ==============================================================================
# Benchmark Runner
# ==============================================================================

def wrap_evaluate(graph: ExpressionGraph, tracker: EvalTracker):
    """Wrap graph.evaluate() to track calls."""
    original = graph.evaluate

    def tracked(x):
        y = original(x)
        tracker.track(y)
        return y

    graph.evaluate = tracked


def run_single_problem(
    graph: ExpressionGraph,
    bounds: List[Tuple[float, float]],
    name: str,
    tracker: EvalTracker,
    max_time: float = 60.0,
    max_evals: int = 100000,
    shift: Optional[np.ndarray] = None,
    optimum_offset: float = 0.0,
) -> RunResult:
    """Run OPOCH kernel on a single problem with tracking."""
    wrap_evaluate(graph, tracker)

    dim = len(bounds)

    problem = ProblemContract(
        objective=graph,
        bounds=bounds,
        epsilon=1e-6,
        name=name
    )
    problem._obj_graph = graph

    config = OPOCHConfig(
        epsilon=1e-6,
        max_time=max_time,
        max_nodes=max_evals,
        log_frequency=max_evals
    )

    start = time.time()
    kernel = OPOCHKernel(problem, config)
    verdict, _ = kernel.solve()
    elapsed = time.time() - start

    error = None
    if shift is not None and kernel.best_solution is not None:
        true_optimum = shift + optimum_offset
        error = np.linalg.norm(kernel.best_solution - true_optimum)

    return RunResult(
        function=name.split('-')[0],
        dim=dim,
        instance=int(name.split('i')[-1]) if 'i' in name else 1,
        certified=verdict == Verdict.UNIQUE_OPT,
        time=elapsed,
        nodes=kernel.nodes_explored,
        total_evals=tracker.evals,
        trajectory=tracker.trajectory.copy(),
        best_x=kernel.best_solution,
        best_y=kernel.upper_bound,
        gap=kernel.upper_bound - kernel.lower_bound,
        error_from_optimum=error
    )


# ==============================================================================
# IOH Output Writers
# ==============================================================================

def write_dat_file(filepath: Path, trajectories: List[List[Tuple[int, float]]]):
    """Write IOHprofiler .dat format file."""
    with open(filepath, 'w') as f:
        for traj in trajectories:
            f.write("evaluations raw_y\n")
            for evals, y in traj:
                f.write(f"{evals} {y}\n")


def write_json_file(
    filepath: Path,
    func_name: str,
    func_id: int,
    runs_by_dim: Dict[int, List[RunResult]],
    algo_name: str
):
    """Write IOHprofiler .json format file."""
    scenarios = []
    for dim, runs in sorted(runs_by_dim.items()):
        scenario = {
            "dimension": dim,
            "path": f"data_f{func_id}_{func_name}/IOHprofiler_f{func_id}_DIM{dim}.dat",
            "runs": []
        }
        for i, run in enumerate(runs, 1):
            traj = run.trajectory
            final_eval = traj[-1][0] if traj else run.total_evals
            final_y = traj[-1][1] if traj else run.best_y

            scenario["runs"].append({
                "instance": i,
                "evals": run.total_evals,
                "best": {
                    "evals": final_eval,
                    "y": final_y,
                    "x": run.best_x.tolist() if run.best_x is not None else []
                }
            })
        scenarios.append(scenario)

    data = {
        "version": "0.3.18",
        "suite": "OPOCH_BBOB",
        "function_id": func_id,
        "function_name": func_name,
        "maximization": False,
        "algorithm": {
            "name": algo_name,
            "info": "Δ*-closure certified branch-and-bound with PhaseProbe identification"
        },
        "attributes": ["evaluations", "raw_y"],
        "scenarios": scenarios
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent="\t")


def create_zip_archive(output_dir: Path, zip_name: str):
    """Create zip archive for IOHanalyzer upload."""
    zip_path = output_dir.parent / f'{zip_name}.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file in output_dir.rglob('*'):
            if file.is_file():
                zf.write(file, file.relative_to(output_dir.parent))
    return zip_path


# ==============================================================================
# Main Benchmark
# ==============================================================================

def run_benchmark(
    dimensions: List[int] = [2, 10, 20],
    n_instances: int = 5,
    max_evals: int = 100000,
    max_time: float = 300.0,
    functions: Optional[List[int]] = None,
    output_dir: Optional[Path] = None,
    verbose: bool = True
) -> Dict[str, List[RunResult]]:
    """
    Run full BBOB benchmark suite.

    Args:
        dimensions: List of dimensions to test (default: [2, 10, 20])
        n_instances: Number of random instances per (function, dim) pair
        max_evals: Maximum function evaluations per run (default: 100k)
        max_time: Maximum time per run in seconds
        functions: List of BBOB function IDs to test (default: all)
        output_dir: Output directory for IOH data
        verbose: Print progress

    Returns:
        Dictionary mapping function names to list of RunResults
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results" / "ioh_data" / "OPOCH_BBOB"
    output_dir.mkdir(parents=True, exist_ok=True)

    if functions is None:
        functions = list(BBOB_FUNCTIONS.keys())

    all_results = {}

    if verbose:
        print("=" * 80)
        print("OPOCH IOHprofiler BBOB Benchmark")
        print(f"Dimensions: {dimensions}")
        print(f"Instances: {n_instances}")
        print(f"Max evals: {max_evals}")
        print("=" * 80)

    for func_id in functions:
        func_info = BBOB_FUNCTIONS[func_id]
        func_name = func_info['name']
        builder = func_info['builder']
        bound_range = func_info['bounds']
        optimum_offset = func_info.get('optimum_offset', 0.0)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Function {func_id}: {func_name}")
            print(f"{'='*60}")

        runs_by_dim = {}
        func_results = []

        for dim in dimensions:
            runs_by_dim[dim] = []
            trajectories = []

            for inst in range(1, n_instances + 1):
                # Deterministic seed for reproducibility
                np.random.seed(func_id * 10000 + dim * 100 + inst)

                # Random shift within bounds (ensure optimum is in bounds)
                margin = 0.2 * (bound_range[1] - bound_range[0])
                shift = np.random.uniform(
                    bound_range[0] + margin - optimum_offset,
                    bound_range[1] - margin - optimum_offset,
                    dim
                )

                graph = builder(dim, shift)
                bounds = [(bound_range[0], bound_range[1])] * dim
                tracker = EvalTracker()

                result = run_single_problem(
                    graph, bounds,
                    f"{func_name}-{dim}D-i{inst}",
                    tracker,
                    max_time=max_time,
                    max_evals=max_evals,
                    shift=shift,
                    optimum_offset=optimum_offset
                )

                trajectories.append(result.trajectory)
                runs_by_dim[dim].append(result)
                func_results.append(result)

            # Write .dat file for this dimension
            data_dir = output_dir / f"data_f{func_id}_{func_name}"
            data_dir.mkdir(exist_ok=True)
            write_dat_file(
                data_dir / f"IOHprofiler_f{func_id}_DIM{dim}.dat",
                trajectories
            )

            # Print summary for this dimension
            if verbose:
                cert = sum(1 for r in runs_by_dim[dim] if r.certified)
                avg_evals = np.mean([r.total_evals for r in runs_by_dim[dim]])
                avg_time = np.mean([r.time for r in runs_by_dim[dim]])
                errors = [r.error_from_optimum for r in runs_by_dim[dim]
                         if r.error_from_optimum is not None]
                avg_err = np.mean(errors) if errors else float('inf')
                print(f"  {dim}D: {cert}/{n_instances} cert | "
                      f"evals={avg_evals:.0f} | time={avg_time:.3f}s | err={avg_err:.2e}")

        # Write .json file for this function
        write_json_file(
            output_dir / f"IOHprofiler_f{func_id}_{func_name}.json",
            func_name, func_id, runs_by_dim, "OPOCH"
        )

        all_results[func_name] = func_results

    # Create zip archive
    zip_path = create_zip_archive(output_dir, "OPOCH_BBOB")

    if verbose:
        print("\n" + "=" * 80)
        print("BENCHMARK COMPLETE")
        print("=" * 80)

        total_runs = sum(len(runs) for runs in all_results.values())
        total_certified = sum(1 for runs in all_results.values()
                             for r in runs if r.certified)

        print(f"\nTotal: {total_certified}/{total_runs} certified "
              f"({100*total_certified/total_runs:.1f}%)")
        print(f"\nOutput: {output_dir}")
        print(f"Zip for IOHanalyzer: {zip_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run OPOCH BBOB benchmark for IOHanalyzer"
    )
    parser.add_argument(
        '--dimensions', '-d',
        type=int, nargs='+',
        default=[2, 10, 20],
        help='Dimensions to test (default: 2 10 20)'
    )
    parser.add_argument(
        '--instances', '-n',
        type=int, default=5,
        help='Number of instances per (function, dim) (default: 5)'
    )
    parser.add_argument(
        '--max-evals', '-e',
        type=int, default=100000,
        help='Maximum evaluations per run (default: 100000)'
    )
    parser.add_argument(
        '--max-time', '-t',
        type=float, default=300.0,
        help='Maximum time per run in seconds (default: 300)'
    )
    parser.add_argument(
        '--functions', '-f',
        type=int, nargs='+',
        default=None,
        help='BBOB function IDs to test (default: all)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path, default=None,
        help='Output directory'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress output'
    )

    args = parser.parse_args()

    run_benchmark(
        dimensions=args.dimensions,
        n_instances=args.instances,
        max_evals=args.max_evals,
        max_time=args.max_time,
        functions=args.functions,
        output_dir=args.output,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
