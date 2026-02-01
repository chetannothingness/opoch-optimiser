"""
IOH-Analyzer Compatible Logger

Generates output format compatible with IOH-analyzer:
https://iohanalyzer.liacs.nl/

Output structure:
  results/
    IOHprofiler_f{fid}_DIM{dim}.info
    data_f{fid}/
      IOHprofiler_f{fid}_DIM{dim}.dat
"""

import os
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
import hashlib


@dataclass
class RunData:
    """Data for a single optimization run."""
    function_id: int
    instance: int
    dimension: int
    algorithm: str
    trajectory: List[Tuple[int, float]]  # (evaluations, f_best)
    final_f: float
    final_x: Optional[np.ndarray]
    evaluations: int
    target_hit: bool
    receipt_hash: str


@dataclass
class IOHLogger:
    """
    Logger that outputs IOH-analyzer compatible format.

    Creates .info files (metadata) and .dat files (trajectories).
    """
    output_dir: str
    algorithm_name: str = "OPOCH-DCMA"
    algorithm_info: str = "Deterministic CMA with quadratic identification"

    # Internal storage
    runs: Dict[Tuple[int, int], List[RunData]] = field(default_factory=dict)

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def log_run(self, run_data: RunData):
        """Log a single optimization run."""
        key = (run_data.function_id, run_data.dimension)
        if key not in self.runs:
            self.runs[key] = []
        self.runs[key].append(run_data)

    def write_all(self):
        """Write all logged data to files."""
        for (fid, dim), runs in self.runs.items():
            self._write_function_data(fid, dim, runs)

    def _write_function_data(self, fid: int, dim: int, runs: List[RunData]):
        """Write data for a single function/dimension combination."""
        # Create data directory
        data_dir = os.path.join(self.output_dir, f"data_f{fid}")
        os.makedirs(data_dir, exist_ok=True)

        # Write .dat file
        dat_filename = f"IOHprofiler_f{fid}_DIM{dim}.dat"
        dat_path = os.path.join(data_dir, dat_filename)

        with open(dat_path, 'w') as f:
            for run_idx, run in enumerate(runs):
                # Header for this run
                f.write(f"\"function evaluation\" \"best-so-far f(x)\"\n")

                # Trajectory data
                for evals, f_best in run.trajectory:
                    f.write(f"{evals} {f_best:.15e}\n")

                # Final point if not in trajectory
                if run.trajectory and run.trajectory[-1][0] != run.evaluations:
                    f.write(f"{run.evaluations} {run.final_f:.15e}\n")

                # Separator between runs
                if run_idx < len(runs) - 1:
                    f.write("\n")

        # Write .info file
        info_filename = f"IOHprofiler_f{fid}_DIM{dim}.info"
        info_path = os.path.join(self.output_dir, info_filename)

        with open(info_path, 'w') as f:
            f.write(f"suite = \"BBOB\", funcId = {fid}, DIM = {dim}, algId = \"{self.algorithm_name}\"\n")
            f.write(f"%\n")
            f.write(f"data_f{fid}/{dat_filename}, ")

            # Run info: instance:evaluations|final_f
            run_infos = []
            for run in runs:
                run_infos.append(f"{run.instance}:{run.evaluations}|{run.final_f:.15e}")
            f.write(", ".join(run_infos))
            f.write("\n")

    def write_meta(self):
        """Write metadata JSON file."""
        meta = {
            "algorithm": self.algorithm_name,
            "info": self.algorithm_info,
            "timestamp": datetime.now().isoformat(),
            "functions": {},
            "receipt_chain": []
        }

        for (fid, dim), runs in self.runs.items():
            key = f"f{fid}_d{dim}"
            meta["functions"][key] = {
                "function_id": fid,
                "dimension": dim,
                "num_runs": len(runs),
                "instances": [r.instance for r in runs],
                "targets_hit": sum(1 for r in runs if r.target_hit),
                "best_f": min(r.final_f for r in runs),
                "total_evaluations": sum(r.evaluations for r in runs)
            }
            meta["receipt_chain"].extend([r.receipt_hash for r in runs])

        # Compute overall receipt hash
        meta["overall_receipt"] = hashlib.sha256(
            ":".join(meta["receipt_chain"]).encode()
        ).hexdigest()

        meta_path = os.path.join(self.output_dir, "meta.json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)


class IOHExperimentLogger:
    """
    High-level experiment logger for complete benchmark suites.

    Manages multiple functions, dimensions, and instances.
    """

    def __init__(
        self,
        output_dir: str,
        algorithm_name: str = "OPOCH-DCMA",
        suite: str = "BBOB"
    ):
        self.output_dir = output_dir
        self.algorithm_name = algorithm_name
        self.suite = suite

        self.logger = IOHLogger(
            output_dir=output_dir,
            algorithm_name=algorithm_name
        )

        # Summary statistics
        self.summary = {
            "total_runs": 0,
            "targets_hit": 0,
            "by_function": {},
            "by_dimension": {}
        }

    def log_run(
        self,
        function_id: int,
        instance: int,
        dimension: int,
        trajectory: List[Tuple[int, float]],
        final_f: float,
        final_x: Optional[np.ndarray],
        evaluations: int,
        target: float = 1e-8,
        receipt_hash: str = ""
    ):
        """Log a single optimization run."""
        target_hit = final_f <= target

        run_data = RunData(
            function_id=function_id,
            instance=instance,
            dimension=dimension,
            algorithm=self.algorithm_name,
            trajectory=trajectory,
            final_f=final_f,
            final_x=final_x,
            evaluations=evaluations,
            target_hit=target_hit,
            receipt_hash=receipt_hash
        )

        self.logger.log_run(run_data)

        # Update summary
        self.summary["total_runs"] += 1
        if target_hit:
            self.summary["targets_hit"] += 1

        fkey = f"f{function_id}"
        if fkey not in self.summary["by_function"]:
            self.summary["by_function"][fkey] = {"runs": 0, "hits": 0}
        self.summary["by_function"][fkey]["runs"] += 1
        if target_hit:
            self.summary["by_function"][fkey]["hits"] += 1

        dkey = f"d{dimension}"
        if dkey not in self.summary["by_dimension"]:
            self.summary["by_dimension"][dkey] = {"runs": 0, "hits": 0}
        self.summary["by_dimension"][dkey]["runs"] += 1
        if target_hit:
            self.summary["by_dimension"][dkey]["hits"] += 1

    def finalize(self):
        """Write all data and summary."""
        self.logger.write_all()
        self.logger.write_meta()

        # Write summary
        summary_path = os.path.join(self.output_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(self.summary, f, indent=2)

        # Print summary
        print(f"\n{'='*60}")
        print(f"BENCHMARK SUMMARY: {self.algorithm_name}")
        print(f"{'='*60}")
        print(f"Total runs: {self.summary['total_runs']}")
        print(f"Targets hit: {self.summary['targets_hit']}")
        print(f"Success rate: {100*self.summary['targets_hit']/max(1,self.summary['total_runs']):.1f}%")
        print()
        print("By function:")
        for fkey, data in sorted(self.summary["by_function"].items()):
            rate = 100 * data["hits"] / max(1, data["runs"])
            print(f"  {fkey}: {data['hits']}/{data['runs']} ({rate:.0f}%)")
        print()
        print("By dimension:")
        for dkey, data in sorted(self.summary["by_dimension"].items(), key=lambda x: int(x[0][1:])):
            rate = 100 * data["hits"] / max(1, data["runs"])
            print(f"  {dkey}: {data['hits']}/{data['runs']} ({rate:.0f}%)")
        print(f"{'='*60}\n")

        return self.summary


def create_trajectory_from_history(
    history: List[Tuple[np.ndarray, float]],
    target: float = 1e-8
) -> List[Tuple[int, float]]:
    """
    Convert (x, f) history to (evaluations, best_f) trajectory.

    Only records improvements.
    """
    trajectory = []
    best_f = float('inf')

    for i, (x, f) in enumerate(history):
        if f < best_f:
            best_f = f
            trajectory.append((i + 1, best_f))
            if best_f <= target:
                break

    return trajectory
