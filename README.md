# OPOCH Optimizer

Deterministic global nonlinear optimization with auditable outputs:
**(1) globally optimal**, **(2) infeasible**, or **(3) exact remaining gap**.
Every run is replayable from receipts.

## Quick start

```bash
pip install -e .
python -m opoch_optimizer.cli solve data/toy/rosenbrock_2d.json --epsilon 1e-6 --time_limit 30
python scripts/replay_verify_run.py runs/rosenbrock_2d/
```

## What you get (no guessing)

For each instance, OPOCH returns exactly one:

| Verdict | Meaning |
|---------|---------|
| **UNIQUE-OPT** | x*, UB, LB, and UB-LB ≤ ε (globally optimal within tolerance) |
| **UNSAT** | Infeasibility certificate (no feasible point exists) |
| **Ω-GAP** | UB, LB, exact gap, and the next best separator action |

## Why this matters

Most optimizers return "best found." OPOCH returns either:
- a **proof** that no better solution exists (within ε), or
- a **proof** that the instance is infeasible, or
- a **precise, quantified statement** of what remains unknown.

## Mathematical Foundation

OPOCH implements a complete branch-and-reduce algorithm with:

1. **Certified Lower Bounds** via interval arithmetic, McCormick relaxations, and FBBT
2. **Certified Upper Bounds** via feasibility verification of candidate solutions
3. **Gap Closure** as the termination criterion: UB - LB ≤ ε is mathematical proof

The witness lattice (Δ*) treats bounds, tightenings, splits, and primal searches as acts with explicit costs, enabling optimal resource allocation.

See [docs/ANCHOR_KERNEL.md](docs/ANCHOR_KERNEL.md) for the complete mathematical specification.

## Reproducibility

A run folder contains:
- Canonical input specification
- Event log (splits/tightenings/prunes/incumbents)
- Certificates for each bound computation
- SHA-256 receipt chain for tamper detection
- Replay verifier that recomputes everything

```bash
# Verify any recorded run
python scripts/replay_verify_run.py results/ioh/run_001/
```

See [docs/CERTIFICATES.md](docs/CERTIFICATES.md) for certificate formats.

## Benchmarks

We support IOHprofiler BBOB logging and produce zips suitable for IOHanalyzer upload.

```bash
# Run BBOB suite with IOH logging
python scripts/run_ioh_bbob.py --dims 2 10 20 --budget 100000 --out results/ioh/OPOCH_BBOB

# Generate comparison plots
python scripts/make_report.py --input results/ioh/OPOCH_BBOB --output results/reports/
```

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) and [docs/IOH_INTEGRATION.md](docs/IOH_INTEGRATION.md).

## Installation

```bash
# From source
git clone https://github.com/yourusername/opoch-optimizer.git
cd opoch-optimizer
pip install -e .

# Run tests
pytest -v
```

### Requirements
- Python >= 3.8
- NumPy >= 1.21
- SciPy >= 1.7

## Project Structure

```
opoch-optimizer/
├── data/               # Benchmark instances
├── docs/               # Mathematical specification and guides
├── results/            # Generated outputs (IOH logs, reports)
├── scripts/            # Benchmark runners and utilities
├── src/opoch_optimizer/
│   ├── bounds/         # Tier 0-3 bound computations
│   ├── primal/         # UB discovery (Sobol, local, PhaseProbe)
│   ├── solver/         # Branch-and-reduce engine
│   └── verify/         # Replay and certificate verification
└── tests/              # Unit and integration tests
```

## Citation

If you use this software, please cite:

```bibtex
@software{opoch_optimizer,
  title = {OPOCH Optimizer},
  author = {OPOCH Team},
  year = {2025},
  url = {https://github.com/yourusername/opoch-optimizer}
}
```

See [CITATION.cff](CITATION.cff) for the full citation.

## Contributing

We welcome:
- Bug reports with reproducible instances
- New benchmark problems (especially "hard" cases)
- Performance improvements that preserve determinism

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License. See [LICENSE](LICENSE).
