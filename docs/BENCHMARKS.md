# Benchmarks and Scoring

This document defines the benchmark suites and scoring methodology.

## Two Leaderboards

### A) Performance Curves (IOH-Compatible)

**Metric:** Best-so-far vs evaluations

**Comparison:** All optimizers (heuristics + certified)

**What it measures:** How quickly do you find good solutions?

### B) Truth Curves (OPOCH-Only)

**Metric:** Certification rate and evaluations-to-certify distribution

**Comparison:** Only certified solvers

**What it measures:** How quickly do you prove optimality/infeasibility?

## Benchmark Suites

### 1. BBOB (Black-Box Optimization Benchmarking)

| Property | Value |
|----------|-------|
| Functions | 24 (f1-f24) |
| Dimensions | 2, 10, 20 |
| Instances | 1-5 per function |
| Budget | 100,000 evaluations |
| Known optima | Yes (target-based) |

**Run command:**
```bash
python scripts/run_ioh_bbob.py --dims 2 10 20 --budget 100000 --out results/ioh/OPOCH_BBOB
```

### 2. Toy Problems (Quick Validation)

| Problem | Dimension | Optimum | Constraints |
|---------|-----------|---------|-------------|
| Sphere | 2 | 0 at origin | Bounds only |
| Rosenbrock | 2 | 0 at (1,1) | Bounds only |
| Constrained Quadratic | 2 | 0.5 at (0.5,0.5) | x + y = 1 |
| Infeasible Box | 1 | UNSAT | x ≤ 0 AND x ≥ 1 |

**Run command:**
```bash
python scripts/run_suite.py --suite data/toy --out results/reports/toy
```

### 3. GLOBALLib (Constrained NLP)

| Property | Value |
|----------|-------|
| Problems | 50 core problems |
| Dimensions | 1-20 |
| Constraints | Equalities + inequalities |
| Budget | 60 seconds per problem |

**Run command:**
```bash
python scripts/run_globallib.py --timeout 60 --epsilon 1e-4 --out results/reports/globallib
```

## Scoring Rules

### Performance Score (Best-Found)

For each problem instance:
1. Record best f(x) found at each evaluation count
2. Compare to known optimum f*
3. Compute error: |f(x) - f*|

**Aggregation:**
- ECDF over problem set
- Geometric mean of errors at fixed budget

### Certification Score

For each problem instance:
1. Record evaluations until UB - LB ≤ ε (certification)
2. If certified: score = evals_to_certify
3. If not certified: score = ∞ (budget exhausted)

**Aggregation:**
- Certification rate: % problems certified
- Mean evaluations-to-certify (certified problems only)
- ECDF of certification times

### Gap Score (Ω-GAP Cases)

For problems not certified within budget:
1. Record final gap = UB - LB
2. Compare to initial gap (before any search)
3. Gap reduction ratio = (initial_gap - final_gap) / initial_gap

## Fair Comparison Rules

1. **Same budget:** All optimizers get identical evaluation budget
2. **Same instances:** All optimizers run on identical problem instances
3. **No tuning on test set:** Hyperparameters fixed before benchmark
4. **Deterministic replay:** Results must be reproducible

## Reporting Format

### Table Format

```
| Problem | Dim | Best Found | Optimum | Error | Verdict | Evals |
|---------|-----|------------|---------|-------|---------|-------|
| f1      | 2   | -418.98    | -418.98 | 1e-7  | UNIQUE  | 4523  |
| f2      | 2   | 1.23       | 0.0     | 1.23  | Ω-GAP   | 100k  |
```

### Summary Statistics

```
Certification Rate: 85% (51/60 problems)
Mean Evals-to-Certify: 12,345
Geometric Mean Error: 1.23e-4
```

### Plots

1. **ECDF curve:** x = evaluations, y = fraction certified
2. **Scatter plot:** x = dimension, y = evals-to-certify
3. **Gap reduction:** x = initial gap, y = final gap

## Generating Reports

```bash
# Generate scoreboard and plots
python scripts/make_report.py \
    --input results/ioh/OPOCH_BBOB \
    --output results/reports/ \
    --format md

# Output files:
# - results/reports/scoreboard.csv
# - results/reports/scoreboard.md
# - results/reports/ecdf.png
# - results/reports/summary.json
```

## Reproducing Published Results

```bash
# Run full benchmark suite
python scripts/run_ioh_bbob.py --dims 2 10 20 --budget 100000

# Verify results
python scripts/replay_verify_run.py results/ioh/OPOCH_BBOB/

# Generate comparison report
python scripts/make_report.py --input results/ioh/OPOCH_BBOB
```

All results should match the published values within floating-point tolerance.
