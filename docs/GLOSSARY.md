# Glossary — Terms and Symbols

## Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| X | Bounded domain (box) ⊂ ℝⁿ |
| f | Objective function f: X → ℝ |
| gᵢ | Inequality constraint gᵢ(x) ≤ 0 |
| hⱼ | Equality constraint hⱼ(x) = 0 |
| F | Feasible set {x ∈ X : g(x) ≤ 0, h(x) = 0} |
| f* | Global optimum inf_{x∈F} f(x) |
| ε | Optimality tolerance |
| UB | Upper bound (from feasible point) |
| LB | Lower bound (from relaxation) |
| R | Region (axis-aligned box) |
| [a, b] | Interval from a to b |
| [f]([x]) | Interval extension of f over [x] |

## Verdicts

| Term | Definition |
|------|------------|
| **UNIQUE-OPT** | Globally optimal within tolerance: UB - LB ≤ ε |
| **UNSAT** | Infeasible: no point satisfies all constraints |
| **Ω-GAP** | Budget exhausted with exact remaining gap |

## Bound Types

| Term | Definition |
|------|------------|
| **Interval bound** | Natural interval extension over box (Tier 0) |
| **McCormick bound** | Convex/concave relaxation on DAG (Tier 1) |
| **FBBT** | Feasibility-based bound tightening (Tier 2a) |
| **OBBT** | Optimization-based bound tightening (Tier 2b) |
| **SOS** | Sum-of-squares relaxation (Tier 3) |

## Solver Components

| Term | Definition |
|------|------------|
| **Act** | Solver action (split, tighten, propagate, primal) |
| **Region** | Axis-aligned box being explored |
| **Witness** | Certificate for a bound or refutation |
| **Δ* (Delta-star)** | Witness lattice: set of all available tests |
| **PhaseProbe** | Structure detector for shifted periodic families |

## Certificates

| Term | Definition |
|------|------------|
| **Certificate** | Proof object for a solver claim |
| **Receipt** | Logged event with hash for replay |
| **Receipt chain** | Sequence of receipts with hash linking |
| **Refutation** | Proof that a region is infeasible |
| **Cover** | Partition of X into refuted regions |

## DAG (Expression Graph)

| Term | Definition |
|------|------------|
| **Node** | Element in expression DAG |
| **Variable** | Input node xᵢ |
| **Constant** | Fixed value node |
| **UnaryOp** | Single-input operation (neg, square, sqrt, exp, log, sin, cos) |
| **BinaryOp** | Two-input operation (add, sub, mul, div, pow) |
| **Topological order** | Nodes ordered so parents come before children |

## Interval Arithmetic

| Term | Definition |
|------|------------|
| **Interval** | Pair [a, b] representing all values in [a, b] |
| **Outward rounding** | Round down for lower, up for upper bounds |
| **Empty interval** | [a, b] where a > b (no values) |
| **Contains zero** | 0 ∈ [a, b] iff a ≤ 0 ≤ b |
| **Width** | b - a for interval [a, b] |

## FBBT

| Term | Definition |
|------|------------|
| **Forward pass** | Compute intervals bottom-up through DAG |
| **Backward pass** | Propagate constraints top-down through DAG |
| **Fixed point** | Bounds that don't change under iteration |
| **Tightening** | Reducing interval width |

## Primal Search

| Term | Definition |
|------|------------|
| **Primal** | Search for feasible points (UB discovery) |
| **Incumbent** | Current best feasible solution |
| **Sobol sequence** | Low-discrepancy deterministic sampling |
| **Local refinement** | L-BFGS-B from a starting point |
| **Multi-start** | Multiple local searches from different seeds |

## Reproducibility

| Term | Definition |
|------|------------|
| **Deterministic** | Same inputs → same outputs, always |
| **Tie-safe** | Deterministic tie-breaking via canonical fingerprints |
| **Fingerprint** | SHA-256 hash of canonical representation |
| **Canonical JSON** | Sorted keys, no whitespace, deterministic |
| **Replay** | Re-execute from receipts and verify |

## IOH Integration

| Term | Definition |
|------|------------|
| **IOHprofiler** | Benchmarking framework for iterative optimization |
| **BBOB** | Black-Box Optimization Benchmarking suite |
| **ECDF** | Empirical cumulative distribution function |
| **Fixed-target** | Evaluations to reach target value |
| **Fixed-budget** | Best value at fixed evaluation count |
| **Anytime** | Algorithm that can be stopped at any point |

## Complexity

| Term | Definition |
|------|------------|
| **n** | Number of variables (dimension) |
| **m** | Number of inequality constraints |
| **p** | Number of equality constraints |
| **k** | Number of DAG nodes |
| **Lipschitz** | Bounded rate of change |
| **Factorable** | Expressible as DAG of standard operations |
