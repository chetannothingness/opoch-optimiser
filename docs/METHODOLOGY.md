# METHODOLOGY — Math → Code Mapping

This document bridges the mathematical specification in [ANCHOR_KERNEL.md](ANCHOR_KERNEL.md) to the implementation in `src/opoch_optimizer/`.

## Expression Graph

**Math:** Factorable function f: ℝⁿ → ℝ represented as directed acyclic graph (DAG)

**Code:** `src/opoch_optimizer/expr_graph.py`

| Concept | Class/Function |
|---------|----------------|
| Variable node xᵢ | `Variable(var_index, name)` |
| Constant node c | `Constant(value)` |
| Unary operation f(x) | `UnaryOp(op, child)` |
| Binary operation f(x,y) | `BinaryOp(op, left, right)` |
| DAG container | `ExpressionGraph` |
| Point evaluation | `ExpressionGraph.evaluate(x)` |
| Topological traversal | `ExpressionGraph.topological_order()` |

## Interval Bounds (Tier 0)

**Math:** Natural interval extension [f]([x]) ⊇ {f(x) : x ∈ [x]}

**Code:** `src/opoch_optimizer/bounds/interval.py`

| Concept | Class/Function |
|---------|----------------|
| Interval [a, b] | `Interval(lo, hi)` |
| Interval arithmetic | `Interval.__add__`, `__mul__`, etc. |
| DAG interval evaluation | `IntervalEvaluator.evaluate(var_intervals)` |
| Refutation (0 ∉ [h]) | `Interval.contains(0)` returning False |
| Outward rounding | `ROUND_EPS = 1e-15` |

## McCormick Relaxations (Tier 1)

**Math:** Convex underestimator cv(x) ≤ f(x) ≤ cc(x) concave overestimator

**Code:** `src/opoch_optimizer/bounds/mccormick.py`

| Concept | Class/Function |
|---------|----------------|
| McCormick bounds | `McCormickBounds(cv, cc)` |
| Bilinear envelope | `_mccormick_mul(x_bounds, y_bounds)` |
| LP relaxation | `RelaxationLP.solve()` |
| Lower bound computation | `McCormickRelaxation.compute_lower_bound()` |

## FBBT for Equalities (Tier 2a)

**Math:** Least fixed point of forward/backward interval propagation

**Code:** `src/opoch_optimizer/bounds/fbbt.py`

| Concept | Class/Function |
|---------|----------------|
| Forward pass | `FBBTOperator._forward_pass()` |
| Backward pass | `FBBTOperator._backward_pass()` |
| Fixed-point iteration | `FBBTOperator.tighten()` |
| Inequality handling | `FBBTInequalityOperator` |
| Multi-constraint FBBT | `apply_fbbt_all_constraints()` |

## Interval Newton (Tier 2a+)

**Math:** xᵢ_new = x_mid - h(x_mid) / [∂h/∂xᵢ]

**Code:** `src/opoch_optimizer/bounds/interval_newton.py`

| Concept | Class/Function |
|---------|----------------|
| Derivative graphs | `IntervalNewtonOperator._derivative_graphs` |
| Newton contraction | `IntervalNewtonOperator.contract()` |
| Empty detection | `IntervalNewtonStatus.EMPTY` |

## Primal Search (UB Discovery)

**Math:** Find feasible x with f(x) < current UB

**Code:** `src/opoch_optimizer/primal/`

| Concept | Class/Function |
|---------|----------------|
| Sobol sampling | `sobol.py: SobolGenerator.generate()` |
| Local refinement | `local.py: LocalSolver.refine()` |
| PhaseProbe | `phaseprobe.py: PhaseProbe.identify_shift()` |
| Portfolio orchestration | `portfolio.py: PrimalPortfolio` |

## Solver Loop

**Math:** Branch-and-reduce with gap closure

**Code:** `src/opoch_optimizer/solver/`

| Concept | Class/Function |
|---------|----------------|
| Region representation | `contract.py: Region` |
| Region queue | `bnb.py: OPOCHKernel._heap` |
| Act selection | `act_select.py: select_best_act()` |
| Split operation | `split.py: RegionSplitter.split()` |
| Termination check | `bnb.py: OPOCHKernel._check_termination()` |

## Certificates

**Math:** Proof objects for UNIQUE/UNSAT/Ω

**Code:** `src/opoch_optimizer/solver/certificates.py`

| Concept | Class/Function |
|---------|----------------|
| Optimality certificate | `OptimalityCertificate` |
| Infeasibility certificate | `UnsatCertificate` |
| Gap certificate | `OmegaGapCertificate` |
| Verification | `verify_certificate()` |

## Receipts and Replay

**Math:** Canonical JSON + SHA-256 chain

**Code:** `src/opoch_optimizer/receipts.py` + `src/opoch_optimizer/verify/`

| Concept | Class/Function |
|---------|----------------|
| Receipt object | `Receipt` |
| Chain construction | `ReceiptChain.add_receipt()` |
| Canonical serialization | `canonical_json.py: canonical_dumps()` |
| Replay verification | `verify/replay.py: replay_and_verify()` |

## Tie-Safe Ordering

**Math:** Deterministic tie-breaking via canonical fingerprints

**Code:** `src/opoch_optimizer/tie_safe.py`

| Concept | Class/Function |
|---------|----------------|
| Fingerprint computation | `canonical_fingerprint()` |
| Tie-safe comparison | `TieSafeChoice.select()` |
| Deterministic ordering | `RegionState.__lt__()` |
