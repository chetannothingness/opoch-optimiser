# ANCHOR_KERNEL — OPOCH Nonlinear Optimization (Complete Mathematical Specification)

This document is the **source of truth** for the OPOCH optimizer. All implementation decisions derive from these equations.

## 0) Output Gate (Absolute)

Given:
- Bounded domain X ⊂ ℝⁿ
- Objective function f: X → ℝ
- Inequality constraints gᵢ(x) ≤ 0, i = 1,...,m
- Equality constraints hⱼ(x) = 0, j = 1,...,p
- Tolerance ε > 0

**Output must be exactly one of:**

| Verdict | Contents | Meaning |
|---------|----------|---------|
| **UNIQUE-OPT** | x*, UB=f(x*), LB, certificate | UB − LB ≤ ε (globally optimal within tolerance) |
| **UNSAT** | Refutation certificate | F = ∅ (no feasible point exists) |
| **Ω-GAP** | UB, LB, gap, next_act | Budget exhausted with exact remaining gap |

**No other output is permitted.**

## 1) The Only Truth Object: The UB–LB Gap

Let F = {x ∈ X : g(x) ≤ 0, h(x) = 0} be the feasible set.

Let f* = inf_{x∈F} f(x) be the global optimum.

**Theorem (Separation Equivalence):**
Any global claim "x* is ε-optimal" is equivalent to a separation witness:
```
∀x ∈ F: f(x) ≥ f(x*) − ε
```

This is equivalent to:
```
UB − LB ≤ ε where UB = f(x*), LB ≤ f*
```

**Corollary:** Any truthful global solver must maintain:
- **UB** from a verified feasible point
- **LB** as a certified lower bound on f*

The problem is **solved** if and only if UB − LB ≤ ε.

## 2) Forced Normal Form: Region Queue + Certified Bounds

**Partition scheme:**
Partition X into regions R₁, R₂, ..., Rₖ (axis-aligned boxes).

For each region R, maintain:
- **Feasibility status:** EMPTY (proven infeasible) or MAYBE (possibly contains optimum)
- **Certified lower bound:** LB(R) ≤ inf_{x ∈ F ∩ R} f(x)

**Global lower bound:**
```
LB = min_{R: status(R) = MAYBE} LB(R)
```

**Pruning rule:**
Prune any region R where LB(R) ≥ UB − ε (cannot contain ε-optimal solution).

**Termination:**
Stop when UB − LB ≤ ε (gap closed) or all regions are EMPTY (infeasible).

This is the **minimal no-slack form** of global optimization.

## 3) Witness Lattice (Δ*): Bounds and Tightenings are Tests

OPOCH treats all solver operations as **acts** with explicit costs:
- Compute a bound → returns (LB, certificate)
- Tighten a box → returns (new_bounds, certificate)
- Split a region → returns (child₁, child₂)
- Find a feasible point → returns (x, f(x)) or FAIL

**Witness ladder (increasing tightness and cost):**

| Tier | Name | Description | Cost |
|------|------|-------------|------|
| 0 | Interval | Natural interval extension | O(n) |
| 1 | McCormick | Convex/concave relaxations on DAG | O(n²) |
| 2a | FBBT | Feasibility-based bound tightening | O(n·k) iterations |
| 2b | OBBT | Optimization-based bound tightening | O(n) LPs |
| 3 | SOS | Sum-of-squares relaxations | O(n^d) SDP |

Each act returns a **certificate object** containing:
- The computed value (bound, new region, etc.)
- Sufficient data to replay/verify the computation
- A SHA-256 hash for tamper detection

## 4) Deterministic Control Law (Tie-Safe)

At each solver step, consider all candidate acts:
- SPLIT(R): Split region R on longest dimension
- TIGHTEN(R, tier): Upgrade R's bound to higher tier
- PROPAGATE(R): Apply FBBT/OBBT to R
- PRIMAL(R): Search for feasible point in R

**Act selection rule:**
```
act* = argmin_{a ∈ candidates} [cost(a) + max_gap_after(a)]
```

where `max_gap_after(a)` is the worst-case remaining gap after executing act a.

**Tie-breaking:**
When scores are equal, break ties using:
1. Canonical region fingerprint (SHA-256 of bounds)
2. Act type ordering: PRUNE < SPLIT < TIGHTEN < PROPAGATE < PRIMAL
3. Region ID (lexicographic)

This ensures **identical execution across runs**.

## 5) Equality Constraints: Closure Operators, Not Checks

Equality constraints h(x) = 0 define curved manifolds in ℝⁿ.

**FBBT (Feasibility-Based Bound Tightening):**

Given h(x) = 0 as an expression DAG and box bounds [l, u]:

1. **Forward pass:** Compute interval bounds for each DAG node bottom-up
2. **Backward pass:** From h(x) ∈ [0, 0], propagate constraints back to variables
3. **Iterate:** Repeat until bounds stabilize (fixed point) or region is refuted

**Mathematical guarantee:**
```
FBBT([l,u]) = [l', u'] where:
- F ∩ [l,u] ⊆ F ∩ [l',u']  (no feasible point lost)
- [l',u'] ⊆ [l,u]          (tightening)
- [l',u'] is the least fixed point of interval propagation
```

**Empty detection:**
If forward pass shows 0 ∉ h([l,u]), region is EMPTY (infeasible).

**Why FBBT is mandatory:**
Without FBBT, manifold constraints remain "fat" and proofs stall. The equality h(x) = 0 generates tightenings that pure interval bounds cannot.

## 6) Interval Newton Contraction

For equality constraints h(x) = 0, Interval Newton provides tighter contraction than FBBT alone.

**Algorithm:**
Given h(x) = 0 and box [l, u]:

1. Compute midpoint x_mid = (l + u) / 2
2. Evaluate h(x_mid)
3. Compute interval Jacobian ∂h/∂xᵢ over [l, u]
4. Apply Newton update: xᵢ_new = x_mid - h(x_mid) / (∂h/∂xᵢ)
5. Intersect with current bounds
6. If intersection empty → EMPTY certificate

**Iteration:** Continue until convergence or max iterations.

## 7) Structured Multimodal Families: Test Synthesis

For shifted periodic families (e.g., shifted Rastrigin), the latent shift is a missing distinguisher.

**Rastrigin function:**
```
f(x) = 10n + Σᵢ[(xᵢ - sᵢ)² - 10·cos(2π(xᵢ - sᵢ))]
```

where s is the (unknown) shift vector.

**PhaseProbe algorithm:**
1. Sample f along coordinate axes using deterministic grid
2. Extract dominant frequency via DFT
3. Compute phase φᵢ = arg(DFT peak) for each dimension
4. Infer shift: sᵢ = -φᵢ / (2π·frequency)
5. Seed primal search at inferred optimum

**Complexity:** O(d · M) evaluations where M is samples per dimension.

**Mathematical justification:**
For periodic f(x) = g(x - s) where g is known, the shift s is identifiable from Fourier phase. This is a derived test in Δ*, not an ad-hoc heuristic.

## 8) Certificates

### UNIQUE-OPT Certificate
Contains:
- Feasible incumbent x* with f(x*) = UB
- Global LB = min_R LB(R) over all active regions
- ε tolerance
- Proof that UB - LB ≤ ε

**Verification:**
1. Check feasibility of x* (all constraints satisfied)
2. Verify each region's LB certificate
3. Confirm UB - LB ≤ ε

### UNSAT Certificate
Contains:
- Refutation cover: partition of X into EMPTY regions
- Each region has an infeasibility certificate (e.g., FBBT refutation)

**Verification:**
1. Check cover completeness (union = X)
2. Re-run EMPTY proofs for each region
3. Confirm all regions are EMPTY

### Ω-GAP Certificate
Contains:
- Current UB (may be ∞ if no feasible point found)
- Current LB (may be -∞ if not yet computed)
- Exact gap = UB - LB
- Next separator act proposal

**Verification:**
1. Verify LB is valid (replay bound certificates)
2. Verify UB is feasible (if finite)
3. Confirm gap calculation

## 9) Receipts and Replay

Every solver event is recorded as:
```json
{
  "seq": 42,
  "action": "SPLIT",
  "region_id": 7,
  "params": {"dimension": 0, "point": 0.5},
  "input_hash": "abc123...",
  "output_hash": "def456...",
  "prev_hash": "...",
  "receipt_hash": "sha256(canonical(this))"
}
```

**Receipt chain:**
Each receipt's hash depends on the previous receipt, forming a Merkle-like chain.

**Replay verification:**
```python
for receipt in chain:
    recomputed = execute(receipt.action, receipt.params)
    assert hash(recomputed) == receipt.output_hash
```

This guarantees:
- Identical inputs produce identical outputs
- Any tampering is detected
- Results are reproducible across machines

## 10) Complexity and Convergence

**Convergence guarantee:**
OPOCH terminates with UNIQUE-OPT or UNSAT for any bounded box-constrained problem with Lipschitz continuous objective and constraints.

**Complexity:**
- Worst case: exponential in dimension (unavoidable for NP-hard problems)
- Typical: polynomial for "well-behaved" problems with good relaxations

**Budget exhaustion:**
If computational budget is exceeded before gap closure, return Ω-GAP with:
- Current best UB and LB
- Exact remaining gap
- The next act that would most reduce the gap

This is honest reporting, not failure.
