# FAQ — Frequently Asked Questions

Pre-emptive answers to questions from Reddit, Hacker News, and academic reviewers.

## General Questions

### "Isn't this just branch-and-bound?"

Partially. Branch-and-bound is the algorithmic skeleton, but OPOCH adds:

1. **Deterministic tie-safe control:** Identical inputs → identical outputs, always
2. **Receipt chain:** Every decision is logged and verifiable
3. **Endogenous test synthesis:** PhaseProbe identifies structure in shifted multimodal families
4. **Forced output gate:** Only UNIQUE-OPT, UNSAT, or Ω-GAP (no "best found maybe")

Standard B&B implementations often have:
- Random tie-breaking
- No replay verification
- Silent failures when bounds diverge
- "Best found" as the only output

### "Why not just use CMA-ES / Bayesian Optimization / DE?"

These are excellent optimizers for finding good solutions. They are not designed to:

1. **Prove** that the solution is globally optimal
2. **Prove** that no feasible solution exists
3. **Quantify** exactly how much gap remains

**CMA-ES output:** "Best found: 0.001"
**OPOCH output:** "UNIQUE-OPT: x* is globally optimal, gap ≤ 1e-6, certificate attached"

If you only need best-found, use CMA-ES. If you need proof, use OPOCH.

### "Does determinism hurt exploration?"

No. OPOCH uses:

1. **Low-discrepancy sequences (Sobol):** Deterministic but well-distributed
2. **PhaseProbe:** Exploits problem structure instead of random search
3. **Certified bounds:** Guide search toward promising regions

Randomness is not magic. Deterministic exploration with good coverage often outperforms random restarts.

### "How do I verify your results?"

```bash
# Clone and install
git clone https://github.com/yourusername/opoch-optimizer.git
pip install -e .

# Run benchmark
python scripts/run_ioh_bbob.py --dims 2 10 20 --budget 100000

# Verify
python scripts/replay_verify_run.py results/ioh/OPOCH_BBOB/
```

Replay recomputes every bound, prune, and incumbent update. If any hash mismatches, verification fails.

### "What problems can OPOCH solve?"

OPOCH handles:
- Box-constrained optimization (x ∈ [l, u])
- Inequality constraints (g(x) ≤ 0)
- Equality constraints (h(x) = 0)
- Factorable objectives (expressible as DAG of standard operations)

OPOCH does **not** handle:
- Discrete variables (use MINLP solvers)
- Black-box derivatives (we compute interval bounds on DAG)
- Unbounded domains (must have finite box)

### "How does PhaseProbe work?"

For shifted periodic functions like Rastrigin:
```
f(x) = 10n + Σᵢ[(xᵢ - sᵢ)² - 10·cos(2π(xᵢ - sᵢ))]
```

The shift s is a latent parameter. PhaseProbe:
1. Samples f along each coordinate axis
2. Computes DFT to extract dominant frequency
3. Reads phase from DFT peak
4. Inverts phase to recover shift

This is O(d · M) evaluations instead of O(exponential) random search.

### "Why not use automatic differentiation?"

We use **interval arithmetic**, not point derivatives. AD gives f'(x) at a point; interval arithmetic gives bounds on f over a region.

Interval bounds are essential for:
- Proving no solution exists in a region
- Computing certified lower bounds
- Tightening variable bounds (FBBT)

We do use AD-like techniques (forward/backward propagation) on the expression DAG, but with intervals.

## Technical Questions

### "What's the complexity?"

Worst case: Exponential in dimension (unavoidable for NP-hard problems)
Typical case: Polynomial for problems with tight relaxations

OPOCH is **not** a polynomial-time solver. It's an exact solver with anytime behavior.

### "How do you handle numerical error?"

1. **Outward rounding:** All interval operations round outward by ROUND_EPS = 1e-15
2. **Feasibility tolerance:** Points are feasible if constraint violation ≤ feas_tol
3. **Epsilon gap:** Certification requires UB - LB ≤ ε, not exact equality

We do not claim infinite precision. We claim:
- LB is a valid lower bound (conservative)
- UB is achievable by a nearly-feasible point
- Gap closure is within specified tolerance

### "Can I add custom relaxations?"

Yes. Implement the `WitnessConstructor` interface:

```python
class CustomRelaxation:
    def compute_bound(self, region, problem):
        # Return (lower_bound, certificate)
        pass

    def cost(self):
        # Return computational cost estimate
        pass
```

Register it in the witness lattice.

### "Why JSON for receipts instead of binary?"

1. **Readability:** Humans can inspect receipts
2. **Portability:** No binary format versioning issues
3. **Canonical form:** Deterministic serialization for hashing
4. **Tooling:** Easy to process with standard tools

For large-scale runs, receipts can be compressed (gzip).

### "How do I cite this?"

```bibtex
@software{opoch_optimizer,
  title = {OPOCH Optimizer},
  author = {OPOCH Team},
  year = {2025},
  url = {https://github.com/yourusername/opoch-optimizer}
}
```

See [CITATION.cff](../CITATION.cff) for the full citation.

## Common Criticisms

### "This is overkill for most problems"

Correct. If you don't need certification, use a simpler optimizer.

OPOCH is for when you need:
- Proof of optimality (safety-critical applications)
- Detection of infeasibility (constraint validation)
- Exact remaining gap (decision-making under uncertainty)

### "The overhead of receipts is wasteful"

Receipts add ~10% overhead. For reproducibility and auditability, this is often worthwhile.

You can disable receipts: `OPOCHConfig(enable_receipts=False)`

### "Nobody will actually verify the receipts"

The point is that they **can** be verified. This changes the trust model:
- Traditional: "Trust the solver"
- OPOCH: "Trust, but verify"

Independent verification is possible without access to the original solver.

### "Real optimization problems don't have known optima"

True. But OPOCH's certificates don't require knowing the optimum:
- **LB** comes from relaxations
- **UB** comes from feasible points found during search
- **Gap** is UB - LB

The certificate proves gap ≤ ε without knowing f*.

## Getting Help

- **Issues:** https://github.com/yourusername/opoch-optimizer/issues
- **Discussions:** https://github.com/yourusername/opoch-optimizer/discussions
- **Email:** opoch@example.com
