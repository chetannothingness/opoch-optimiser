# OPOCH Nonlinear Optimizer — Complete Math Anchor

**A single proof-carrying optimization kernel across industries (deterministic, replayable, no guesses)**

This document is the one canonical reference for the OPOCH nonlinear optimizer: the mathematical contract, the forced truth objects, the complete witness algebra (Δ\*), and the end-to-end algorithms for deterministic (DAG), constrained, MINLP, least-squares, noisy, black-box, and large-scale regimes.

Everything is stated so an implementation agent can map it directly to code and verifiers.

---

## 0) The only admissible outputs (global output gate)

Given a problem instance $\mathcal{I}$, OPOCH must output exactly one:

### UNIQUE-OPT

A feasible point $x^*$ and a proof bundle such that:

$$UB - LB \le \varepsilon, \quad UB = f(x^*), \quad LB \le f^*$$

### UNSAT

A proof bundle that the feasible set is empty:

$$F = \emptyset$$

### Ω-GAP

Only if a user sets a hard resource cap. Output:
- best feasible UB found (or $+\infty$),
- certified LB,
- the gap $UB - LB$,
- and the minimal next missing distinguisher.

**No other output exists. "Best effort" is forbidden.**

---

## 1) The optimization contract (what is being solved)

### 1.1 Deterministic constrained optimization (proof mode)

Given:
- bounded domain $X = [\ell, u] \subset \mathbb{R}^n$,
- objective $f: X \to \mathbb{R}$,
- constraints $g_i(x) \le 0$ $(i = 1..m)$,
- constraints $h_j(x) = 0$ $(j = 1..p)$,
- tolerance $\varepsilon > 0$.

Feasible set:

$$F = \{x \in X : g(x) \le 0, h(x) = 0\}$$

Optimum:

$$f^* = \inf_{x \in F} f(x)$$

Certification requires:
- a feasible witness $x^*$ (UB),
- a global lower bound witness $LB \le f^*$,
- gap closure $UB - LB \le \varepsilon$.

### 1.2 MINLP extension

A subset of variables $z \subset x$ must be integer:

$$z \in \mathbb{Z}^k$$

Proof mode still requires UNIQUE-OPT or UNSAT; integer feasibility is enforced by deterministic branching.

### 1.3 Noisy objective (statistical proof mode)

Each evaluation is random:

$$Y(x) = \mu(x) + \eta$$

Truth becomes a $(1 - \delta)$-certificate:
- UNIQUE-OPT($\delta$) if $UB - LB \le \varepsilon$ holds with prob $\ge 1 - \delta$.

### 1.4 Pure black-box (structure-contract mode)

If only $f(x)$ queries exist and no structural model is declared, global certification is not definable from finite witnesses.

Therefore black-box proof mode requires an explicit structure contract (e.g., Lipschitz, low effective dimension, factor graph). Otherwise Ω by interface deficit.

---

## 2) The forced truth object (why every solver must look like this)

Any "global optimum" claim is the separation claim:

$$\forall x \in F, f(x) \ge UB - \varepsilon$$

So every truthful solver must maintain:
- **UB**: value of some verified feasible point,
- **LB**: a certified global lower bound,

and terminate only when $UB - LB \le \varepsilon$.

This forces the **region-wise bound normal form**:

Partition $X$ into regions $R$ (boxes). Maintain $LB(R)$ such that:

$$LB(R) \le \inf_{x \in F \cap R} f(x)$$

Then:

$$LB = \min_{R \in \mathcal{Q}} LB(R)$$

where $\mathcal{Q}$ is the active region queue.

---

## 3) The real missing foundation: Π-fixed enclosure semantics (dependency problem solved)

Plain interval arithmetic is sound but not Π-fixed: it breaks algebraic identity (dependency).

So the base witness algebra must use an enclosure semantics that preserves correlations:

### 3.1 Affine arithmetic

Represent uncertain variables using shared noise symbols:

$$x = x_0 + \sum_k a_k \epsilon_k + I, \quad \epsilon_k \in [-1, 1]$$

Then $x - x = 0$ exactly, and correlations persist.

### 3.2 Taylor models (polynomial + remainder)

Over a box $R$, represent:

$$f(x) \in P_f(\epsilon) + I_f$$

where $P_f$ is a degree-$p$ polynomial in the affine symbols and $I_f$ is a rigorous remainder.

**This is the foundational fix** that makes LB tightening and pruning actually converge without exponential overestimation.

---

## 4) Δ\*: the complete witness algebra (what tests exist)

OPOCH's power is not "search." It is **endogenous closure of admissible witnesses**.

### 4.1 Primitive interface (Δ₀)

- trace objective and constraints into an expression IR (see §5),
- enclosure evaluation (AA/TM),
- verifiers for feasibility,
- deterministic region splitting.

### 4.2 Closure constructors (Δ\*)

Δ\* must include:

#### (A) Constraint contractors (forced)

- **FBBT** (inequalities + equalities): fixed point of forward/backward narrowing.
- **Krawczyk / interval Newton** for equality manifolds: refute or contract boxes.
- **Kink splitting** (abs/sqrt/max) and denominator sign splitting (division near 0).

#### (B) Bound constructors (forced)

- **Taylor-model LB**: $\underline{f}(R)$ from TM.
- **McCormick convex relaxations** built on tight variable ranges (optional strengthening).
- **OBBT** (optimization-based tightening) as needed.

#### (C) Structure witnesses (forced)

- **Separable decomposition**: detect disjoint variable supports → exact block LB.
- **Least-squares IR**: residual-level reasoning and Gauss–Newton bounds.
- **Chain factor DP bound**: for objectives of form $\sum \phi(x_i, x_{i+1})$, compute certified global LB by interval-rectangle tables + DP.

#### (D) Integrality witness (MINLP)

- Branch on most fractional integer variable deterministically.

#### (E) Precision escalation

- Deterministic precision ladder for ill-conditioning; contraction stability is a witness.

---

## 5) The canonical objective/constraint IR (the missing bridge that unifies industries)

The solver must not accept only a scalar callable. It must accept a **Π-fixed normal form IR**.

**ObjectiveIR** has exactly one of:

1. **ExprIR**: full expression DAG of $f, g, h$
2. **ResidualIR**: residual vector $r(\theta)$ s.t. $f(\theta) = \|r(\theta)\|^2$
3. **FactorIR**: $f(x) = \sum_\alpha f_\alpha(x_{S_\alpha})$ with scopes $S_\alpha$

### How to obtain IR

- If user provides IR: use it.
- Else trace callables into ExprIR via operator-overloaded tracing that expands dot/loops/sum/prod into DAGs.
- Then run IR detectors:
  - detect sum-of-squares → ResidualIR
  - detect pairwise chain → FactorIR
  - detect separability → block factors

**This is the "once and for all" interface closure**: it enables GN bounds (NIST) and chain DP bounds (Rana/Eggholder) deterministically.

---

## 6) The complete deterministic algorithms

### 6.1 Feasibility-first (mandatory UB or UNSAT)

**Feasibility is a proof problem, not luck.**

Maintain a queue of regions $R$ covering $X$. For each $R$:

1. Apply contractors to fixed point:
   - FBBT (TM)
   - Krawczyk/Newton (equalities)
   - kink/denominator splits as required
2. If any variable interval empties → refute region with certificate
3. If region small → test midpoint feasibility
4. Else split deterministically (longest side or highest remainder contribution)

If a feasible point found → UB seed exists.
If all regions refuted → UNSAT cover.

### 6.2 Optimization (branch-and-reduce to close UB−LB)

Maintain active regions with certified $LB(R)$. Repeat:

1. Reduce region $R$ with contractors (and precision escalation if unstable)
2. Compute $LB(R)$ using strongest applicable witness:
   - separable exact block LB
   - least-squares GN/TM lower model
   - chain-factor DP LB
   - McCormick/TM LB fallback
3. Improve UB deterministically inside contracted regions
4. MINLP: if integer vars fractional in relaxation → branch deterministically
5. Prune if $LB(R) \ge UB - \varepsilon$
6. Split otherwise

Terminate when:

$$UB - \min_{R \in \mathcal{Q}} LB(R) \le \varepsilon$$

Output UNIQUE-OPT bundle.

---

## 7) Specialized math for key industry classes

### 7.1 Least squares (NIST, calibration, system ID)

Given ResidualIR $r(\theta)$, Jacobian $J$ is computable by AD on residual graph.

**Gauss–Newton lower model** on a region $R$:

$$r(\theta) = r(\theta_0) + J(\theta_0)\Delta + e(\theta), \quad \|e(\theta)\| \le E_R$$

from Taylor remainders.

Then:

$$\|r(\theta)\| \ge \|r(\theta_0) + J(\theta_0)\Delta\| - E_R$$

so:

$$f(\theta) \ge \left(\max(0, \|r(\theta_0) + J\Delta\| - E_R)\right)^2$$

Minimize this convex expression over $\Delta \in R - \theta_0$ to obtain $LB_R > 0$ when the true minimum is $> 0$.

Also add **UB-induced residual caps**:

$$f(\theta) \le UB \Rightarrow |r_i(\theta)| \le \sqrt{UB} \quad \forall i$$

and propagate via contractors to contract $\theta$.

### 7.2 Nearest-neighbor nonseparable chains (Rana/Eggholder)

If:

$$f(x) = \sum_{i=1}^{n-1} \phi(x_i, x_{i+1})$$

partition each variable range into $K$ intervals $I_{i,k}$. Compute certified rectangle lower bounds:

$$m_i(k, \ell) \le \inf_{x \in I_{i,k}, y \in I_{i+1,\ell}} \phi(x, y)$$

via TM enclosure (and optional 2D tightening).

Then global LB is DP:

$$LB = \min_{k_1, \dots, k_n} \sum_{i=1}^{n-1} m_i(k_i, k_{i+1})$$

in $O(nK^2)$, fully certified.

Eggholder (2D) is the $n = 2$ case.

### 7.3 MINLP (industrial design + discrete choices)

Branch on most fractional integer variable:

$$z \le \lfloor z^* \rfloor \quad \vee \quad z \ge \lceil z^* \rceil$$

with deterministic tie-break. Combine with same contractors + LB.

### 7.4 Ill-conditioning / thin feasible manifolds (Princeton)

Use precision escalation until contractors stabilize. The correctness of refutation/contraction is a witness property; if it changes with precision, increase precision deterministically and log it.

---

## 8) Verification and replay (industry-grade)

Every run emits:
- canonical input contract hash
- IR hash (ExprIR/ResidualIR/FactorIR)
- event log (contract/contractor applications, splits, bounds, prunes, UB updates)
- certificates for each EMPTY and each $LB(R)$
- final result bundle
- SHA-256 chain over canonical JSON events

**Replay verifier recomputes all steps and matches hashes.**

---

## 9) Benchmarking across industries (how to publish "nailed it" honestly)

Publish two scoreboards:

### Proof scoreboard (Π-OPT)

- % UNIQUE-OPT + % UNSAT = 100%
- time-to-proof p50/p95
- nodes explored
- max precision used
- replay hash pass rate (must be 100%)

### Anytime scoreboard (COCO/IOH)

- best-so-far curves and ECDF/ERT
- determinism (same curve replayable)
- optional Ω gap objects

**Never mix "proof mode" and "anytime mode" into one misleading percentage.**

---

## 10) The one-line claim (true and complete)

> **OPOCH is a deterministic, proof-carrying global optimizer whose witness algebra is complete for bounded computable problems when expressed in a canonical IR, using Π-fixed enclosures (AA/TM), contractors (FBBT/Krawczyk), structure witnesses (separable / least-squares / chain DP), integrality branching, and precision escalation—so every instance terminates in UNIQUE-OPT or UNSAT with replayable receipts.**

---

## Document Hierarchy

This is the **primary mathematical reference**. All other docs support specific aspects:

| Document | Purpose |
|----------|---------|
| **MATH_ANCHOR.md** (this) | Complete mathematical foundation |
| PI_FIXED_INTERFACE.md | ObjectiveIR implementation details |
| FAMILY_WITNESS.md | Structure detection heuristics |
| DELTA_COMPLETION.md | Witness algebra completion |
| CERTIFICATES.md | Certificate format specification |
| METHODOLOGY.md | Implementation methodology |
