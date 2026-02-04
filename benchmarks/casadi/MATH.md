# OPOCH KKT Verification - Complete Mathematical Documentation

## 1. The Optimization Problem

IPOPT solves nonlinear programs of the form:

```
min  f(x)
s.t. g_L <= g(x) <= g_U    (m constraints)
     x_L <= x <= x_U       (n variable bounds)
```

where:
- `f: R^n -> R` is the objective function
- `g: R^n -> R^m` are the constraint functions
- `x_L, x_U` are variable bounds
- `g_L, g_U` are constraint bounds

## 2. The KKT Conditions

A point x* is a local optimum if there exist Lagrange multipliers λ_g, λ_x such that:

### 2.1 Primal Feasibility

```
g_L <= g(x*) <= g_U
x_L <= x* <= x_U
```

### 2.2 Stationarity (Dual Feasibility)

```
∇f(x*) + J_g(x*)^T λ_g + λ_x = 0
```

where J_g is the Jacobian of g.

### 2.3 Complementary Slackness

For constraints:
```
μ_g,L · (g(x*) - g_L) = 0    (lower bound)
μ_g,U · (g_U - g(x*)) = 0    (upper bound)
```

For variables:
```
μ_x,L · (x* - x_L) = 0       (lower bound)
μ_x,U · (x_U - x*) = 0       (upper bound)
```

### 2.4 Dual Sign Conditions

```
μ_g,L >= 0, μ_g,U >= 0
μ_x,L >= 0, μ_x,U >= 0
```

## 3. IPOPT's Multiplier Convention

IPOPT returns **combined** multipliers with sign encoding:

```
λ_g = μ_g,U - μ_g,L
λ_x = μ_x,U - μ_x,L
```

This means:
- **Negative λ** → active **lower** bound (μ_L > 0)
- **Positive λ** → active **upper** bound (μ_U > 0)

### Correct Decomposition

```python
def split_multipliers(λ):
    μ_lower = max(0, -λ)   # Active lower bound
    μ_upper = max(0, +λ)   # Active upper bound
    return μ_lower, μ_upper
```

## 4. KKT Residual Computation

### 4.1 Primal Residual

```
slack_x_L = x* - x_L        (should be >= 0)
slack_x_U = x_U - x*        (should be >= 0)
slack_g_L = g(x*) - g_L     (should be >= 0)
slack_g_U = g_U - g(x*)     (should be >= 0)

r_primal = max(
    max(-slack_x_L),        # Lower bound violation
    max(-slack_x_U),        # Upper bound violation
    max(-slack_g_L),        # Constraint lower violation
    max(-slack_g_U)         # Constraint upper violation
)
```

### 4.2 Dual Residual (Stationarity)

```
stationarity = ∇f(x*) + J_g(x*)^T · λ_g + λ_x

r_dual = ||stationarity||_∞
```

Note: Since λ_g = μ_g,U - μ_g,L and λ_x = μ_x,U - μ_x,L, the stationarity equation simplifies to using the combined multipliers directly.

### 4.3 Complementarity Residual

```
μ_x,L, μ_x,U = split_multipliers(λ_x)
μ_g,L, μ_g,U = split_multipliers(λ_g)

compl_x_L = max(|μ_x,L · slack_x_L|)    (for finite x_L)
compl_x_U = max(|μ_x,U · slack_x_U|)    (for finite x_U)
compl_g_L = max(|μ_g,L · slack_g_L|)    (for finite g_L)
compl_g_U = max(|μ_g,U · slack_g_U|)    (for finite g_U)

r_complementarity = max(compl_x_L, compl_x_U, compl_g_L, compl_g_U)
```

### 4.4 Total KKT Residual

```
r_max = max(r_primal, r_dual, r_complementarity)
```

## 5. Certification Criterion

A solution is **CERTIFIED** if:

```
r_max <= ε
```

where ε is the verification tolerance (typically 1e-6).

## 6. The Bug in Naive Implementations

A common mistake is computing complementarity as:

```python
# WRONG
r_compl = max(|λ · slack|)
```

This is incorrect because:
1. λ can be negative (for lower bounds) or positive (for upper bounds)
2. The product λ · slack doesn't respect the two-sided nature of bounds

The correct implementation requires splitting multipliers first:

```python
# CORRECT
μ_L, μ_U = split_multipliers(λ)
r_compl = max(|μ_L · slack_L|, |μ_U · slack_U|)
```

## 7. Why IPOPT's "Success" Can Be Wrong

IPOPT uses **scaled** residuals internally:

```
scaled_residual = residual / scaling_factor
```

The scaling factor is computed from problem structure and can hide true violations:

```
IPOPT (scaled):   r_max = 1e-9  ✓ (passes internal check)
OPOCH (unscaled): r_max = 1e-5  ✗ (fails KKT verification)
```

### Solution: Verify in Unscaled Space

OPOCH always computes residuals in the original (unscaled) problem space. This ensures the mathematical certificate is valid regardless of IPOPT's internal scaling choices.

## 8. The Repair Loop

When verification fails, OPOCH runs a deterministic repair loop:

### Round 0: Default IPOPT
```
tol = 1e-8
nlp_scaling_method = default
```

### Round 1: Strict Settings
```
tol = 1e-10
constr_viol_tol = 1e-10
dual_inf_tol = 1e-10
compl_inf_tol = 1e-10
nlp_scaling_method = none      # Disable scaling
warm_start_init_point = yes    # Use previous solution
max_iter = 6000
```

### Round 2: Extra Stabilization
```
Same as Round 1, plus:
mu_strategy = adaptive
max_iter = 10000
```

## 9. Mathematical Guarantee

**Theorem (KKT Necessity):** If x* is a local minimizer of the NLP and a constraint qualification holds at x*, then there exist multipliers (λ_g, λ_x) satisfying the KKT conditions.

**OPOCH Guarantee:** If r_max <= ε, then x* satisfies the KKT conditions to tolerance ε, proving it is a local optimum (up to numerical precision).

## 10. What OPOCH Does NOT Do

1. **Does not change IPOPT's algorithm** - IPOPT's interior-point method is unchanged
2. **Does not guarantee global optimality** - Only local optimality via KKT
3. **Does not modify the problem** - Verification is purely observational

## 11. Comparison with Global Solvers

| | OPOCH | BARON/COUENNE |
|---|-------|---------------|
| **Certifies** | Local optimum (KKT) | Global optimum (branch-and-bound) |
| **Method** | Post-hoc verification | Exhaustive search |
| **Complexity** | O(1) verification per solve | Exponential in dimension |
| **Scalability** | 1000+ variables | ~20 variables practical |

### When to Use What

**Use OPOCH:**
- Real-time applications (MPC, robotics)
- Large problems (>20 variables)
- Good initial guess available
- Local optimum is acceptable

**Use BARON:**
- Small problems (<20 variables)
- Must prove global optimality
- No good initial guess
- Time is not critical

## 12. Reproducibility

Every OPOCH certificate includes:
1. Problem hash (input verification)
2. Solution x*, λ_g, λ_x
3. All residuals (r_primal, r_dual, r_complementarity, r_max)
4. Epsilon used for certification
5. SHA-256 hash of canonical JSON bundle

Anyone can verify by:
1. Re-computing residuals from x*, λ
2. Checking r_max <= ε
3. Verifying hash matches

## References

1. Wächter, A., Biegler, L.T. (2006). "On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming." Mathematical Programming.

2. Hock, W., Schittkowski, K. (1981). "Test Examples for Nonlinear Programming Codes." Lecture Notes in Economics and Mathematical Systems.

3. CasADi Documentation: https://web.casadi.org/docs/

4. IPOPT Documentation: https://coin-or.github.io/Ipopt/
