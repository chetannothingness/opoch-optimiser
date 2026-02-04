# CasADi Benchmark Implementation Plan

## Goal: 100% Certification Across All CasADi Benchmarks

**The only Π-fixed definition**: Every case ends in UNIQUE / UNSAT / Ω with replayable receipts—never silent timeout or "best-effort".

---

## Architecture Overview: Two Contracts, One System

### Contract L: Local-KKT Certified
For any CasADi NLP (including huge OCP transcriptions):
- **UNIQUE(KKT)**: (x*, λ*) with certified KKT residuals ≤ ε_kkt and feasibility ≤ ε_feas
- **UNSAT**: Infeasible with certificate
- **Ω**: Solver cap reached with exact stop reason + residual bounds

### Contract G: Global Proof (UB–LB)
For bounded problems chosen for global certification:
- **UNIQUE-OPT**: UB - LB ≤ ε
- **UNSAT**: Refutation cover
- **Ω**: Only if cap imposed

---

## Implementation Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    BENCHMARK HARNESS                            │
│  run_casadi_suite.py → produces proof bundles for all cases     │
├─────────────────────────────────────────────────────────────────┤
│                    PROOF LAYER (Contract L/G)                   │
│  kkt_certificate.py     │  global_certificate.py                │
│  - Feasibility r_p      │  - UB from primal                     │
│  - Stationarity r_s     │  - LB from bounds engine              │
│  - Complementarity r_c  │  - Gap closure                        │
├─────────────────────────────────────────────────────────────────┤
│                    Δ* WITNESS SYNTHESIS                         │
│  - FBBT/Krawczyk contractors                                    │
│  - Precision escalation                                         │
│  - ResidualIR → GN bounds                                       │
│  - FactorIR → Chain DP bounds                                   │
│  - MINLP branching                                              │
├─────────────────────────────────────────────────────────────────┤
│                    IR COMPILER                                  │
│  casadi_to_ir() → ObjectiveIR (ExprIR/ResidualIR/FactorIR)     │
├─────────────────────────────────────────────────────────────────┤
│                    CASADI ADAPTER                               │
│  - SX/MX graph walking                                          │
│  - AD functions: grad_f, jac_g, hess_L                          │
│  - Canonical NLP representation                                 │
├─────────────────────────────────────────────────────────────────┤
│                    LOCAL SOLVER (Speed Layer)                   │
│  - IPOPT via CasADi nlpsol                                      │
│  - Bonmin for MINLP                                             │
│  - Deterministic options                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
src/opoch_optimizer/
├── casadi/
│   ├── __init__.py
│   ├── adapter.py              # CasADi SX/MX → ObjectiveIR
│   ├── graph_walker.py         # Walk CasADi expression trees
│   ├── nlp_contract.py         # Canonical NLP representation
│   ├── kkt_certificate.py      # KKT residual computation
│   ├── global_certificate.py   # UB-LB gap certificate
│   └── solver_wrapper.py       # Deterministic IPOPT/Bonmin wrapper

benchmarks/
├── casadi/
│   ├── __init__.py
│   ├── suite_a_industrial.py   # OCP, robotics, control
│   ├── suite_b_regression.py   # NIST-like least squares
│   ├── suite_c_minlp.py        # Mixed-integer NLP
│   ├── run_casadi_kkt.py       # Contract L runner
│   ├── run_casadi_global.py    # Contract G runner
│   ├── run_casadi_all.py       # Complete benchmark
│   └── replay_verify.py        # Verification script

runs/                           # Output directory
└── casadi/
    └── <suite>/<case_id>/
        ├── contract.json
        ├── nlp.json
        ├── result.json
        ├── kkt_certificate.json
        ├── global_certificate.json (if Contract G)
        └── receipts/chain.json
```

---

## Phase 1: CasADi Adapter

### 1.1 Core Adapter (`casadi/adapter.py`)

```python
@dataclass
class CasADiNLP:
    """Canonical CasADi NLP representation."""
    x: ca.SX              # Decision variables
    f: ca.SX              # Objective
    g: ca.SX              # Constraints
    p: ca.SX              # Parameters (optional)
    lbx: np.ndarray       # Variable lower bounds
    ubx: np.ndarray       # Variable upper bounds
    lbg: np.ndarray       # Constraint lower bounds
    ubg: np.ndarray       # Constraint upper bounds
    x0: np.ndarray        # Initial guess
    name: str

class CasADiAdapter:
    """Convert CasADi NLP to OPOCH ObjectiveIR."""

    def __init__(self, nlp: CasADiNLP):
        self.nlp = nlp
        self._build_ad_functions()

    def _build_ad_functions(self):
        """Build CasADi AD functions for gradients/Jacobians."""
        x = self.nlp.x
        f = self.nlp.f
        g = self.nlp.g

        # Gradient of objective
        self.grad_f = ca.Function('grad_f', [x], [ca.jacobian(f, x).T])

        # Jacobian of constraints
        self.jac_g = ca.Function('jac_g', [x], [ca.jacobian(g, x)])

        # Hessian of Lagrangian (for second-order info)
        lam = ca.SX.sym('lam', g.shape[0])
        L = f + ca.dot(lam, g)
        self.hess_L = ca.Function('hess_L', [x, lam], [ca.hessian(L, x)[0]])

    def to_objective_ir(self) -> ObjectiveIR:
        """Compile CasADi NLP to ObjectiveIR."""
        # Walk CasADi expression tree
        graph = self._walk_casadi_graph(self.nlp.f)

        # Detect structure
        if self._is_sum_of_squares():
            return self._build_residual_ir()
        elif self._is_factor_structure():
            return self._build_factor_ir()
        else:
            return ExprIR(graph=graph, n_vars=self.nlp.x.shape[0])

    def _walk_casadi_graph(self, expr: ca.SX) -> ExpressionGraph:
        """Convert CasADi SX to OPOCH ExpressionGraph."""
        # Recursive walk of CasADi DAG
        pass
```

### 1.2 Graph Walker (`casadi/graph_walker.py`)

```python
class CasADiGraphWalker:
    """Walk CasADi SX/MX expression trees."""

    # CasADi operation codes
    OP_CONST = 0
    OP_INPUT = 1
    OP_ADD = 2
    OP_SUB = 3
    OP_MUL = 4
    OP_DIV = 5
    OP_SQ = 6
    OP_SQRT = 7
    OP_SIN = 8
    OP_COS = 9
    OP_EXP = 10
    OP_LOG = 11
    # ... etc

    def walk(self, expr: ca.SX) -> ExpressionGraph:
        """Convert CasADi expression to ExpressionGraph."""
        graph = ExpressionGraph()
        node_map = {}

        # Get all operations in topological order
        work = ca.SX.get_input(expr) if hasattr(ca.SX, 'get_input') else [expr]

        for i in range(expr.n_dep()):
            dep = expr.dep(i)
            # Recursively process dependencies
            self._process_node(dep, graph, node_map)

        return graph
```

---

## Phase 2: KKT Certificate (Contract L)

### 2.1 KKT Residuals (`casadi/kkt_certificate.py`)

```python
@dataclass
class KKTCertificate:
    """Certified KKT conditions for local optimality."""
    x: np.ndarray                    # Primal solution
    lam_g: np.ndarray               # Constraint multipliers
    lam_x: np.ndarray               # Bound multipliers

    # Residuals
    r_primal: float                  # Primal feasibility
    r_stationarity: float            # Stationarity (∇L = 0)
    r_complementarity: float         # Complementarity

    # Tolerances
    eps_feas: float
    eps_kkt: float
    eps_comp: float

    # Status
    status: str                      # UNIQUE_KKT, FAIL, OMEGA

    # Hashes for replay
    input_hash: str
    certificate_hash: str

class KKTCertifier:
    """Compute and verify KKT certificates."""

    def __init__(self, adapter: CasADiAdapter):
        self.adapter = adapter
        self.nlp = adapter.nlp

    def certify(self, x: np.ndarray, lam_g: np.ndarray,
                eps_feas: float = 1e-6,
                eps_kkt: float = 1e-6,
                eps_comp: float = 1e-6) -> KKTCertificate:
        """Compute KKT certificate for solution (x, λ)."""

        # 1. Primal feasibility
        r_p = self._primal_residual(x)

        # 2. Stationarity (∇f + J_g^T λ + ν = 0)
        r_s = self._stationarity_residual(x, lam_g)

        # 3. Complementarity (λ ⊙ slack = 0)
        r_c = self._complementarity_residual(x, lam_g)

        # Determine status
        if r_p <= eps_feas and r_s <= eps_kkt and r_c <= eps_comp:
            status = "UNIQUE_KKT"
        else:
            status = "FAIL"

        return KKTCertificate(
            x=x, lam_g=lam_g, lam_x=self._compute_bound_multipliers(x),
            r_primal=r_p, r_stationarity=r_s, r_complementarity=r_c,
            eps_feas=eps_feas, eps_kkt=eps_kkt, eps_comp=eps_comp,
            status=status,
            input_hash=self._compute_input_hash(),
            certificate_hash=self._compute_certificate_hash(x, lam_g, r_p, r_s, r_c)
        )

    def _primal_residual(self, x: np.ndarray) -> float:
        """r_p = max{||[lbx-x]_+||, ||[x-ubx]_+||, ||[lbg-g(x)]_+||, ||[g(x)-ubg]_+||}"""
        nlp = self.nlp

        # Variable bound violations
        lb_viol = np.maximum(0, nlp.lbx - x)
        ub_viol = np.maximum(0, x - nlp.ubx)

        # Constraint violations
        g_val = np.array(self.adapter.nlp.g_func(x)).flatten()
        g_lb_viol = np.maximum(0, nlp.lbg - g_val)
        g_ub_viol = np.maximum(0, g_val - nlp.ubg)

        return max(
            np.max(np.abs(lb_viol)) if len(lb_viol) > 0 else 0,
            np.max(np.abs(ub_viol)) if len(ub_viol) > 0 else 0,
            np.max(np.abs(g_lb_viol)) if len(g_lb_viol) > 0 else 0,
            np.max(np.abs(g_ub_viol)) if len(g_ub_viol) > 0 else 0
        )

    def _stationarity_residual(self, x: np.ndarray, lam_g: np.ndarray) -> float:
        """r_s = ||∇f(x) + J_g(x)^T λ + ν||_∞"""
        grad_f = np.array(self.adapter.grad_f(x)).flatten()
        jac_g = np.array(self.adapter.jac_g(x))

        # Bound multipliers from active set
        nu = self._compute_bound_multipliers(x)

        # Stationarity: ∇f + J_g^T λ + ν = 0
        stationarity = grad_f + jac_g.T @ lam_g + nu

        return np.max(np.abs(stationarity))

    def _complementarity_residual(self, x: np.ndarray, lam_g: np.ndarray) -> float:
        """r_c = ||λ ⊙ constraint_slack||_∞"""
        g_val = np.array(self.adapter.nlp.g_func(x)).flatten()
        nlp = self.nlp

        # Slack for inequality constraints
        slack_lb = g_val - nlp.lbg
        slack_ub = nlp.ubg - g_val

        # Complementarity
        comp = np.abs(lam_g * np.minimum(slack_lb, slack_ub))

        return np.max(comp) if len(comp) > 0 else 0.0

    def _compute_bound_multipliers(self, x: np.ndarray) -> np.ndarray:
        """Compute ν from active bounds (projection-based)."""
        nlp = self.nlp
        nu = np.zeros_like(x)

        # Active lower bounds: ν_i < 0
        active_lb = np.abs(x - nlp.lbx) < 1e-8
        # Active upper bounds: ν_i > 0
        active_ub = np.abs(x - nlp.ubx) < 1e-8

        # Compute from stationarity (simplified)
        grad_f = np.array(self.adapter.grad_f(x)).flatten()
        nu[active_lb] = -grad_f[active_lb]  # Simplified
        nu[active_ub] = -grad_f[active_ub]

        return nu
```

---

## Phase 3: Global Certificate (Contract G)

### 3.1 Global Certifier (`casadi/global_certificate.py`)

```python
@dataclass
class GlobalCertificate:
    """Certified global optimality via UB-LB gap."""
    x_opt: np.ndarray
    upper_bound: float
    lower_bound: float
    gap: float
    epsilon: float
    status: str  # UNIQUE_OPT, UNSAT, OMEGA

    nodes_explored: int
    contractor_applications: int
    certificate_hash: str

class GlobalCertifier:
    """Global optimization certification using OPOCH engine."""

    def __init__(self, adapter: CasADiAdapter, config: OPOCHConfig):
        self.adapter = adapter
        self.config = config
        self.ir = adapter.to_objective_ir()

    def certify(self) -> GlobalCertificate:
        """Run global certification via branch-and-reduce."""
        # Build problem contract
        contract = self._build_contract()

        # Use appropriate kernel based on IR type
        if isinstance(self.ir, ResidualIR):
            # Use Gauss-Newton bounds
            kernel = self._build_ls_kernel(contract)
        elif isinstance(self.ir, FactorIR) and self.ir.is_chain:
            # Use chain-factor DP
            kernel = self._build_chain_kernel(contract)
        else:
            # Standard OPOCH kernel
            kernel = OPOCHKernel(contract, self.config)

        # Solve
        verdict, result = kernel.solve()

        # Build certificate
        return GlobalCertificate(
            x_opt=result.x_optimal if hasattr(result, 'x_optimal') else None,
            upper_bound=result.upper_bound,
            lower_bound=result.lower_bound,
            gap=result.gap,
            epsilon=self.config.epsilon,
            status=verdict.value,
            nodes_explored=result.nodes_explored,
            contractor_applications=0,  # TODO: track
            certificate_hash=self._compute_hash(result)
        )
```

---

## Phase 4: Deterministic Solver Wrapper

### 4.1 IPOPT Wrapper (`casadi/solver_wrapper.py`)

```python
class DeterministicIPOPT:
    """Deterministic IPOPT wrapper with fixed options."""

    # Fixed options for reproducibility
    DEFAULT_OPTIONS = {
        'ipopt.print_level': 0,
        'ipopt.tol': 1e-8,
        'ipopt.max_iter': 3000,
        'ipopt.acceptable_tol': 1e-6,
        'ipopt.linear_solver': 'mumps',
        'ipopt.mu_strategy': 'adaptive',
        'ipopt.warm_start_init_point': 'no',
        # Deterministic settings
        'ipopt.honor_original_bounds': 'yes',
        'ipopt.check_derivatives_for_naninf': 'yes',
    }

    def __init__(self, nlp: CasADiNLP, options: Dict = None):
        self.nlp = nlp
        self.options = {**self.DEFAULT_OPTIONS, **(options or {})}
        self._build_solver()

    def _build_solver(self):
        """Build CasADi nlpsol."""
        nlp_dict = {
            'x': self.nlp.x,
            'f': self.nlp.f,
            'g': self.nlp.g,
        }
        if self.nlp.p is not None:
            nlp_dict['p'] = self.nlp.p

        self.solver = ca.nlpsol('solver', 'ipopt', nlp_dict, self.options)

    def solve(self, x0: np.ndarray = None) -> Dict:
        """Solve with deterministic initial guess."""
        x0 = x0 if x0 is not None else self.nlp.x0

        result = self.solver(
            x0=x0,
            lbx=self.nlp.lbx,
            ubx=self.nlp.ubx,
            lbg=self.nlp.lbg,
            ubg=self.nlp.ubg,
        )

        return {
            'x': np.array(result['x']).flatten(),
            'f': float(result['f']),
            'g': np.array(result['g']).flatten(),
            'lam_g': np.array(result['lam_g']).flatten(),
            'lam_x': np.array(result['lam_x']).flatten(),
            'solver_stats': self.solver.stats(),
        }
```

---

## Phase 5: Benchmark Suites

### 5.1 Suite A: Industrial NLP (`benchmarks/casadi/suite_a_industrial.py`)

```python
def get_ocp_problems() -> List[CasADiNLP]:
    """Optimal Control Problems (robotics/control)."""
    problems = []

    # 1. Van der Pol oscillator (direct multiple shooting)
    problems.append(van_der_pol_ocp())

    # 2. Rocket landing (direct collocation)
    problems.append(rocket_landing_ocp())

    # 3. Robot arm trajectory
    problems.append(robot_arm_trajectory())

    # 4. NMPC tracking problem
    problems.append(nmpc_tracking())

    # 5. Parameter estimation under constraints
    problems.append(constrained_parameter_estimation())

    return problems

def van_der_pol_ocp() -> CasADiNLP:
    """Van der Pol oscillator control."""
    import casadi as ca

    # Time horizon and discretization
    T = 10.0
    N = 40

    # State and control
    x1 = ca.SX.sym('x1')
    x2 = ca.SX.sym('x2')
    u = ca.SX.sym('u')
    x = ca.vertcat(x1, x2)

    # ODE
    xdot = ca.vertcat(
        (1 - x2**2) * x1 - x2 + u,
        x1
    )

    # Formulate as NLP via multiple shooting
    # ... (standard CasADi OCP formulation)

    return CasADiNLP(...)
```

### 5.2 Suite B: Regression (`benchmarks/casadi/suite_b_regression.py`)

```python
def get_regression_problems() -> List[CasADiNLP]:
    """NIST-like nonlinear regression problems."""
    problems = []

    # 1. Misra1a
    problems.append(misra1a_regression())

    # 2. Chwirut
    problems.append(chwirut_regression())

    # 3. Gauss
    problems.append(gauss_regression())

    # 4. Box-Boden
    problems.append(box_boden_regression())

    return problems

def misra1a_regression() -> CasADiNLP:
    """Misra1a NIST regression problem."""
    import casadi as ca

    # Data
    x_data = np.array([77.6, 114.9, 141.1, ...])
    y_data = np.array([10.07, 14.73, 17.94, ...])

    # Parameters
    b1 = ca.SX.sym('b1')
    b2 = ca.SX.sym('b2')
    theta = ca.vertcat(b1, b2)

    # Residuals r(θ) where f = ||r||²
    residuals = []
    for i in range(len(x_data)):
        r_i = y_data[i] - b1 * (1 - ca.exp(-b2 * x_data[i]))
        residuals.append(r_i)

    r = ca.vertcat(*residuals)
    f = ca.dot(r, r)  # ||r||²

    return CasADiNLP(
        x=theta,
        f=f,
        g=ca.SX([]),  # No constraints
        lbx=np.array([0, 0]),
        ubx=np.array([1000, 1]),
        lbg=np.array([]),
        ubg=np.array([]),
        x0=np.array([250, 0.0005]),
        name='misra1a'
    )
```

### 5.3 Suite C: MINLP (`benchmarks/casadi/suite_c_minlp.py`)

```python
def get_minlp_problems() -> List[CasADiNLP]:
    """Mixed-Integer NLP problems."""
    problems = []

    # 1. ex1223a (MINLPLib)
    problems.append(ex1223a_minlp())

    # 2. Facility location
    problems.append(facility_location_minlp())

    # 3. Pooling problem
    problems.append(pooling_minlp())

    return problems
```

---

## Phase 6: Benchmark Runner

### 6.1 Main Runner (`benchmarks/casadi/run_casadi_all.py`)

```python
def run_casadi_benchmark(
    suites: List[str] = ['a', 'b', 'c'],
    contract: str = 'L',  # 'L' for KKT, 'G' for global, 'LG' for both
    output_dir: str = 'runs/casadi'
) -> Dict:
    """Run complete CasADi benchmark suite."""

    results = {}

    for suite_name in suites:
        suite = load_suite(suite_name)
        suite_results = []

        for problem in suite:
            print(f"[{suite_name}] {problem.name}")

            # Create output directory
            case_dir = Path(output_dir) / suite_name / problem.name
            case_dir.mkdir(parents=True, exist_ok=True)

            # Build adapter
            adapter = CasADiAdapter(problem)

            # Save canonical NLP
            save_nlp_json(problem, case_dir / 'nlp.json')

            # Run Contract L (KKT)
            if 'L' in contract:
                kkt_result = run_contract_l(adapter, case_dir)
                suite_results.append(kkt_result)

            # Run Contract G (Global) if requested
            if 'G' in contract:
                global_result = run_contract_g(adapter, case_dir)
                suite_results.append(global_result)

            # Save receipts
            save_receipts(case_dir / 'receipts')

        results[suite_name] = suite_results

    # Generate summary
    print_summary(results)

    return results

def run_contract_l(adapter: CasADiAdapter, case_dir: Path) -> Dict:
    """Run Contract L (KKT certification)."""
    # Solve with IPOPT
    solver = DeterministicIPOPT(adapter.nlp)
    sol = solver.solve()

    # Certify KKT
    certifier = KKTCertifier(adapter)
    cert = certifier.certify(sol['x'], sol['lam_g'])

    # Save certificate
    save_json(cert.to_dict(), case_dir / 'kkt_certificate.json')

    # Save result
    result = {
        'contract': 'L',
        'status': cert.status,
        'objective': sol['f'],
        'residuals': {
            'primal': cert.r_primal,
            'stationarity': cert.r_stationarity,
            'complementarity': cert.r_complementarity,
        }
    }
    save_json(result, case_dir / 'result.json')

    return result
```

### 6.2 Replay Verifier (`benchmarks/casadi/replay_verify.py`)

```python
def verify_all(runs_dir: str = 'runs/casadi') -> bool:
    """Verify all benchmark results."""
    runs_path = Path(runs_dir)
    all_passed = True

    for suite_dir in runs_path.iterdir():
        if not suite_dir.is_dir():
            continue

        for case_dir in suite_dir.iterdir():
            if not case_dir.is_dir():
                continue

            passed = verify_case(case_dir)
            if not passed:
                print(f"FAIL: {case_dir}")
                all_passed = False
            else:
                print(f"PASS: {case_dir}")

    return all_passed

def verify_case(case_dir: Path) -> bool:
    """Verify a single case."""
    # Load NLP
    nlp = load_nlp_json(case_dir / 'nlp.json')

    # Load stored result
    result = load_json(case_dir / 'result.json')

    # Re-run solver with same options
    adapter = CasADiAdapter(nlp)
    solver = DeterministicIPOPT(nlp)
    sol = solver.solve()

    # Re-compute certificate
    certifier = KKTCertifier(adapter)
    cert = certifier.certify(sol['x'], sol['lam_g'])

    # Load stored certificate
    stored_cert = load_json(case_dir / 'kkt_certificate.json')

    # Verify hashes match
    if cert.certificate_hash != stored_cert['certificate_hash']:
        return False

    # Verify receipt chain
    receipts = load_json(case_dir / 'receipts' / 'chain.json')
    if not verify_receipt_chain(receipts):
        return False

    return True
```

---

## Phase 7: Implementation Order

### Week 1: Core CasADi Adapter
1. `casadi/adapter.py` - CasADiNLP dataclass + AD function builders
2. `casadi/graph_walker.py` - SX/MX to ExpressionGraph conversion
3. `casadi/nlp_contract.py` - Canonical JSON serialization
4. Test with simple problems

### Week 2: Contract L (KKT)
1. `casadi/kkt_certificate.py` - Full KKT residual computation
2. `casadi/solver_wrapper.py` - Deterministic IPOPT wrapper
3. Receipt integration
4. Test on Suite B (regression - simplest)

### Week 3: Benchmark Suites
1. `benchmarks/casadi/suite_a_industrial.py` - OCP problems
2. `benchmarks/casadi/suite_b_regression.py` - NIST regression
3. `benchmarks/casadi/suite_c_minlp.py` - MINLP problems
4. `benchmarks/casadi/run_casadi_kkt.py` - Contract L runner

### Week 4: Contract G (Global)
1. `casadi/global_certificate.py` - Global certification
2. Integration with existing OPOCH bounds engine
3. IR-specific optimizations (ResidualIR, FactorIR)
4. `benchmarks/casadi/run_casadi_global.py` - Contract G runner

### Week 5: Verification & Polish
1. `benchmarks/casadi/replay_verify.py` - Complete verifier
2. `benchmarks/casadi/run_casadi_all.py` - Unified runner
3. CI integration
4. Documentation

---

## Success Criteria

**100% = every case produces a valid bundle that replay-verifies**

| Metric | Target |
|--------|--------|
| Contract L (KKT) pass rate | 100% |
| Contract G (Global) pass rate | 100% (on selected subset) |
| Replay verification | 100% |
| Receipt chain validity | 100% |

---

## Hard Gates (What Competitors Cannot Offer)

1. **Feasibility Gate**: r_p ≤ ε_feas (proven, not "approximately")
2. **KKT Gate**: r_s ≤ ε_kkt (certified stationarity)
3. **Replay Gate**: Same input → same output hash
4. **Auditability**: Every step in receipt chain

**This is how you beat everyone: make the comparison gate-based, then dominate the Pareto frontier.**
