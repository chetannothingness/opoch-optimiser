# CERTIFICATES — What the Outputs Mean and How to Verify Them

OPOCH produces three types of certificates, each with specific contents and verification procedures.

## Overview

| Verdict | Meaning | Verification |
|---------|---------|--------------|
| **UNIQUE-OPT** | x* is globally ε-optimal | Check feasibility + gap closure |
| **UNSAT** | No feasible point exists | Check refutation cover |
| **Ω-GAP** | Budget exhausted with exact gap | Check bound validity |

## UNIQUE-OPT Certificate

### Contents

```json
{
  "verdict": "UNIQUE-OPT",
  "x_optimal": [0.5, 0.5],
  "upper_bound": 0.5,
  "lower_bound": 0.499998,
  "gap": 0.000002,
  "epsilon": 0.0001,
  "constraint_violations": {
    "ineq_max": 0.0,
    "eq_max": 1e-15
  },
  "region_certificates": [
    {"region_id": 0, "lb": 0.499998, "tier": 1, "hash": "abc123..."}
  ],
  "nodes_explored": 42,
  "receipt_chain_hash": "final_hash..."
}
```

### Verification Procedure

1. **Feasibility check:**
   - Confirm x* ∈ X (within bounds)
   - Confirm gᵢ(x*) ≤ feas_tol for all inequality constraints
   - Confirm |hⱼ(x*)| ≤ feas_tol for all equality constraints

2. **Upper bound check:**
   - Recompute f(x*) and confirm it equals UB

3. **Lower bound check:**
   - For each region certificate, verify the LB is valid:
     - Tier 0: Recompute interval bound
     - Tier 1: Recompute McCormick relaxation
     - Tier 2: Verify FBBT/OBBT certificate
   - Confirm global LB = min over all region LBs

4. **Gap closure check:**
   - Confirm UB - LB ≤ ε

### Code

```python
from opoch_optimizer.verify.checks import verify_unique_opt_certificate

result = verify_unique_opt_certificate(certificate, problem)
assert result.valid, result.error_message
```

## UNSAT Certificate

### Contents

```json
{
  "verdict": "UNSAT",
  "refutation_type": "cover",
  "cover": [
    {
      "region_id": 0,
      "bounds": {"lower": [0, 0], "upper": [0.5, 1]},
      "refutation": {
        "type": "fbbt_infeasible",
        "constraint_index": 0,
        "output_interval": [0.1, 0.5]
      }
    },
    {
      "region_id": 1,
      "bounds": {"lower": [0.5, 0], "upper": [1, 1]},
      "refutation": {
        "type": "interval_refutation",
        "constraint_index": 0,
        "output_interval": [0.2, 0.8]
      }
    }
  ],
  "nodes_explored": 100,
  "receipt_chain_hash": "final_hash..."
}
```

### Verification Procedure

1. **Cover completeness:**
   - Confirm the union of all refuted regions equals X
   - No gaps in coverage

2. **Refutation validity:**
   - For each region in the cover:
     - Re-run the refutation proof
     - Interval: Verify 0 ∉ constraint interval
     - FBBT: Verify forward pass shows 0 ∉ h([l,u])
     - Newton: Verify empty intersection

3. **No false refutations:**
   - Confirm each refutation is mathematically sound

### Code

```python
from opoch_optimizer.verify.checks import verify_unsat_certificate

result = verify_unsat_certificate(certificate, problem)
assert result.valid, result.error_message
```

## Ω-GAP Certificate

### Contents

```json
{
  "verdict": "OMEGA-GAP",
  "upper_bound": 1.5,
  "lower_bound": 0.8,
  "gap": 0.7,
  "budget_exhausted": "time",
  "x_best": [0.3, 0.7],
  "next_separator_action": {
    "type": "SPLIT",
    "region_id": 5,
    "expected_gap_reduction": 0.15
  },
  "active_regions": 23,
  "nodes_explored": 10000,
  "receipt_chain_hash": "final_hash..."
}
```

### Verification Procedure

1. **Lower bound validity:**
   - Verify each active region's LB certificate
   - Confirm LB = min over active region LBs

2. **Upper bound validity (if finite):**
   - Verify x_best is feasible
   - Confirm f(x_best) = UB

3. **Gap calculation:**
   - Confirm gap = UB - LB

4. **Next action validity:**
   - Confirm the proposed action is executable
   - Verify expected gap reduction estimate is reasonable

### Code

```python
from opoch_optimizer.verify.checks import verify_omega_gap_certificate

result = verify_omega_gap_certificate(certificate, problem)
assert result.valid, result.error_message
```

## Receipt Chain Verification

Every certificate includes a `receipt_chain_hash` that summarizes the entire solve trajectory.

### Procedure

```python
from opoch_optimizer.verify.replay import replay_and_verify

# Replay the entire solve from receipts
result = replay_and_verify(run_folder)

# Check all hashes match
assert result.chain_valid, "Receipt chain tampering detected"
assert result.final_hash == certificate["receipt_chain_hash"]
```

### What Replay Checks

1. **Bound computations:** Each bound certificate is recomputed
2. **Prune decisions:** Each prune is verified (LB ≥ UB - ε)
3. **Split operations:** Child bounds are recomputed
4. **Incumbent updates:** Each UB update is verified feasible
5. **Hash chain:** Each receipt hash matches the recomputed value

## Common Verification Failures

| Failure | Meaning | Action |
|---------|---------|--------|
| Feasibility violation | x* doesn't satisfy constraints | Bug in feasibility check |
| Bound mismatch | Recomputed LB differs | Bug in bound computation |
| Gap not closed | UB - LB > ε but claimed UNIQUE-OPT | Bug in termination check |
| Incomplete cover | UNSAT claimed but gaps in cover | Bug in region tracking |
| Hash mismatch | Receipt chain broken | Possible tampering or bug |

## Verification Commands

```bash
# Verify a single run
python scripts/replay_verify_run.py results/ioh/run_001/

# Verify all runs in a folder
python scripts/replay_verify_run.py results/ioh/ --recursive

# Detailed verification output
python scripts/replay_verify_run.py results/ioh/run_001/ --verbose
```
