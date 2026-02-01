# IOH Integration — Exact Steps

This document describes how OPOCH integrates with IOHprofiler for standardized benchmarking.

## Goal

Produce valid IOHprofiler logs uploadable to [IOHanalyzer](https://iohanalyzer.liacs.nl/).

## IOH Rules (Non-Negotiable)

1. **IOH controls objective evaluation.** All f(x) calls go through the IOH wrapper.
2. **Best-so-far must be finite after first evaluation.** No NaN/inf at any point.
3. **Map undefined to finite penalty.** If f(x) is undefined, return a large finite value deterministically.
4. **OPOCH runs in anytime mode.** Log best-so-far after each evaluation.
5. **Certificates are sidecar files.** Store full OPOCH output alongside IOH logs.

## Directory Structure

```
results/ioh/OPOCH_BBOB/
├── data_f1/                    # IOH format: per-function folders
│   ├── IOHprofiler_f1_DIM2.dat
│   ├── IOHprofiler_f1_DIM10.dat
│   └── IOHprofiler_f1_DIM20.dat
├── data_f2/
│   └── ...
├── IOHprofiler_f1.info          # IOH metadata
├── IOHprofiler_f2.info
├── ...
├── certificates/               # OPOCH sidecar
│   ├── f1_d2_i1.json
│   ├── f1_d10_i1.json
│   └── ...
└── meta.json                   # Run metadata
```

## Running BBOB with IOH

```bash
# Full BBOB suite, d=2,10,20, budget=100k evals
python scripts/run_ioh_bbob.py \
    --dims 2 10 20 \
    --budget 100000 \
    --functions 1-24 \
    --instances 1-5 \
    --out results/ioh/OPOCH_BBOB

# Quick test run
python scripts/run_ioh_bbob.py \
    --dims 2 \
    --budget 1000 \
    --functions 1 \
    --instances 1 \
    --out results/ioh/test_run
```

## What the Runner Does

1. **Setup IOH logger:**
   ```python
   logger = ioh.logger.Analyzer(root=output_dir, algorithm_name="OPOCH")
   ```

2. **Wrap objective with IOH:**
   ```python
   problem = ioh.get_problem(fid, instance=iid, dimension=dim)
   problem.attach_logger(logger)
   ```

3. **Run OPOCH in anytime mode:**
   - Every evaluation is logged to IOH
   - Best-so-far is tracked and reported
   - Final certificate is saved to sidecar

4. **Handle undefined values:**
   ```python
   def safe_evaluate(x):
       try:
           val = problem(x)
           if not np.isfinite(val):
               return PENALTY_VALUE  # e.g., 1e10
           return val
       except:
           return PENALTY_VALUE
   ```

## Verification

```bash
# Verify IOH output is valid
python scripts/replay_verify_run.py results/ioh/OPOCH_BBOB/

# Check IOH log format
python -c "import ioh; ioh.logger.Analyzer.verify('results/ioh/OPOCH_BBOB')"
```

## Upload to IOHanalyzer

```bash
# Create zip for upload
cd results/ioh/OPOCH_BBOB
zip -r OPOCH_BBOB.zip data_* IOHprofiler_*.info

# Upload at https://iohanalyzer.liacs.nl/
# Select "Upload Data" → Choose OPOCH_BBOB.zip
```

## Performance Curves

IOHanalyzer will generate:
- **ECDF curves:** Fraction of problems solved vs evaluations
- **Fixed-target curves:** Evaluations to reach target vs target
- **Fixed-budget curves:** Best value reached at budget vs budget

## OPOCH-Specific Metrics (Sidecar)

In addition to IOH logs, we record:
- **Certification time:** Evaluations until UB - LB ≤ ε
- **Final verdict:** UNIQUE-OPT / UNSAT / Ω-GAP
- **Gap at budget:** If Ω-GAP, what was the remaining gap?

```json
{
  "function": 1,
  "dimension": 10,
  "instance": 1,
  "verdict": "UNIQUE-OPT",
  "evals_to_certify": 4523,
  "final_gap": 1.2e-7,
  "best_found": -418.9829,
  "optimum": -418.9829,
  "certificate_hash": "abc123..."
}
```

## Comparing with Other Optimizers

OPOCH is compared fairly:
- **Best-so-far curves:** Same as CMA-ES, DE, etc.
- **Certification curves:** Only OPOCH (others don't certify)

The comparison shows:
1. OPOCH finds good solutions (comparable to heuristics)
2. OPOCH additionally provides mathematical certificates

## Configuration Options

```python
# In run_ioh_bbob.py
config = OPOCHConfig(
    epsilon=1e-6,           # Certification tolerance
    max_evals=100000,       # IOH budget
    max_time=float('inf'),  # No time limit (IOH controls budget)
    log_frequency=100,      # Progress logging
)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| NaN in logs | Check `safe_evaluate` wrapper |
| Missing data files | Check function/dimension/instance ranges |
| Verification fails | Run `replay_verify_run.py` for details |
| Upload rejected | Ensure zip contains IOH format exactly |
