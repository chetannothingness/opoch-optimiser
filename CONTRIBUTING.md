# Contributing to OPOCH Optimizer

We welcome contributions that improve the solver, add benchmark problems, or enhance documentation.

## What to Submit

### Bug Reports
- A minimal reproducible example (JSON instance or Python script)
- Expected vs actual output
- OPOCH version and Python version

### New Benchmark Problems
Submit a JSON instance with:
- Bounded domain
- Objective function (as expression DAG or callable)
- Constraints (optional)
- Expected tolerance ε
- Any known best value (optional)
- An explanation of why it's interesting (e.g., deceptive, equality-heavy, high-dimensional)

### Code Contributions
- Fork the repository
- Create a feature branch
- Ensure all tests pass: `pytest -v`
- Submit a pull request with a clear description

## What We Guarantee

All OPOCH outputs are one of:
- **UNIQUE-OPT**: Globally optimal within ε, with certificate
- **UNSAT**: Infeasible with certificate
- **Ω-GAP**: Exact remaining gap with next separator action

Every run produces replayable receipts.

## Development Setup

```bash
git clone https://github.com/yourusername/opoch-optimizer.git
cd opoch-optimizer
pip install -e ".[dev]"
pytest -v
```

## Code Style

- Type hints for all public functions
- Docstrings explaining mathematical foundations
- Deterministic behavior (no random seeds without explicit control)
- All floating-point comparisons use explicit tolerances

## Testing

Run the test suite:
```bash
pytest -v                          # All tests
pytest tests/test_interval.py -v   # Specific module
pytest --cov=opoch_optimizer       # With coverage
```

## Documentation

When adding features:
1. Update relevant docs/*.md files
2. Add docstrings to new functions/classes
3. Include examples in tests

## Questions?

Open an issue with the "question" label.
