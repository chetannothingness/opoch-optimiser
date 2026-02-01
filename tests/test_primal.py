"""
Tests for Primal Search Components (Sobol, PhaseProbe, Portfolio)
"""

import numpy as np
import pytest
from opoch_optimizer.primal.sobol import SobolGenerator
from opoch_optimizer.primal.phase_probe import PhaseProbe, PhaseProbeResult
from opoch_optimizer.primal.portfolio import PrimalPortfolio, PrimalActType


class TestSobolGenerator:
    """Test Sobol sequence generator."""

    def test_creation(self):
        """Test generator creation."""
        bounds = [(-5, 5)] * 3
        gen = SobolGenerator(dimension=3, bounds=bounds)
        assert gen.dimension == 3

    def test_generate_points(self):
        """Test point generation."""
        bounds = [(-5, 5)] * 2
        gen = SobolGenerator(dimension=2, bounds=bounds)
        points = gen.generate(10)
        assert len(points) == 10
        for p in points:
            assert len(p.x) == 2
            # Points are scaled to bounds
            assert all(-5 <= xi <= 5 for xi in p.x)

    def test_generate_in_region(self):
        """Test generation within a specific region."""
        bounds = [(-10, 10)] * 2
        gen = SobolGenerator(dimension=2, bounds=bounds)
        region_lb = np.array([-1.0, -1.0])
        region_ub = np.array([1.0, 1.0])
        # Use a valid hex hash string
        region_hash = "abcd1234ef567890"
        points = gen.generate_in_region(10, region_lb, region_ub, region_hash)

        assert len(points) == 10
        for p in points:
            assert len(p.x) == 2
            assert all(-1 <= xi <= 1 for xi in p.x)
            assert p.region_hash == region_hash

    def test_deterministic(self):
        """Test that generation is deterministic."""
        bounds = [(-5, 5)] * 2
        gen1 = SobolGenerator(dimension=2, bounds=bounds, seed=42)
        gen2 = SobolGenerator(dimension=2, bounds=bounds, seed=42)

        points1 = gen1.generate(5)
        points2 = gen2.generate(5)

        for p1, p2 in zip(points1, points2):
            np.testing.assert_array_almost_equal(p1.x, p2.x)

    def test_reset(self):
        """Test reset functionality."""
        bounds = [(-5, 5)] * 2
        gen = SobolGenerator(dimension=2, bounds=bounds, seed=42)

        points1 = gen.generate(5)
        gen.reset()
        points2 = gen.generate(5)

        # After reset, should produce same sequence
        for p1, p2 in zip(points1, points2):
            np.testing.assert_array_almost_equal(p1.x, p2.x)


class TestPhaseProbe:
    """Test PhaseProbe for periodic function identification."""

    def test_creation(self):
        """Test PhaseProbe creation."""
        def f(x):
            return np.sum(x**2)

        bounds = [(-5, 5)] * 3
        probe = PhaseProbe(objective=f, dimension=3, bounds=bounds)
        assert probe.dimension == 3

    def test_detect_periodicity(self):
        """Test periodicity detection on Rastrigin-like function."""
        def f(x):
            return 10 + x[0]**2 - 10 * np.cos(2 * np.pi * x[0])

        bounds = [(-5.12, 5.12)]
        probe = PhaseProbe(objective=f, dimension=1, bounds=bounds)

        x0 = np.array([0.0])
        is_periodic, energy = probe.detect_periodicity(x0, dim=0, M=32)

        # Rastrigin should be detected as periodic
        assert is_periodic
        assert energy > 0.3

    def test_extract_phase(self):
        """Test phase extraction on shifted cosine."""
        shift = 0.25

        def f(x):
            diff = x[0] - shift
            return diff**2 - 10 * np.cos(2 * np.pi * diff)

        bounds = [(-5.12, 5.12)]
        probe = PhaseProbe(objective=f, dimension=1, bounds=bounds)

        x0 = np.array([0.0])
        estimated, confidence = probe.extract_phase(x0, dim=0, M=32)

        # Should have some reasonable confidence
        assert confidence > 0.1

    def test_identify_and_refine(self):
        """Test full identification and refinement."""
        np.random.seed(42)
        shift = np.array([0.5, -0.3])

        def f(x):
            diff = x - shift
            return 20 + np.sum(diff**2 - 10 * np.cos(2 * np.pi * diff))

        bounds = [(-5.12, 5.12)] * 2
        probe = PhaseProbe(objective=f, dimension=2, bounds=bounds)
        result = probe.identify_and_refine(M=32)

        # Should return a PhaseProbeResult
        assert isinstance(result, PhaseProbeResult)
        assert len(result.candidate_x) == 2
        assert result.total_evals > 0

    def test_reset(self):
        """Test reset functionality."""
        def f(x):
            return np.sum(x**2)

        bounds = [(-5, 5)] * 2
        probe = PhaseProbe(objective=f, dimension=2, bounds=bounds)

        # Do some evaluations
        probe.detect_periodicity(np.array([0.0, 0.0]), dim=0, M=16)
        assert probe.total_evals > 0

        # Reset should clear counter
        probe.reset()
        assert probe.total_evals == 0


class TestPrimalPortfolio:
    """Test primal portfolio for coordinated search."""

    def test_creation(self):
        """Test portfolio creation."""
        def f(x):
            return np.sum(x**2)

        bounds = [(-5, 5)] * 3
        portfolio = PrimalPortfolio(dimension=3, bounds=bounds, objective=f)
        assert portfolio.dimension == 3

    def test_global_sobol_seeding(self):
        """Test global Sobol sampling."""
        def f(x):
            return np.sum(x**2)

        bounds = [(-5, 5)] * 2
        portfolio = PrimalPortfolio(dimension=2, bounds=bounds, objective=f)
        act = portfolio.global_sobol_seeding(n_points=20)

        assert act.act_type == PrimalActType.GLOBAL_SOBOL
        assert act.points_evaluated == 20
        assert act.x is not None
        assert act.f >= 0  # Sum of squares is non-negative

    def test_local_refine(self):
        """Test local refinement."""
        def f(x):
            return np.sum((x - 1)**2)  # Optimum at (1, 1)

        bounds = [(-5, 5)] * 2
        portfolio = PrimalPortfolio(dimension=2, bounds=bounds, objective=f)

        x0 = np.array([0.5, 0.5])
        act = portfolio.local_refine(x0, maxiter=50)

        assert act.act_type == PrimalActType.LOCAL_REFINE
        # Should get closer to (1, 1)
        if act.x is not None:
            assert np.linalg.norm(act.x - np.array([1.0, 1.0])) < 1.0

    def test_best_tracking(self):
        """Test that portfolio tracks best solution."""
        def f(x):
            return np.sum((x - 1)**2)  # Optimum at (1, 1)

        bounds = [(-5, 5)] * 2
        portfolio = PrimalPortfolio(dimension=2, bounds=bounds, objective=f)
        portfolio.global_sobol_seeding(n_points=100)

        # Best should be reasonably close to (1, 1)
        best_f, best_x = portfolio.get_upper_bound()
        assert best_x is not None
        assert best_f < 50  # Should find something decent

    def test_region_sobol(self):
        """Test region-focused Sobol sampling."""
        def f(x):
            return np.sum(x**2)

        bounds = [(-10, 10)] * 2
        portfolio = PrimalPortfolio(dimension=2, bounds=bounds, objective=f)

        region_lb = np.array([-1.0, -1.0])
        region_ub = np.array([1.0, 1.0])
        # Use a valid hex hash string
        region_hash = "abcd1234ef567890"
        act = portfolio.region_sobol(region_lb, region_ub, n_points=10, region_hash=region_hash)

        assert act.act_type == PrimalActType.REGION_SOBOL
        assert act.region_hash == region_hash
        assert act.points_evaluated == 10

    def test_phase_probe_identification(self):
        """Test PhaseProbe integration."""
        np.random.seed(42)
        shift = np.array([0.5, 0.5])

        def f(x):
            diff = x - shift
            return 20 + np.sum(diff**2 - 10 * np.cos(2 * np.pi * diff))

        bounds = [(-5.12, 5.12)] * 2
        portfolio = PrimalPortfolio(dimension=2, bounds=bounds, objective=f)

        success, act = portfolio.phase_probe_identification()

        # Should detect periodic structure
        if success:
            assert act is not None
            assert act.act_type == PrimalActType.PHASE_PROBE


class TestIntegration:
    """Integration tests for primal components."""

    def test_full_exploration(self):
        """Test complete exploration strategy."""
        np.random.seed(42)

        def f(x):
            return np.sum(x**2)

        bounds = [(-5, 5)] * 3
        portfolio = PrimalPortfolio(dimension=3, bounds=bounds, objective=f)

        act = portfolio.full_exploration(total_budget=500)

        # Should find a good solution
        assert act.x is not None
        assert act.f < 5  # Reasonably close to 0

    def test_eval_tracker(self):
        """Test evaluation tracking callback."""
        evals = []

        def tracker(f):
            evals.append(f)

        def f(x):
            return np.sum(x**2)

        bounds = [(-5, 5)] * 2
        portfolio = PrimalPortfolio(
            dimension=2, bounds=bounds, objective=f, eval_tracker=tracker
        )
        portfolio.global_sobol_seeding(n_points=10)

        assert len(evals) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
