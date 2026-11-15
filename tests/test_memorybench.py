"""Tests for MemoryBench: lattice capacity and noise curves."""

import numpy as np
import pytest

from memorybench import HebbianLattice, capacity_curve, noisy_recall_curve


def test_single_pattern_recalls_exactly():
    lattice = HebbianLattice(dim=16)
    p = np.ones(16)
    p[:8] = -1.0
    lattice.store(p)
    assert np.array_equal(lattice.recall(p), p)


def test_store_validates():
    lattice = HebbianLattice(dim=8)
    with pytest.raises(ValueError):
        lattice.store(np.ones(4))
    with pytest.raises(ValueError):
        lattice.recall(np.ones(4))
    with pytest.raises(ValueError):
        lattice.recall(np.ones(8), steps=0)


def test_capacity_curve_shape_and_monotone_decline():
    curve = capacity_curve(dim=32, max_patterns=8, trials=2, seed=1)
    assert [c["patterns"] for c in curve] == list(range(1, 9))
    assert curve[0]["recall_accuracy"] == pytest.approx(1.0)
    assert curve[-1]["recall_accuracy"] <= curve[0]["recall_accuracy"]


def test_noisy_curve_accuracy_falls_with_noise():
    curve = noisy_recall_curve(dim=64, n_patterns=3, noise_levels=(0.0, 0.2), trials=2, seed=2)
    assert curve[0]["recall_accuracy"] >= curve[1]["recall_accuracy"]


def test_curve_validation():
    with pytest.raises(ValueError):
        capacity_curve(dim=1)
    with pytest.raises(ValueError):
        noisy_recall_curve(noise_levels=(1.0,))
