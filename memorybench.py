# MemoryBench - Benchmark wave-lattice memory: capacity and noise robustness
# Part of the Kapildhinwa2010-dev Toolkit.

"""MemoryBench - wave-lattice capacity benchmark.

Lineage: memory-layer repos. Stores bipolar patterns in a Hebbian
outer-product lattice with Hopfield-style sign updates and measures how
recall accuracy decays as patterns are added (capacity curve) and as cue
noise grows (noise curve).

Expected honest result: accuracy falls well before the theoretical
0.138*dim pattern count, and noise hurts quickly. This benchmark exists
to show that, not to hide it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["HebbianLattice", "capacity_curve", "noisy_recall_curve"]


def _bipolar(vector: np.ndarray) -> np.ndarray:
    v = np.sign(np.asarray(vector, dtype=np.float64).ravel())
    return np.where(v == 0, 1.0, v)


class HebbianLattice:
    """Outer-product pattern lattice with Hopfield-style recall."""

    def __init__(self, dim: int) -> None:
        if dim < 2:
            raise ValueError("dim must be >= 2")
        self.dim = int(dim)
        self.lattice = np.zeros((self.dim, self.dim), dtype=np.float64)
        self.patterns = []

    def store(self, pattern: np.ndarray) -> None:
        p = _bipolar(pattern)
        if p.size != self.dim:
            raise ValueError("pattern must have exactly {} entries".format(self.dim))
        self.lattice += np.outer(p, p)
        self.patterns.append(p)

    def recall(self, cue: np.ndarray, steps: int = 5) -> np.ndarray:
        """Hopfield-style recall: repeated sign of lattice times state."""
        if steps < 1:
            raise ValueError("steps must be >= 1")
        state = _bipolar(cue)
        if state.size != self.dim:
            raise ValueError("cue must have exactly {} entries".format(self.dim))
        for _ in range(steps):
            state = _bipolar(self.lattice @ state)
        return state


def capacity_curve(dim: int = 64, max_patterns: int = 32, trials: int = 3, seed: int = 0) -> list:
    """Recall accuracy (exact match) as patterns accumulate, 1..max_patterns."""
    if dim < 2 or max_patterns < 1 or trials < 1:
        raise ValueError("dim >= 2, max_patterns >= 1, trials >= 1 required")
    rng = np.random.default_rng(seed)
    curve = []
    for n in range(1, max_patterns + 1):
        accuracies = []
        for _ in range(trials):
            lattice = HebbianLattice(dim)
            patterns = [rng.standard_normal(dim) for _ in range(n)]
            for p in patterns:
                lattice.store(p)
            correct = 0
            for p in patterns:
                if np.array_equal(lattice.recall(p), lattice.patterns[0] * 0 + _bipolar(p)):
                    correct += 1
            accuracies.append(correct / n)
        curve.append({"patterns": n, "recall_accuracy": float(np.mean(accuracies))})
    return curve


def noisy_recall_curve(dim: int = 64, n_patterns: int = 5,
                       noise_levels=(0.0, 0.05, 0.1, 0.2, 0.3),
                       trials: int = 3, seed: int = 0) -> list:
    """Recall accuracy when a fraction of cue bits is flipped, per noise level."""
    if dim < 2 or n_patterns < 1 or trials < 1:
        raise ValueError("dim >= 2, n_patterns >= 1, trials >= 1 required")
    rng = np.random.default_rng(seed)
    curve = []
    for noise in noise_levels:
        if not 0.0 <= noise < 1.0:
            raise ValueError("noise levels must be in [0, 1)")
        accuracies = []
        for _ in range(trials):
            lattice = HebbianLattice(dim)
            patterns = [rng.standard_normal(dim) for _ in range(n_patterns)]
            for p in patterns:
                lattice.store(p)
            correct = 0
            for p in lattice.patterns:
                flips = rng.random(dim) < noise
                cue = np.where(flips, -p, p)
                if np.array_equal(lattice.recall(cue), p):
                    correct += 1
            accuracies.append(correct / n_patterns)
        curve.append({"noise": float(noise), "recall_accuracy": float(np.mean(accuracies))})
    return curve
