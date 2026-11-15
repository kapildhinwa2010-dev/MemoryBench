# MemoryBench

[![CI](https://github.com/kapildhinwa2010-dev/MemoryBench/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/MemoryBench/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/MemoryBench/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/MemoryBench/actions/workflows/codeql.yml)

Benchmark wave-lattice memory: recall accuracy vs stored-pattern count and cue noise.

## What it does

Stores bipolar patterns in a Hebbian outer-product lattice with
Hopfield-style sign updates, then measures two honest decay curves:

- capacity_curve: exact-recall fraction as patterns accumulate (1..max)
- noisy_recall_curve: exact-recall fraction as a fraction of cue bits flip

    from memorybench import capacity_curve, noisy_recall_curve

    for row in capacity_curve(dim=64, max_patterns=16):
        print(row["patterns"], row["recall_accuracy"])

## API

- HebbianLattice(dim) - store(pattern), recall(cue, steps)
- capacity_curve(dim, max_patterns, trials, seed)
- noisy_recall_curve(dim, n_patterns, noise_levels, trials, seed)

## Honest results and limitations

- Expected result: accuracy decays well before the classical 0.138*dim
  capacity bound, and a 20-30 percent bit-flip rate already damages recall
  badly. The curves exist to show this, not to hide it.
- "Accuracy" means exact match with the stored bipolar pattern; partial
  overlap is not scored.
- Results vary with seed and dimension; curves are illustrative, not
  certified capacity constants.

## Status

Part of the kapildhinwa2010-dev toolkit experiments. CI and CodeQL run on
every push to main. License: MIT.
