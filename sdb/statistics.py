"""Statistical utilities for evaluating SDBench results."""

from __future__ import annotations

import csv
import random
import statistics
from typing import Sequence


def load_scores(path: str, column: str = "score") -> list[float]:
    """Load numeric scores from a CSV file.

    Parameters
    ----------
    path:
        Path to a CSV file with result metrics.
    column:
        Name of the column containing the score of interest.

    Returns
    -------
    list[float]
        Sequence of scores parsed from the CSV.
    """

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if column not in reader.fieldnames:
            raise ValueError(f"Column '{column}' not found in {path}")
        return [float(row[column]) for row in reader]


def permutation_test(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    *,
    num_rounds: int = 10000,
    seed: int | None = None,
) -> float:
    """Perform a two-sided permutation test on the difference of means.

    Parameters
    ----------
    sample_a:
        First set of measurements.
    sample_b:
        Second set of measurements.
    num_rounds:
        Number of permutations to sample.
    seed:
        Optional seed for deterministic shuffling.

    Returns
    -------
    float
        Estimated p-value for the null hypothesis that both samples come
        from the same distribution.
    """

    rng = random.Random(seed)
    obs = abs(statistics.mean(sample_a) - statistics.mean(sample_b))
    combined = list(sample_a) + list(sample_b)
    n_a = len(sample_a)
    count = 0

    for _ in range(num_rounds):
        rng.shuffle(combined)
        a = combined[:n_a]
        b = combined[n_a:]
        diff = abs(statistics.mean(a) - statistics.mean(b))
        if diff >= obs:
            count += 1

    return count / num_rounds
