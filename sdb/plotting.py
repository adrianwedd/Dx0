from __future__ import annotations

import csv
from typing import Iterable, Tuple

import matplotlib.pyplot as plt


def plot_accuracy_vs_cost(
    rows: Iterable[Tuple[float, float]],
    output_file: str | None = None,
) -> None:
    """Create a scatter plot of accuracy versus cost."""
    costs, accs = zip(*rows)
    plt.figure()
    plt.scatter(costs, accs)
    plt.xlabel("Total Cost")
    plt.ylabel("Accuracy")
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()


def load_results(path: str) -> list[Tuple[float, float]]:
    """Load cost/accuracy pairs from CSV."""
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append((float(row["cost"]), float(row["accuracy"])))
    return rows
