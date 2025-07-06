import csv

from sdb.statistics import load_scores, permutation_test


def test_load_scores(tmp_path):
    path = tmp_path / "scores.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["score"])
        writer.writeheader()
        writer.writerow({"score": "1"})
        writer.writerow({"score": "0"})
    scores = load_scores(str(path))
    assert scores == [1.0, 0.0]


def test_permutation_test_significant():
    a = [1, 1, 1, 1, 1]
    b = [0, 0, 0, 0, 0]
    p = permutation_test(a, b, num_rounds=1000, seed=0)
    assert p < 0.01
