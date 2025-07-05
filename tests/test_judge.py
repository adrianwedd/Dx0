from sdb.judge import Judge


def test_judge_scoring():
    rubric = {"exact_threshold": 0.9, "partial_threshold": 0.6}
    j = Judge(rubric)
    res = j.evaluate("Influenza", "Influenza")
    assert res.score == 5
    res = j.evaluate("Influenza A", "Influenza")
    assert res.score >= 4
    res = j.evaluate("Cold", "Influenza")
    assert res.score <= 2
