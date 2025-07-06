from sdb.ensemble import (
    DiagnosisResult,
    WeightedVoter,
    MetaPanel,
    cost_adjusted_selection,
)


def test_weighted_voter_selects_highest_score():
    voter = WeightedVoter()
    results = [
        DiagnosisResult("flu", 0.6),
        DiagnosisResult("cold", 0.4),
        DiagnosisResult("flu", 0.3),
    ]
    assert voter.vote(results) == "flu"


def test_meta_panel_uses_voter():
    mp = MetaPanel()
    results = [DiagnosisResult("a", 1.0), DiagnosisResult("b", 2.0)]
    assert mp.synthesize(results) == "b"


def test_cost_adjusted_selection_penalizes_cost():
    results = [
        DiagnosisResult("x", confidence=1.0, cost=100.0),
        DiagnosisResult("y", confidence=0.9, cost=10.0),
    ]
    choice = cost_adjusted_selection(results, cost_weight=0.01)
    assert choice == "y"
