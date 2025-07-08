# WeightedVoter

The `WeightedVoter` class aggregates multiple `DiagnosisResult` objects by summing
`confidence × weight` for each distinct diagnosis. The final prediction is the
label with the highest total score.

A mapping from `run_id` to a numeric weight can be provided via the `weights`
argument. When omitted, every result contributes equally (weight `1.0`).

```python
from sdb.ensemble import DiagnosisResult, WeightedVoter

results = [
    DiagnosisResult(diagnosis="flu", confidence=0.6, run_id="A"),
    DiagnosisResult(diagnosis="cold", confidence=0.9, run_id="B"),
    DiagnosisResult(diagnosis="flu", confidence=0.4, run_id="C"),
]

weights = {"A": 1.5, "B": 1.0, "C": 0.5}

voter = WeightedVoter()
print(voter.vote(results, weights=weights))  # prints "flu"
```

In this example the two "flu" results combine to a score of `1.1`, beating the
`0.9` score for "cold". Without the `weights` argument all votes would be
weighted equally and "cold" would have won by confidence alone.
