import json
import sdb.retrieval as retrieval
from sdb.case_database import CaseDatabase
from scripts import retrieval_eval as rev


def test_evaluate_retrieval(tmp_path, monkeypatch):
    monkeypatch.setattr(retrieval, "FAISS_AVAILABLE", False, raising=False)
    monkeypatch.setattr(retrieval, "TRANSFORMERS_AVAILABLE", False, raising=False)
    cases = [
        {"id": "1", "summary": "cough", "full_text": "patient cough"},
        {"id": "2", "summary": "fever", "full_text": "high fever"},
    ]
    path = tmp_path / "cases.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cases, fh)
    db = CaseDatabase.load_from_json(str(path))
    recall, mrr = rev.evaluate_retrieval(db, top_k=1)
    assert recall == 1.0
    assert mrr == 1.0
