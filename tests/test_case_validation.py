import csv
import json
from sdb.case_database import CaseDatabase


def test_json_validation(tmp_path):
    data = [
        {"id": "1", "summary": "s1", "full_text": "f1"},
        {"id": 2, "summary": "bad", "full_text": "bad"},
        {"summary": "s3", "full_text": "f3"},
    ]
    path = tmp_path / "cases.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    db = CaseDatabase.load_from_json(str(path))
    assert list(db.cases.keys()) == ["1"]


def test_csv_validation(tmp_path):
    rows = [
        {"id": "a", "summary": "s", "full_text": "f"},
        {"id": "", "summary": "x", "full_text": "y"},
        {"summary": "s2", "full_text": "z"},
    ]
    path = tmp_path / "cases.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "summary", "full_text"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    db = CaseDatabase.load_from_csv(str(path))
    assert list(db.cases.keys()) == ["a"]
