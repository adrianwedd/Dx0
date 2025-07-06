import csv
import tempfile

from sdb.case_database import CaseDatabase


def test_load_from_csv():
    rows = [
        {"id": "1", "summary": "s1", "full_text": "f1"},
        {"id": "2", "summary": "s2", "full_text": "f2"},
    ]
    with tempfile.NamedTemporaryFile("w", newline="", delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=["id", "summary", "full_text"])
        writer.writeheader()
        writer.writerows(rows)
        path = f.name
    db = CaseDatabase.load_from_csv(path)
    assert db.get_case("1").summary == "s1"
    assert db.get_case("2").full_text == "f2"
