import csv
import json
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


def test_load_from_json(tmp_path):
    cases = [
        {"id": "3", "summary": "sx", "full_text": "fx"},
    ]
    path = tmp_path / "cases.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cases, f)
    db = CaseDatabase.load_from_json(str(path))
    assert db.get_case("3").summary == "sx"


def test_load_from_directory(tmp_path):
    case_dir = tmp_path / "4"
    case_dir.mkdir()
    (case_dir / "summary.txt").write_text("ss")
    (case_dir / "full.txt").write_text("ff")
    db = CaseDatabase.load_from_directory(tmp_path)
    assert db.get_case("4").full_text == "ff"
