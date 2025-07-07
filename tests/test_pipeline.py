import json
from sdb.ingest.pipeline import run_pipeline


def test_run_pipeline(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    hidden_dir = tmp_path / "hidden"
    raw_dir.mkdir()
    # year 2025 goes to hidden
    (raw_dir / "case_001.txt").write_text("N Engl J Med. 2025 Jan;\n\nPara")
    run_pipeline(
        raw_dir=str(raw_dir),
        output_dir=str(out_dir),
        hidden_dir=str(hidden_dir),
        fetch=False,
    )
    assert (hidden_dir / "case_001.json").exists()
    data = json.loads((hidden_dir / "case_001.json").read_text())
    assert data["id"] == "case_001"


def test_run_pipeline_sqlite(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    db_file = tmp_path / "cases.db"
    raw_dir.mkdir()
    (raw_dir / "case_001.txt").write_text("N Engl J Med. 2023 Jan;\n\nPara")
    run_pipeline(
        raw_dir=str(raw_dir),
        output_dir=str(out_dir),
        hidden_dir=None,
        fetch=False,
        sqlite_path=str(db_file),
    )
    from sdb.sqlite_db import load_from_sqlite

    db = load_from_sqlite(str(db_file))
    assert db.get_case("case_001").summary
