import json
import gc
import os

import psutil

from sdb.case_database import SQLiteCaseDatabase
from sdb.sqlite_db import save_to_sqlite, load_from_sqlite
from scripts import migrate_to_sqlite as m2s


def test_sqlite_case_database(tmp_path):
    path = tmp_path / "cases.db"
    cases = [{"id": "1", "summary": "s1", "full_text": "f1"}]
    save_to_sqlite(str(path), cases)
    db = SQLiteCaseDatabase(str(path))
    assert db.get_case("1").full_text == "f1"
    assert "cases" not in db.__dict__


def test_migrate_to_sqlite(tmp_path):
    data = [{"id": "2", "summary": "s", "full_text": "t"}]
    json_file = tmp_path / "cases.json"
    json_file.write_text(json.dumps(data))
    db_file = tmp_path / "out.db"
    m2s.main([str(json_file), str(db_file)])
    db = load_from_sqlite(str(db_file))
    assert db.get_case("2").summary == "s"


def test_lazy_memory_usage(tmp_path):
    big_cases = [
        {"id": str(i), "summary": "s", "full_text": "x" * 10000}
        for i in range(50)
    ]
    db_path = tmp_path / "big.db"
    save_to_sqlite(str(db_path), big_cases)
    del big_cases
    gc.collect()
    proc = psutil.Process(os.getpid())
    before = proc.memory_info().rss
    db = SQLiteCaseDatabase(str(db_path))
    _ = db.get_case("0")
    after = proc.memory_info().rss
    assert after - before < 1024 * 1024
