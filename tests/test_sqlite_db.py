from sdb.sqlite_db import (
    save_to_sqlite,
    load_from_sqlite,
    iter_sqlite_cases,
)


def test_save_and_load(tmp_path):
    path = tmp_path / "cases.db"
    cases = [{"id": "1", "summary": "s", "full_text": "t"}]
    save_to_sqlite(str(path), cases)
    db = load_from_sqlite(str(path))
    assert db.get_case("1").full_text == "t"


def test_iter_sqlite_cases(tmp_path):
    path = tmp_path / "cases.db"
    cases = [
        {"id": "1", "summary": "s1", "full_text": "t1"},
        {"id": "2", "summary": "s2", "full_text": "t2"},
    ]
    save_to_sqlite(str(path), cases)
    ids = [c.id for c in iter_sqlite_cases(str(path))]
    assert ids == ["1", "2"]
