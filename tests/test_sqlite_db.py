from sdb.sqlite_db import save_to_sqlite, load_from_sqlite


def test_save_and_load(tmp_path):
    path = tmp_path / "cases.db"
    cases = [{"id": "1", "summary": "s", "full_text": "t"}]
    save_to_sqlite(str(path), cases)
    db = load_from_sqlite(str(path))
    assert db.get_case("1").full_text == "t"
