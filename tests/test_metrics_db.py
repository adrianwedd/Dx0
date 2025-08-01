from sdb.services.metrics_db import MetricsDB
from sdb.evaluation import SessionResult


def test_metrics_db_records(tmp_path):
    db_path = tmp_path / "m.db"
    db = MetricsDB(str(db_path))
    result = SessionResult(total_cost=10.0, score=5, correct=True, duration=1.0)
    db.record("c1", result)
    import sqlite3

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT case_id, total_cost, score, correct, duration FROM results")
    row = cur.fetchone()
    conn.close()
    assert row == ("c1", 10.0, 5, 1, 1.0)


def test_metrics_db_bulk_record_success_and_error(tmp_path):
    db_path = tmp_path / "m.db"
    db = MetricsDB(str(db_path))
    res1 = SessionResult(total_cost=5.0, score=5, correct=True, duration=2.0)
    res2 = SessionResult(total_cost=7.0, score=2, correct=False, duration=1.0)
    db.bulk_record([("a", res1), ("b", res2)])

    import sqlite3

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT case_id, score, correct FROM results ORDER BY case_id"
    )
    rows = cur.fetchall()
    conn.close()
    assert rows == [("a", 5, 1), ("b", 2, 0)]
