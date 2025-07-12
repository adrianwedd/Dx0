import os
from sdb.ingest import pipeline as pl


def test_collect_new_cases(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "case_001.txt").write_text("PMID: 111\n")

    monkeypatch.setattr(
        pl, "fetch_case_pmids", lambda count=304: ["111", "222"]
    )

    async def fake_fetch(session, pmid):
        return f"PMID: {pmid}"

    monkeypatch.setattr(pl, "fetch_case_text_async", fake_fetch)

    paths = pl.collect_new_cases(str(raw_dir))
    assert len(paths) == 1
    assert os.path.basename(paths[0]) == "case_002.txt"
    assert (raw_dir / "case_002.txt").read_text().strip() == "PMID: 222"
