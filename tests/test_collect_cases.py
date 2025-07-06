import os

from scripts.collect_cases import save_case_text


def test_save_case_text(tmp_path):
    out = save_case_text(1, "Example", tmp_path)
    assert os.path.basename(out) == "case_001.txt"
    assert (tmp_path / "case_001.txt").read_text() == "Example\n"


def test_expected_case_count(tmp_path):
    for i in range(1, 305):
        save_case_text(i, f"Case {i}", tmp_path)
    files = list(tmp_path.glob("case_*.txt"))
    assert len(files) == 304
    for f in files:
        text = f.read_text()
        assert text.endswith("\n")
        assert text.strip().startswith("Case")
