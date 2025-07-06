import json
import subprocess
import sys
from sdb.ingest.convert import convert_directory


def test_convert_directory(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    (raw_dir / "case_001.txt").write_text("Para1\n\nPara2")
    (raw_dir / "case_002.txt").write_text("First\n\nSecond")
    written = convert_directory(str(raw_dir), str(out_dir))
    assert len(written) == 2
    for idx in [1, 2]:
        path = out_dir / f"case_{idx:03d}.json"
        data = json.loads(path.read_text())
        assert data["id"] == f"case_{idx:03d}"
        assert len(data["steps"]) == 2
        assert data["summary"]


def test_cli_convert(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    (raw_dir / "case_001.txt").write_text("P1")
    cmd = [
        sys.executable,
        "cli.py",
        "--convert",
        "--raw-dir",
        str(raw_dir),
        "--output-dir",
        str(out_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert (out_dir / "case_001.json").exists()
