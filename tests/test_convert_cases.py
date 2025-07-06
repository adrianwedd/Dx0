import json
import subprocess
import sys
from sdb.ingest.convert import convert_directory


def test_convert_directory(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    hidden_dir = tmp_path / "hidden"
    raw_dir.mkdir()
    (raw_dir / "case_001.txt").write_text("N Engl J Med. 2025 Jan;\n\nPara2")
    (raw_dir / "case_002.txt").write_text("N Engl J Med. 2023 Jan;\n\nSecond")
    written = convert_directory(str(raw_dir), str(out_dir), str(hidden_dir))
    assert len(written) == 1
    path = out_dir / "case_002.json"
    data = json.loads(path.read_text())
    assert data["id"] == "case_002"
    assert len(data["steps"]) == 2
    assert data["summary"]
    summary_file = out_dir / "case_002_summary.txt"
    assert summary_file.exists()
    assert (hidden_dir / "case_001.json").exists()
    assert (hidden_dir / "case_001_summary.txt").exists()


def test_cli_convert(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    hidden_dir = tmp_path / "hidden"
    raw_dir.mkdir()
    (raw_dir / "case_001.txt").write_text("P1 2025\n")
    cmd = [
        sys.executable,
        "cli.py",
        "--convert",
        "--raw-dir",
        str(raw_dir),
        "--output-dir",
        str(out_dir),
        "--hidden-dir",
        str(hidden_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert (hidden_dir / "case_001.json").exists()
    assert (hidden_dir / "case_001_summary.txt").exists()
