import json
import csv
import subprocess
import sys


def test_cli_outputs_final_results(tmp_path):
    cases = [{"id": "1", "summary": "viral infection", "full_text": "cough"}]
    case_file = tmp_path / "cases.json"
    with open(case_file, "w", encoding="utf-8") as f:
        json.dump(cases, f)

    rubric_file = tmp_path / "rubric.json"
    with open(rubric_file, "w", encoding="utf-8") as f:
        json.dump({}, f)

    cost_file = tmp_path / "costs.csv"
    with open(cost_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["test_name", "cpt_code", "price"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "test_name": "complete blood count",
                "cpt_code": "100",
                "price": "10",
            }
        )

    cmd = [
        sys.executable,
        "cli.py",
        "--db",
        str(case_file),
        "--case",
        "1",
        "--rubric",
        str(rubric_file),
        "--costs",
        str(cost_file),
        "--panel-engine",
        "rule",
        "--llm-model",
        "gpt-4",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    assert "Final diagnosis" in result.stdout
    assert "Total cost" in result.stdout
    assert "Session score" in result.stdout
    assert "Total time" in result.stdout


def test_cli_flag_parsing(tmp_path):
    cases = [{"id": "1", "summary": "s", "full_text": "t"}]
    case_file = tmp_path / "cases.json"
    with open(case_file, "w", encoding="utf-8") as f:
        json.dump(cases, f)

    rubric_file = tmp_path / "r.json"
    with open(rubric_file, "w", encoding="utf-8") as f:
        json.dump({}, f)

    cost_file = tmp_path / "c.csv"
    with open(cost_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["test_name", "cpt_code", "price"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "test_name": "complete blood count",
                "cpt_code": "1",
                "price": "1",
            }
        )

    cmd = [
        sys.executable,
        "cli.py",
        "--db",
        str(case_file),
        "--case",
        "1",
        "--rubric",
        str(rubric_file),
        "--costs",
        str(cost_file),
        "--panel-engine",
        "llm",
        "--llm-provider",
        "openai",
        "--llm-model",
        "turbo",
        "--quiet",
        "--semantic-retrieval",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0


def test_cli_stats_command(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    for path, v1, v2 in [(a, 1, 0), (b, 0, 0)]:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["score"])
            writer.writeheader()
            writer.writerow({"score": str(v1)})
            writer.writerow({"score": str(v2)})

    cmd = [
        sys.executable,
        "cli.py",
        "stats",
        str(a),
        str(b),
        "--column",
        "score",
        "--rounds",
        "100",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert "p-value" in result.stdout
