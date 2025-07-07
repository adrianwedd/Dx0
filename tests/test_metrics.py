import json
import csv
import sys

import cli
from sdb.metrics import start_metrics_server


def _prepare_args(tmp_path):
    cases = [{"id": "1", "summary": "x", "full_text": "y"}]
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
                "cpt_code": "1",
                "price": "1",
            }
        )

    return case_file, rubric_file, cost_file


def test_start_metrics_server_env(monkeypatch):
    port = {}

    def fake_start(port_arg):
        port["value"] = port_arg

    monkeypatch.setattr("sdb.metrics.start_http_server", fake_start)
    monkeypatch.setenv("SDB_METRICS_PORT", "9100")
    start_metrics_server(None)
    assert port["value"] == 9100


def test_cli_metrics_port_flag(tmp_path, monkeypatch):
    case_file, rubric_file, cost_file = _prepare_args(tmp_path)

    called = {}

    def fake_start(port):
        called["port"] = port

    monkeypatch.setattr("sdb.metrics.start_http_server", fake_start)
    argv = [
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
        "--metrics-port",
        "9200",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert called["port"] == 9200


def test_cli_metrics_port_env(tmp_path, monkeypatch):
    case_file, rubric_file, cost_file = _prepare_args(tmp_path)

    called = {}

    def fake_start(port):
        called["port"] = port

    monkeypatch.setattr("sdb.metrics.start_http_server", fake_start)
    monkeypatch.setenv("SDB_METRICS_PORT", "9300")
    argv = [
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
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert called["port"] == 9300
