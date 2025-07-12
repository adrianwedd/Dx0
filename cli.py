"""Command line interface for running a demo diagnostic session."""

import argparse
import csv
import getpass
import json
import logging
import os
import sys
from pathlib import Path

import bcrypt
import yaml

from sdb import WeightedVoter  # noqa: F401 - exposed for tests
from sdb import (
    BudgetManager,
    CaseDatabase,
    load_cost_estimator,
    DiagnosisResult,
    Evaluator,
    Gatekeeper,
    Judge,
    LLMEngine,
    MetaPanel,
    MetricsDB,
    OllamaClient,
    OpenAIClient,
    Orchestrator,
    RuleEngine,
    VirtualPanel,
    batch_evaluate,
    bundle_to_case,
    configure_logging,
    diagnostic_report_to_case,
    load_from_sqlite,
    load_scores,
    ordered_tests_to_fhir,
    permutation_test,
    run_pipeline,
    start_metrics_server,
    transcript_to_fhir,
)
from sdb.config import load_settings, settings
from sdb import token
from sdb.token import TOKEN_PATH

_ = WeightedVoter


def _load_weights(arg: str | None) -> dict[str, float] | None:
    """Return a mapping from run ID to vote weight."""

    if arg is None:
        return None
    try:
        if os.path.exists(arg):
            with open(arg, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return json.loads(arg)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid vote weights: {exc}") from exc


def _load_weights_file(path: str | None) -> dict[str, float] | None:
    """Return a mapping from run ID to weight loaded from ``path``."""

    if path is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            if path.endswith((".yaml", ".yml")):
                return yaml.safe_load(fh) or {}
            return json.load(fh)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise SystemExit(f"Invalid weights file: {exc}") from exc


def _load_persona_models(arg: str | None) -> dict[str, str] | None:
    """Return a mapping from persona name to model."""

    if arg is None:
        return None
    try:
        if os.path.exists(arg):
            with open(arg, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return json.loads(arg)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid persona models: {exc}") from exc


def _load_json_or_path(arg: str | None) -> dict[str, object] | None:
    """Return a dictionary loaded from ``arg`` if provided."""

    if arg is None:
        return None
    try:
        if os.path.exists(arg):
            with open(arg, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return json.loads(arg)
    except json.JSONDecodeError as exc:  # pragma: no cover - arg validated
        raise SystemExit(f"Invalid metadata: {exc}") from exc


def _load_case_dicts(path: str) -> list[dict[str, object]]:
    """Return a list of case dicts from ``path``."""

    if os.path.isdir(path):
        cases = []
        for cid in sorted(os.listdir(path)):
            summary_file = os.path.join(path, cid, "summary.txt")
            full_file = os.path.join(path, cid, "full.txt")
            if not os.path.isfile(summary_file) or not os.path.isfile(full_file):
                continue
            with open(summary_file, "r", encoding="utf-8") as sf:
                summary = sf.read().strip()
            with open(full_file, "r", encoding="utf-8") as ff:
                full = ff.read().strip()
            cases.append({"id": cid, "summary": summary, "full_text": full})
        return cases
    if path.endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        return list(data.values())
    return data


def stats_main(argv: list[str]) -> None:
    """Run a permutation test on two result CSV files."""

    parser = argparse.ArgumentParser(description="Run significance test")
    parser.add_argument("baseline", help="CSV file for baseline results")
    parser.add_argument("variant", help="CSV file for variant results")
    parser.add_argument("--config", default=None, help="YAML settings file")
    parser.add_argument(
        "--column",
        default="score",
        help="CSV column containing numeric scores",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1000,
        help="Number of permutations",
    )
    args = parser.parse_args(argv)
    load_settings(args.config)

    a = load_scores(args.baseline, args.column)
    b = load_scores(args.variant, args.column)
    p = permutation_test(a, b, num_rounds=args.rounds)
    print(f"p-value: {p:.4f}")


def batch_eval_main(argv: list[str]) -> None:
    """Run evaluations for multiple cases concurrently."""

    parser = argparse.ArgumentParser(description="Batch evaluate cases")
    parser.add_argument("--config", default=None, help="YAML settings file")
    parser.add_argument("--db", help="Path to case JSON, CSV or directory")
    parser.add_argument("--db-sqlite", help="Path to case SQLite database")
    parser.add_argument("--rubric", required=True, help="Scoring rubric JSON")
    parser.add_argument("--costs", help="Test cost table CSV")
    parser.add_argument(
        "--cost-table",
        dest="cost_table",
        default=None,
        help="CSV file mapping test names to costs",
    )
    parser.add_argument(
        "--cost-estimator",
        dest="cost_estimator",
        default=None,
        help="Cost estimator plugin name",
    )
    parser.add_argument("--output", required=True, help="CSV file for results")
    parser.add_argument(
        "--results-db",
        default=None,
        help="SQLite database file to store metrics",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Number of concurrent sessions",
    )
    parser.add_argument(
        "--correct-threshold",
        type=int,
        default=4,
        help="Judge score required for a correct diagnosis",
    )
    parser.add_argument(
        "--panel-engine",
        choices=["rule", "llm"],
        default="rule",
    )
    parser.add_argument(
        "--llm-provider", choices=["openai", "ollama"], default="openai"
    )
    parser.add_argument("--llm-model", default="gpt-4")
    parser.add_argument(
        "--persona-models",
        default=None,
        help="JSON string or file with persona to model mapping",
    )
    parser.add_argument(
        "--ollama-base-url",
        default="http://localhost:11434",
        help="Base URL for the Ollama server",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache LLM responses",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=128,
        help="Maximum number of responses to keep in the cache",
    )
    parser.add_argument(
        "--budget-limit",
        type=float,
        default=None,
        help="Maximum total spend allowed during the session",
    )
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument(
        "--mode",
        choices=["unconstrained", "budgeted", "question-only", "instant"],
        default="unconstrained",
    )
    parser.add_argument(
        "--vote-weights",
        default=None,
        help=("JSON string or path with run ID weights for ensemble voting"),
    )
    parser.add_argument(
        "--weights-file",
        default=None,
        help="JSON or YAML file mapping run IDs to vote weights",
    )
    semantic = parser.add_mutually_exclusive_group()
    semantic.add_argument(
        "--semantic-retrieval",
        dest="semantic",
        action="store_true",
    )
    semantic.add_argument(
        "--no-semantic-retrieval",
        dest="semantic",
        action="store_false",
    )
    parser.set_defaults(semantic=False)
    parser.add_argument(
        "--cross-encoder-model",
        default=None,
        help="Cross-encoder model name for semantic retrieval",
    )
    parser.add_argument(
        "--retrieval-backend",
        default=None,
        help="Retrieval plugin name to use when semantic retrieval is enabled",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_settings(args.config)
    if args.db is None and args.db_sqlite is None:
        args.db = cfg.case_db
        args.db_sqlite = cfg.case_db_sqlite
    if args.retrieval_backend is None:
        args.retrieval_backend = cfg.retrieval_backend
    if args.cost_estimator is None:
        args.cost_estimator = cfg.cost_estimator_plugin

    vote_weights = _load_weights_file(args.weights_file)
    if vote_weights is None:
        vote_weights = _load_weights(args.vote_weights)
    meta_panel = MetaPanel(weights=vote_weights)
    persona_models = _load_persona_models(args.persona_models) or cfg.persona_models
    metrics_db = MetricsDB(args.results_db) if args.results_db else None

    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    configure_logging(level)

    if args.db is None and args.db_sqlite is None:
        parser.error("--db or --db-sqlite is required")

    if args.db_sqlite:
        db = load_from_sqlite(args.db_sqlite, lazy=True)
    elif os.path.isdir(args.db):
        db = CaseDatabase.load_from_directory(args.db)
    elif args.db.endswith(".csv"):
        db = CaseDatabase.load_from_csv(args.db)
    else:
        db = CaseDatabase.load_from_json(args.db)

    with open(args.rubric, "r", encoding="utf-8") as fh:
        rubric = json.load(fh)

    cost_path = args.cost_table or args.costs
    if cost_path is None:
        parser.error("--cost-table or --costs is required")
    cost_estimator = load_cost_estimator(
        cost_path, plugin_name=args.cost_estimator or "csv"
    )
    judge = Judge(rubric)
    evaluator = Evaluator(
        judge,
        cost_estimator,
        correct_threshold=args.correct_threshold,
    )

    def run_case(case_id: str) -> dict[str, str]:
        gatekeeper = Gatekeeper(
            db,
            case_id,
            use_semantic_retrieval=args.semantic,
            cross_encoder_name=args.cross_encoder_model,
            retrieval_backend=args.retrieval_backend,
        )
        if args.panel_engine == "rule":
            engine = RuleEngine()
        else:
            cache_path = "llm_cache.jsonl" if args.cache else None
            if args.llm_provider == "ollama":
                client = OllamaClient(
                    base_url=args.ollama_base_url or settings.ollama_base_url,
                    cache_path=cache_path,
                    cache_size=args.cache_size,
                )
            else:
                client = OpenAIClient(
                    cache_path=cache_path,
                    cache_size=args.cache_size,
                )
            engine = LLMEngine(
                model=args.llm_model,
                client=client,
                persona_models=persona_models,
            )

        panel = VirtualPanel(decision_engine=engine)
        limit = args.budget if args.mode == "budgeted" else args.budget_limit
        budget_manager = BudgetManager(
            cost_estimator,
            budget=limit,
        )
        orch_kwargs = {}
        if args.mode == "question-only":
            orch_kwargs["question_only"] = True

        orchestrator = Orchestrator(
            panel,
            gatekeeper,
            budget_manager=budget_manager,
            **orch_kwargs,
        )
        turn = 0
        max_turns = 1 if args.mode == "instant" else 10
        while not orchestrator.finished and turn < max_turns:
            orchestrator.run_turn("")
            turn += 1

        diag_result = DiagnosisResult(
            orchestrator.final_diagnosis or "",
            1.0,
            run_id=case_id,
        )
        final_diag = meta_panel.synthesize([diag_result])
        truth = db.get_case(case_id).summary
        result = evaluator.evaluate(
            final_diag,
            truth,
            orchestrator.ordered_tests,
            visits=turn,
            duration=orchestrator.total_time,
        )
        if metrics_db is not None:
            metrics_db.record(case_id, result)
        return {
            "id": case_id,
            "total_cost": f"{result.total_cost:.2f}",
            "score": str(result.score),
            "correct": str(result.correct),
            "duration": f"{result.duration:.2f}",
        }

    results = batch_evaluate(
        db.cases,
        run_case,
        concurrency=args.concurrency,
    )

    results.sort(key=lambda r: r["id"])

    fieldnames = ["id", "total_cost", "score", "correct", "duration"]
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def fhir_export_main(argv: list[str]) -> None:
    """Serialize a transcript and ordered tests as a FHIR bundle."""

    parser = argparse.ArgumentParser(description="Export session data to FHIR")
    parser.add_argument("transcript", help="Path to JSON transcript file")
    parser.add_argument("tests", help="Path to JSON ordered tests file")
    parser.add_argument(
        "--patient-id",
        default="example",
        help="Patient identifier used in references",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="File path for the output bundle (defaults to stdout)",
    )
    args = parser.parse_args(argv)

    with open(args.transcript, "r", encoding="utf-8") as fh:
        transcript = json.load(fh)
    with open(args.tests, "r", encoding="utf-8") as fh:
        tests = json.load(fh)

    bundle = transcript_to_fhir(transcript, patient_id=args.patient_id)
    test_bundle = ordered_tests_to_fhir(tests, patient_id=args.patient_id)
    bundle["entry"].extend(test_bundle["entry"])

    output = json.dumps(bundle, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(output)
    else:
        print(output)


def export_fhir_main(argv: list[str]) -> None:
    """Convert a saved transcript to a FHIR bundle file."""

    parser = argparse.ArgumentParser(description="Export a transcript to FHIR")
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to JSON transcript file",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Destination for the generated bundle",
    )
    parser.add_argument(
        "--patient-id",
        default="example",
        help="Patient identifier used in references",
    )
    args = parser.parse_args(argv)

    with open(args.input, "r", encoding="utf-8") as fh:
        transcript = json.load(fh)

    bundle = transcript_to_fhir(transcript, patient_id=args.patient_id)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, indent=2)


def fhir_import_main(argv: list[str]) -> None:
    """Convert a FHIR bundle or DiagnosticReport to case JSON."""

    parser = argparse.ArgumentParser(description="Import case from FHIR")
    parser.add_argument("input", help="Path to FHIR JSON file")
    parser.add_argument(
        "--case-id",
        default="case_001",
        help="Identifier for the generated case",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="File path for the output case (defaults to stdout)",
    )
    args = parser.parse_args(argv)

    with open(args.input, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if data.get("resourceType") == "Bundle":
        case = bundle_to_case(data, case_id=args.case_id)
    else:
        case = diagnostic_report_to_case(data, case_id=args.case_id)

    output = json.dumps(case, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(output)
    else:
        print(output)


def annotate_case_main(argv: list[str]) -> None:
    """Add notes or test mappings to a case JSON file."""

    parser = argparse.ArgumentParser(description="Annotate a case")
    parser.add_argument("--config", default=None, help="YAML settings file")
    parser.add_argument("--db", help="Path to case JSON, CSV or directory")
    parser.add_argument("--db-sqlite", help="Path to case SQLite database")
    parser.add_argument("--case", required=True, help="Case identifier")
    parser.add_argument(
        "--notes",
        default=None,
        help="Free text notes or path to a text file",
    )
    parser.add_argument(
        "--test-mapping",
        default=None,
        help="JSON file mapping aliases to canonical test names",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=("Destination for the annotated case (defaults to annotations/<id>.json)"),
    )
    args = parser.parse_args(argv)

    cfg = load_settings(args.config)
    if args.db is None and args.db_sqlite is None:
        args.db = cfg.case_db
        args.db_sqlite = cfg.case_db_sqlite

    if args.db_sqlite:
        db = load_from_sqlite(args.db_sqlite)
    elif os.path.isdir(args.db):
        db = CaseDatabase.load_from_directory(args.db)
    elif args.db.endswith(".csv"):
        db = CaseDatabase.load_from_csv(args.db)
    else:
        db = CaseDatabase.load_from_json(args.db)

    case = db.get_case(args.case)
    data = {"id": case.id, "summary": case.summary, "full_text": case.full_text}

    if args.notes:
        if os.path.exists(args.notes):
            with open(args.notes, "r", encoding="utf-8") as fh:
                data["notes"] = fh.read().strip()
        else:
            data["notes"] = args.notes

    if args.test_mapping:
        with open(args.test_mapping, "r", encoding="utf-8") as fh:
            data["test_mappings"] = json.load(fh)

    out_dir = "annotations"
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.output or os.path.join(out_dir, f"{case.id}.json")
    if args.output:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def filter_cases_main(argv: list[str]) -> None:
    """Select cases by keywords or metadata."""

    parser = argparse.ArgumentParser(description="Filter cases")
    parser.add_argument("--config", default=None, help="YAML settings file")
    parser.add_argument("--db", help="Path to case JSON or CSV file")
    parser.add_argument("--db-sqlite", help="Path to case SQLite database")
    parser.add_argument(
        "--keywords", default=None, help="Comma separated keywords to match"
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="JSON string or file with key/value filters",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Destination JSON or CSV file for the filtered cases",
    )
    args = parser.parse_args(argv)

    cfg = load_settings(args.config)
    if args.db is None and args.db_sqlite is None:
        args.db = cfg.case_db
        args.db_sqlite = cfg.case_db_sqlite

    if args.db_sqlite:
        db = load_from_sqlite(args.db_sqlite)
        cases = [c.__dict__ for c in db.cases.values()]
    else:
        cases = _load_case_dicts(args.db)

    keywords = []
    if args.keywords:
        keywords = [kw.strip().lower() for kw in args.keywords.split(",") if kw.strip()]

    meta_filter = _load_json_or_path(args.metadata) or {}

    selected: list[dict[str, object]] = []
    for case in cases:
        text = f"{case.get('summary', '')} {case.get('full_text', '')}".lower()
        if keywords and not any(kw in text for kw in keywords):
            continue
        match = True
        for key, val in meta_filter.items():
            if str(case.get(key)) != str(val):
                match = False
                break
        if match:
            selected.append(case)

    if args.output.endswith(".csv"):
        fieldnames: list[str] = []
        for item in selected:
            for name in item.keys():
                if name not in fieldnames:
                    fieldnames.append(name)
        with open(args.output, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected)
    else:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(selected, fh, indent=2)


def login_main(argv: list[str]) -> None:
    """Authenticate with the API and store a session token."""

    parser = argparse.ArgumentParser(description="Login to Dx0 API")
    parser.add_argument("--api-url", default="http://localhost:8000/api/v1")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", default=None)
    parser.add_argument(
        "--token-file",
        default=str(TOKEN_PATH),
        help="Path to store the session token",
    )
    args = parser.parse_args(argv)

    pwd = args.password or getpass.getpass("Password: ")
    token.login(
        args.api_url,
        args.username,
        pwd,
        path=Path(args.token_file),
    )
    print(f"Token saved to {args.token_file}")


def manage_users_main(argv: list[str]) -> None:
    """Add, remove or list web UI users."""

    parser = argparse.ArgumentParser(description="Manage UI users")
    parser.add_argument(
        "--file",
        default=os.environ.get(
            "UI_USERS_FILE",
            str(Path(__file__).parent / "sdb" / "ui" / "users.yml"),
        ),
        help="Path to credentials YAML file",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    add_p = sub.add_parser("add", help="Add a user")
    add_p.add_argument("username")
    add_p.add_argument("--password", default=None, help="User password")
    add_p.add_argument("--group", default="default", help="User group")

    rem_p = sub.add_parser("remove", help="Remove a user")
    rem_p.add_argument("username")

    sub.add_parser("list", help="List users")

    args = parser.parse_args(argv)

    data = {"users": {}}
    if os.path.exists(args.file):
        with open(args.file, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {"users": {}}
    users = data.get("users", {})

    if args.cmd == "add":
        pwd = args.password or getpass.getpass("Password: ")
        hashed = bcrypt.hashpw(pwd.encode(), bcrypt.gensalt()).decode()
        users[args.username] = {"password": hashed, "group": args.group}
        data["users"] = users
        os.makedirs(os.path.dirname(args.file), exist_ok=True)
        with open(args.file, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh)
        print(f"Added user {args.username}")
    elif args.cmd == "remove":
        if users.pop(args.username, None) is not None:
            data["users"] = users
            with open(args.file, "w", encoding="utf-8") as fh:
                yaml.safe_dump(data, fh)
            print(f"Removed user {args.username}")
        else:
            print(f"User {args.username} not found", file=sys.stderr)
    else:  # list
        for name in sorted(users):
            print(name)


def main() -> None:
    """Run a demo diagnostic session using the virtual panel."""

    parser = argparse.ArgumentParser(
        description="Run a simple diagnostic session",
    )
    parser.add_argument("--config", default=None, help="YAML settings file")
    parser.add_argument(
        "--db",
        help="Path to case JSON, CSV or directory",
    )
    parser.add_argument(
        "--db-sqlite",
        help="Path to case SQLite database",
    )
    parser.add_argument("--case", help="Case identifier")
    parser.add_argument(
        "--rubric",
        help="Path to scoring rubric JSON",
    )
    parser.add_argument(
        "--costs",
        help="Path to test cost table CSV",
    )
    parser.add_argument(
        "--cost-table",
        dest="cost_table",
        default=None,
        help="CSV file mapping test names to costs",
    )
    parser.add_argument(
        "--cost-estimator",
        dest="cost_estimator",
        default=None,
        help="Cost estimator plugin name",
    )
    parser.add_argument(
        "--correct-threshold",
        type=int,
        default=4,
        help="Judge score required for a correct diagnosis",
    )
    parser.add_argument(
        "--panel-engine",
        choices=["rule", "llm"],
        default="rule",
        help="Decision engine to use for the panel",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "ollama"],
        default="openai",
        help="LLM provider for LLM engine",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4",
        help="Model name for LLM engine",
    )
    parser.add_argument(
        "--persona-models",
        default=None,
        help="JSON string or file with persona to model mapping",
    )
    parser.add_argument(
        "--ollama-base-url",
        default="http://localhost:11434",
        help="Base URL for the Ollama server",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache LLM responses",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=128,
        help="Maximum number of responses to keep in the cache",
    )
    parser.add_argument(
        "--budget-limit",
        type=float,
        default=None,
        help="Maximum total spend allowed during the session",
    )
    semantic = parser.add_mutually_exclusive_group()
    semantic.add_argument(
        "--semantic-retrieval",
        dest="semantic",
        action="store_true",
        help="Enable semantic retrieval for Gatekeeper",
    )
    semantic.add_argument(
        "--no-semantic-retrieval",
        dest="semantic",
        action="store_false",
        help="Disable semantic retrieval for Gatekeeper",
    )
    parser.set_defaults(semantic=False)
    parser.add_argument(
        "--cross-encoder-model",
        default=None,
        help="Cross-encoder model name for semantic retrieval",
    )
    parser.add_argument(
        "--retrieval-backend",
        default=None,
        help="Retrieval plugin name to use when semantic retrieval is enabled",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging noise",
    )
    parser.add_argument(
        "--convert", action="store_true", help="Convert raw cases to JSON"
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw_cases",
        help="Directory with raw text cases for conversion",
    )
    parser.add_argument(
        "--output-dir",
        default="data/sdbench/cases",
        help="Destination for converted JSON cases",
    )
    parser.add_argument(
        "--hidden-dir",
        default="data/sdbench/hidden_cases",
        help="Directory for held-out cases from 2024-2025",
    )
    parser.add_argument(
        "--export-sqlite",
        default=None,
        help="Path to SQLite file when using --convert",
    )
    parser.add_argument(
        "--results-db",
        default=None,
        help="SQLite database to store evaluation metrics",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Budget limit for Budgeted mode",
    )
    parser.add_argument(
        "--mode",
        choices=[
            "unconstrained",
            "budgeted",
            "question-only",
            "instant",
            "ensemble",
        ],
        default="unconstrained",
        help="Run mode",
    )
    parser.add_argument(
        "--vote-weights",
        default=None,
        help=("JSON string or path with run ID weights for ensemble voting"),
    )
    parser.add_argument(
        "--weights-file",
        default=None,
        help="JSON or YAML file mapping run IDs to vote weights",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=None,
        help="Port for Prometheus metrics server (default 8000)",
    )
    args = parser.parse_args()
    cfg = load_settings(args.config)
    if args.db is None and args.db_sqlite is None:
        args.db = cfg.case_db
        args.db_sqlite = cfg.case_db_sqlite
    if args.retrieval_backend is None:
        args.retrieval_backend = cfg.retrieval_backend

    vote_weights = _load_weights_file(args.weights_file)
    if vote_weights is None:
        vote_weights = _load_weights(args.vote_weights)
    meta_panel = MetaPanel(weights=vote_weights)
    persona_models = _load_persona_models(args.persona_models) or cfg.persona_models
    metrics_db = MetricsDB(args.results_db) if args.results_db else None

    start_metrics_server(args.metrics_port)

    if args.convert:
        run_pipeline(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            hidden_dir=args.hidden_dir,
            fetch=False,
            sqlite_path=args.export_sqlite,
        )
        return

    cost_path = args.cost_table or args.costs
    required = [args.case, args.rubric, cost_path]
    if args.db is None and args.db_sqlite is None:
        parser.error("--db or --db-sqlite is required for a session")
    if any(item is None for item in required):
        parser.error(
            "--case, --rubric and --cost-table/--costs are required for a session"
        )

    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    configure_logging(level)

    if args.db_sqlite:
        db = load_from_sqlite(args.db_sqlite, lazy=True)
    elif os.path.isdir(args.db):
        db = CaseDatabase.load_from_directory(args.db)
    elif args.db.endswith(".csv"):
        db = CaseDatabase.load_from_csv(args.db)
    else:
        db = CaseDatabase.load_from_json(args.db)

    gatekeeper = Gatekeeper(
        db,
        args.case,
        use_semantic_retrieval=args.semantic,
        cross_encoder_name=args.cross_encoder_model,
        retrieval_backend=args.retrieval_backend,
    )
    gatekeeper.register_test_result("complete blood count", "normal")

    cost_estimator = load_cost_estimator(
        cost_path, plugin_name=args.cost_estimator or "csv"
    )

    with open(args.rubric, "r", encoding="utf-8") as fh:
        rubric = json.load(fh)

    judge = Judge(rubric)
    evaluator = Evaluator(
        judge,
        cost_estimator,
        correct_threshold=args.correct_threshold,
    )

    if args.panel_engine == "rule":
        engine = RuleEngine()
    else:
        cache_path = "llm_cache.jsonl" if args.cache else None
        if args.llm_provider == "ollama":
            client = OllamaClient(
                base_url=args.ollama_base_url or settings.ollama_base_url,
                cache_path=cache_path,
                cache_size=args.cache_size,
            )
        else:
            client = OpenAIClient(
                cache_path=cache_path,
                cache_size=args.cache_size,
            )
        engine = LLMEngine(
            model=args.llm_model,
            client=client,
            persona_models=persona_models,
        )

    panel = VirtualPanel(decision_engine=engine)

    limit = args.budget if args.mode == "budgeted" else args.budget_limit
    budget_manager = BudgetManager(cost_estimator, budget=limit)

    orch_kwargs = {}
    if args.mode == "question-only":
        orch_kwargs["question_only"] = True

    orchestrator = Orchestrator(
        panel,
        gatekeeper,
        budget_manager=budget_manager,
        **orch_kwargs,
    )

    turn = 0
    max_turns = 1 if args.mode == "instant" else 10
    while not orchestrator.finished and turn < max_turns:
        response = orchestrator.run_turn("")
        print(f"Turn {turn + 1}: {response}")
        turn += 1

    diag_result = DiagnosisResult(
        orchestrator.final_diagnosis or "",
        1.0,
        run_id=args.case,
    )
    final_diag = meta_panel.synthesize([diag_result])
    truth = db.get_case(args.case).summary
    result = evaluator.evaluate(
        final_diag,
        truth,
        orchestrator.ordered_tests,
        visits=turn,
        duration=orchestrator.total_time,
    )
    if metrics_db is not None:
        metrics_db.record(args.case, result)

    print(f"Final diagnosis: {final_diag}")
    print(f"Total cost: ${result.total_cost:.2f}")
    print(f"Session score: {result.score}")
    print(f"Correct diagnosis: {result.correct}")
    print(f"Total time: {result.duration:.2f}s")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        stats_main(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "batch-eval":
        batch_eval_main(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "fhir-export":
        fhir_export_main(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "fhir-import":
        fhir_import_main(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "export-fhir":
        export_fhir_main(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "annotate-case":
        annotate_case_main(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "filter-cases":
        filter_cases_main(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "login":
        login_main(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "manage-users":
        manage_users_main(sys.argv[2:])
    else:
        main()
