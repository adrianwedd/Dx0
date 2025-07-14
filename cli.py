"""Command line interface for running a demo diagnostic session."""

from enum import Enum

import typer
import csv
import getpass
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

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
    HFLocalClient,
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


class PanelEngine(str, Enum):
    """Decision engine options for the virtual panel."""

    RULE = "rule"
    LLM = "llm"


class Verbosity(str, Enum):
    """Logging verbosity levels."""

    QUIET = "quiet"
    INFO = "info"
    DEBUG = "debug"


class LLMProvider(str, Enum):
    """Available LLM service providers."""

    OPENAI = "openai"
    OLLAMA = "ollama"
    HF_LOCAL = "hf-local"


app = typer.Typer(help="Dx0 command line interface")


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


@app.command()
def stats(
    baseline: str,
    variant: str,
    config: str | None = typer.Option(None, help="YAML settings file"),
    column: str = typer.Option("score", help="CSV column containing numeric scores"),
    rounds: int = typer.Option(1000, help="Number of permutations"),
) -> None:
    """Run a permutation test on two result CSV files."""

    load_settings(config)

    a = load_scores(baseline, column)
    b = load_scores(variant, column)
    p = permutation_test(a, b, num_rounds=rounds)
    print(f"p-value: {p:.4f}")


@app.command("batch-eval")
def batch_eval(
    config: str | None = typer.Option(None, help="YAML settings file"),
    db: str | None = typer.Option(None, help="Path to case JSON, CSV or directory"),
    db_sqlite: str | None = typer.Option(None, help="Path to case SQLite database"),
    rubric: str = typer.Option(..., help="Scoring rubric JSON"),
    costs: str | None = typer.Option(None, help="Test cost table CSV"),
    cost_table: str | None = typer.Option(None, help="CSV file mapping test names to costs"),
    cost_estimator: str | None = typer.Option(None, help="Cost estimator plugin name"),
    output: str = typer.Option(..., help="CSV file for results"),
    results_db: str | None = typer.Option(None, help="SQLite database file to store metrics"),
    concurrency: int = typer.Option(2, help="Number of concurrent sessions"),
    correct_threshold: int = typer.Option(4, help="Judge score required for a correct diagnosis"),
    panel_engine: PanelEngine = typer.Option(PanelEngine.RULE, help="Decision engine"),
    llm_provider: LLMProvider = typer.Option(LLMProvider.OPENAI, help="LLM provider"),
    llm_model: str = typer.Option("gpt-4", help="Model name"),
    hf_model: str | None = typer.Option(None, help="Local HF model path"),
    persona_models: str | None = typer.Option(None, help="JSON string or file with persona to model mapping"),
    ollama_base_url: str = typer.Option("http://localhost:11434", help="Base URL for the Ollama server"),
    cache: bool = typer.Option(False, help="Cache LLM responses"),
    cache_size: int = typer.Option(128, help="Maximum number of responses to keep in the cache"),
    budget_limit: float | None = typer.Option(None, help="Maximum total spend allowed during the session"),
    budget: float | None = typer.Option(None, help="Budget limit for budgeted mode"),
    mode: str = typer.Option("unconstrained", help="Run mode"),
    vote_weights: str | None = typer.Option(None, help="JSON string or path with run ID weights for ensemble voting"),
    weights_file: str | None = typer.Option(None, help="JSON or YAML file mapping run IDs to vote weights"),
    semantic: bool = typer.Option(
        False,
        "--semantic",
        "--semantic-retrieval",
        help="Enable semantic retrieval",
    ),
    cross_encoder_model: str | None = typer.Option(None, help="Cross-encoder model name for semantic retrieval"),
    retrieval_backend: str | None = typer.Option(None, help="Retrieval plugin name"),
    verbosity: Verbosity = typer.Option(Verbosity.INFO, help="Logging verbosity"),
) -> None:
    """Run evaluations for multiple cases concurrently."""

    if isinstance(config, list):
        # Called programmatically with a list of CLI arguments
        return main(["batch-eval", *config])

    args = SimpleNamespace(
        config=config,
        db=db,
        db_sqlite=db_sqlite,
        rubric=rubric,
        costs=costs,
        cost_table=cost_table,
        cost_estimator=cost_estimator,
        output=output,
        results_db=results_db,
        concurrency=concurrency,
        correct_threshold=correct_threshold,
        panel_engine=panel_engine.value,
        llm_provider=llm_provider,
        llm_model=llm_model,
        hf_model=hf_model,
        persona_models=persona_models,
        ollama_base_url=ollama_base_url,
        cache=cache,
        cache_size=cache_size,
        budget_limit=budget_limit,
        budget=budget,
        mode=mode,
        vote_weights=vote_weights,
        weights_file=weights_file,
        semantic=semantic,
        cross_encoder_model=cross_encoder_model,
        retrieval_backend=retrieval_backend,
        verbose=verbosity == Verbosity.DEBUG,
        quiet=verbosity == Verbosity.QUIET,
    )
    cfg = load_settings(args.config)
    if args.db is None and args.db_sqlite is None:
        args.db = cfg.case_db
        args.db_sqlite = cfg.case_db_sqlite
    if args.retrieval_backend is None:
        args.retrieval_backend = cfg.retrieval_backend
    if args.hf_model is None:
        args.hf_model = cfg.hf_model
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
        raise SystemExit("--db or --db-sqlite is required")

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
        raise SystemExit("--cost-table or --costs is required")
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
            if args.llm_provider == LLMProvider.OLLAMA:
                client = OllamaClient(
                    base_url=args.ollama_base_url or settings.ollama_base_url,
                    cache_path=cache_path,
                    cache_size=args.cache_size,
                )
            elif args.llm_provider == LLMProvider.HF_LOCAL:
                client = HFLocalClient(
                    model_path=args.hf_model or settings.hf_model,
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


@app.command("fhir-export")
def fhir_export(
    transcript: str,
    tests: str,
    patient_id: str = typer.Option("example", help="Patient identifier used in references"),
    output: str | None = typer.Option(None, help="File path for the output bundle (defaults to stdout)"),
) -> None:
    """Serialize a transcript and ordered tests as a FHIR bundle."""

    with open(transcript, "r", encoding="utf-8") as fh:
        transcript_data = json.load(fh)
    with open(tests, "r", encoding="utf-8") as fh:
        tests_data = json.load(fh)

    bundle = transcript_to_fhir(transcript_data, patient_id=patient_id)
    test_bundle = ordered_tests_to_fhir(tests_data, patient_id=patient_id)
    bundle["entry"].extend(test_bundle["entry"])

    output_text = json.dumps(bundle, indent=2)
    if output:
        with open(output, "w", encoding="utf-8") as fh:
            fh.write(output_text)
    else:
        print(output_text)


@app.command("export-fhir")
def export_fhir(
    input: str = typer.Option(
        ..., "-i", "--input", help="Path to JSON transcript file"
    ),
    output: str = typer.Option(
        ..., "-o", "--output", help="Destination for the generated bundle"
    ),
    patient_id: str = typer.Option("example", help="Patient identifier used in references"),
) -> None:
    """Convert a saved transcript to a FHIR bundle file."""

    with open(input, "r", encoding="utf-8") as fh:
        transcript = json.load(fh)

    bundle = transcript_to_fhir(transcript, patient_id=patient_id)

    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, indent=2)


@app.command("fhir-import")
def fhir_import(
    input: str,
    case_id: str = typer.Option("case_001", help="Identifier for the generated case"),
    output: str | None = typer.Option(None, "-o", help="File path for the output case (defaults to stdout)"),
) -> None:
    """Convert a FHIR bundle or DiagnosticReport to case JSON."""

    with open(input, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if data.get("resourceType") == "Bundle":
        case = bundle_to_case(data, case_id=case_id)
    else:
        case = diagnostic_report_to_case(data, case_id=case_id)

    output_text = json.dumps(case, indent=2)
    if output:
        with open(output, "w", encoding="utf-8") as fh:
            fh.write(output_text)
    else:
        print(output_text)


@app.command("annotate-case")
def annotate_case(
    case: str = typer.Option(..., help="Case identifier"),
    config: str | None = typer.Option(None, help="YAML settings file"),
    db: str | None = typer.Option(None, help="Path to case JSON, CSV or directory"),
    db_sqlite: str | None = typer.Option(None, help="Path to case SQLite database"),
    notes: str | None = typer.Option(None, help="Free text notes or path to a text file"),
    test_mapping: str | None = typer.Option(None, help="JSON file mapping aliases to canonical test names"),
    output: str | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Destination for the annotated case (defaults to annotations/<id>.json)",
    ),
) -> None:
    """Add notes or test mappings to a case JSON file."""

    args = SimpleNamespace(
        case=case,
        config=config,
        db=db,
        db_sqlite=db_sqlite,
        notes=notes,
        test_mapping=test_mapping,
        output=output,
    )

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


@app.command("filter-cases")
def filter_cases(
    output: str = typer.Option(
        ..., "-o", "--output", help="Destination JSON or CSV file for the filtered cases"
    ),
    config: str | None = typer.Option(None, help="YAML settings file"),
    db: str | None = typer.Option(None, help="Path to case JSON or CSV file"),
    db_sqlite: str | None = typer.Option(None, help="Path to case SQLite database"),
    keywords: str | None = typer.Option(None, help="Comma separated keywords to match"),
    metadata: str | None = typer.Option(None, help="JSON string or file with key/value filters"),
) -> None:
    """Select cases by keywords or metadata."""

    args = SimpleNamespace(
        output=output,
        config=config,
        db=db,
        db_sqlite=db_sqlite,
        keywords=keywords,
        metadata=metadata,
    )

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


@app.command("login")
def login(
    username: str,
    api_url: str = typer.Option("http://localhost:8000/api/v1"),
    password: str | None = typer.Option(None),
    token_file: str = typer.Option(str(TOKEN_PATH), help="Path to store the session token"),
) -> None:
    """Authenticate with the API and store a session token."""

    pwd = password or getpass.getpass("Password: ")
    token.login(
        api_url,
        username,
        pwd,
        path=Path(token_file),
    )
    print(f"Token saved to {token_file}")


@app.command("manage-users")
def manage_users(
    cmd: str = typer.Argument(..., help="add, remove or list"),
    username: str | None = typer.Argument(None),
    password: str | None = typer.Option(None),
    group: str = typer.Option("default", help="User group"),
    file: str = typer.Option(
        os.environ.get(
            "UI_USERS_FILE",
            str(Path(__file__).parent / "sdb" / "ui" / "users.yml"),
        ),
        help="Path to credentials YAML file",
    ),
) -> None:
    """Add, remove or list web UI users."""

    data = {"users": {}}
    if os.path.exists(file):
        with open(file, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {"users": {}}
    users = data.get("users", {})

    if cmd == "add":
        pwd = password or getpass.getpass("Password: ")
        hashed = bcrypt.hashpw(pwd.encode(), bcrypt.gensalt()).decode()
        if not username:
            raise SystemExit("Username required")
        users[username] = {"password": hashed, "group": group}
        data["users"] = users
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh)
        print(f"Added user {username}")
    elif cmd == "remove":
        if username and users.pop(username, None) is not None:
            data["users"] = users
            with open(file, "w", encoding="utf-8") as fh:
                yaml.safe_dump(data, fh)
            print(f"Removed user {username}")
        else:
            print(f"User {username} not found", file=sys.stderr)
    else:  # list
        for name in sorted(users):
            print(name)


@app.callback(invoke_without_command=True)
def _main(
    ctx: typer.Context,
    config: str | None = typer.Option(None, help="YAML settings file"),
    db: str | None = typer.Option(None, help="Path to case JSON, CSV or directory"),
    db_sqlite: str | None = typer.Option(None, help="Path to case SQLite database"),
    case: str | None = typer.Option(None, help="Case identifier"),
    rubric: str | None = typer.Option(None, help="Path to scoring rubric JSON"),
    costs: str | None = typer.Option(None, help="Path to test cost table CSV"),
    cost_table: str | None = typer.Option(None, help="CSV file mapping test names to costs"),
    cost_estimator: str | None = typer.Option(None, help="Cost estimator plugin name"),
    correct_threshold: int = typer.Option(4, help="Judge score required for a correct diagnosis"),
    panel_engine: PanelEngine = typer.Option(PanelEngine.RULE, help="Decision engine to use for the panel"),
    llm_provider: LLMProvider = typer.Option(LLMProvider.OPENAI, help="LLM provider for LLM engine"),
    llm_model: str = typer.Option("gpt-4", help="Model name for LLM engine"),
    hf_model: str | None = typer.Option(None, help="Local HF model path"),
    persona_models: str | None = typer.Option(None, help="JSON string or file with persona to model mapping"),
    ollama_base_url: str = typer.Option("http://localhost:11434", help="Base URL for the Ollama server"),
    cache: bool = typer.Option(False, help="Cache LLM responses"),
    cache_size: int = typer.Option(128, help="Maximum number of responses to keep in the cache"),
    budget_limit: float | None = typer.Option(None, help="Maximum total spend allowed during the session"),
    semantic: bool = typer.Option(
        False,
        "--semantic",
        "--semantic-retrieval",
        help="Enable semantic retrieval for Gatekeeper",
    ),
    cross_encoder_model: str | None = typer.Option(None, help="Cross-encoder model name for semantic retrieval"),
    retrieval_backend: str | None = typer.Option(None, help="Retrieval plugin name"),
    convert: bool = typer.Option(False, help="Convert raw cases to JSON"),
    raw_dir: str = typer.Option("data/raw_cases", help="Directory with raw text cases for conversion"),
    output_dir: str = typer.Option("data/sdbench/cases", help="Destination for converted JSON cases"),
    hidden_dir: str = typer.Option("data/sdbench/hidden_cases", help="Directory for held-out cases from 2024-2025"),
    export_sqlite: str | None = typer.Option(None, help="Path to SQLite file when using --convert"),
    results_db: str | None = typer.Option(None, help="SQLite database to store evaluation metrics"),
    budget: float | None = typer.Option(None, help="Budget limit for Budgeted mode"),
    mode: str = typer.Option("unconstrained", help="Run mode"),
    vote_weights: str | None = typer.Option(None, help="JSON string or path with run ID weights for ensemble voting"),
    weights_file: str | None = typer.Option(None, help="JSON or YAML file mapping run IDs to vote weights"),
    metrics_port: int | None = typer.Option(None, help="Port for Prometheus metrics server (default 8000)"),
    verbosity: Verbosity = typer.Option(Verbosity.INFO, help="Logging verbosity"),
) -> None:
    """Run a demo diagnostic session using the virtual panel."""

    if ctx.invoked_subcommand:
        return

    args = SimpleNamespace(
        config=config,
        db=db,
        db_sqlite=db_sqlite,
        case=case,
        rubric=rubric,
        costs=costs,
        cost_table=cost_table,
        cost_estimator=cost_estimator,
        correct_threshold=correct_threshold,
        panel_engine=panel_engine.value,
        llm_provider=llm_provider,
        llm_model=llm_model,
        hf_model=hf_model,
        persona_models=persona_models,
        ollama_base_url=ollama_base_url,
        cache=cache,
        cache_size=cache_size,
        budget_limit=budget_limit,
        semantic=semantic,
        cross_encoder_model=cross_encoder_model,
        retrieval_backend=retrieval_backend,
        convert=convert,
        raw_dir=raw_dir,
        output_dir=output_dir,
        hidden_dir=hidden_dir,
        export_sqlite=export_sqlite,
        results_db=results_db,
        budget=budget,
        mode=mode,
        vote_weights=vote_weights,
        weights_file=weights_file,
        metrics_port=metrics_port,
        verbose=verbosity == Verbosity.DEBUG,
        quiet=verbosity == Verbosity.QUIET,
    )
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
        raise SystemExit("--db or --db-sqlite is required for a session")
    if any(item is None for item in required):
        raise SystemExit(
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
        if args.llm_provider == LLMProvider.OLLAMA:
            client = OllamaClient(
                base_url=args.ollama_base_url or settings.ollama_base_url,
                cache_path=cache_path,
                cache_size=args.cache_size,
            )
        elif args.llm_provider == LLMProvider.HF_LOCAL:
            client = HFLocalClient(
                model_path=args.hf_model or settings.hf_model,
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


def main(argv: list[str] | None = None) -> None:
    """Entry point for programmatic invocation."""

    from typer.main import get_command

    get_command(app).main(args=argv or sys.argv[1:], standalone_mode=False)


if __name__ == "__main__":
    from typer.main import get_command

    get_command(app).main(args=sys.argv[1:])
