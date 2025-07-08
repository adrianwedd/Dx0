"""Command line interface for running a demo diagnostic session."""

import argparse
import json
import logging
import os
import sys
from sdb.config import load_settings, settings
from sdb import (
    CaseDatabase,
    Gatekeeper,
    CostEstimator,
    VirtualPanel,
    RuleEngine,
    LLMEngine,
    OpenAIClient,
    OllamaClient,
    Orchestrator,
    Judge,
    Evaluator,
    DiagnosisResult,
    MetaPanel,
    batch_evaluate,
    configure_logging,
    run_pipeline,
    start_metrics_server,
    load_scores,
    permutation_test,
    load_from_sqlite,
    transcript_to_fhir,
    ordered_tests_to_fhir,
    diagnostic_report_to_case,
    bundle_to_case,
)
import csv


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
    parser.add_argument("--costs", required=True, help="Test cost table CSV")
    parser.add_argument("--output", required=True, help="CSV file for results")
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
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument(
        "--mode",
        choices=["unconstrained", "budgeted", "question-only", "instant"],
        default="unconstrained",
    )
    parser.add_argument(
        "--vote-weights",
        default=None,
        help=(
            "JSON string or path with run ID weights for ensemble voting"
        ),
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
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    vote_weights = _load_weights(args.vote_weights)
    meta_panel = MetaPanel(weights=vote_weights)

    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    configure_logging(level)

    if args.db is None and args.db_sqlite is None:
        parser.error("--db or --db-sqlite is required")

    if args.db_sqlite:
        db = load_from_sqlite(args.db_sqlite)
    elif os.path.isdir(args.db):
        db = CaseDatabase.load_from_directory(args.db)
    elif args.db.endswith(".csv"):
        db = CaseDatabase.load_from_csv(args.db)
    else:
        db = CaseDatabase.load_from_json(args.db)

    with open(args.rubric, "r", encoding="utf-8") as fh:
        rubric = json.load(fh)

    cost_estimator = CostEstimator.load_from_csv(args.costs)
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
            engine = LLMEngine(model=args.llm_model, client=client)

        panel = VirtualPanel(decision_engine=engine)
        orch_kwargs = {}
        if args.mode == "budgeted":
            orch_kwargs["cost_estimator"] = cost_estimator
            orch_kwargs["budget"] = args.budget
        if args.mode == "question-only":
            orch_kwargs["question_only"] = True

        orchestrator = Orchestrator(panel, gatekeeper, **orch_kwargs)
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
        help=(
            "JSON string or path with run ID weights for ensemble voting"
        ),
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=None,
        help="Port for Prometheus metrics server (default 8000)",
    )
    args = parser.parse_args()
    load_settings(args.config)

    vote_weights = _load_weights(args.vote_weights)
    meta_panel = MetaPanel(weights=vote_weights)

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

    required = [args.case, args.rubric, args.costs]
    if args.db is None and args.db_sqlite is None:
        parser.error("--db or --db-sqlite is required for a session")
    if any(item is None for item in required):
        parser.error("--case, --rubric and --costs are required for a session")

    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    configure_logging(level)

    if args.db_sqlite:
        db = load_from_sqlite(args.db_sqlite)
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
    )
    gatekeeper.register_test_result("complete blood count", "normal")

    cost_estimator = CostEstimator.load_from_csv(args.costs)

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
        engine = LLMEngine(model=args.llm_model, client=client)

    panel = VirtualPanel(decision_engine=engine)

    orch_kwargs = {}
    if args.mode == "budgeted":
        orch_kwargs["cost_estimator"] = cost_estimator
        orch_kwargs["budget"] = args.budget
    if args.mode == "question-only":
        orch_kwargs["question_only"] = True

    orchestrator = Orchestrator(panel, gatekeeper, **orch_kwargs)

    turn = 0
    max_turns = 1 if args.mode == "instant" else 10
    while not orchestrator.finished and turn < max_turns:
        response = orchestrator.run_turn("")
        print(f"Turn {turn+1}: {response}")
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
    else:
        main()
