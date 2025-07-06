"""Command line interface for running a demo diagnostic session."""

import argparse
import json
import logging
import os
import sys
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
    run_pipeline,
    start_metrics_server,
    load_scores,
    permutation_test,
)


def stats_main(argv: list[str]) -> None:
    """Run a permutation test on two result CSV files."""

    parser = argparse.ArgumentParser(description="Run significance test")
    parser.add_argument("baseline", help="CSV file for baseline results")
    parser.add_argument("variant", help="CSV file for variant results")
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

    a = load_scores(args.baseline, args.column)
    b = load_scores(args.variant, args.column)
    p = permutation_test(a, b, num_rounds=args.rounds)
    print(f"p-value: {p:.4f}")


def main() -> None:
    """Run a demo diagnostic session using the virtual panel."""

    parser = argparse.ArgumentParser(
        description="Run a simple diagnostic session"
    )
    parser.add_argument(
        "--db",
        help="Path to case JSON, CSV or directory",
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
    args = parser.parse_args()

    start_metrics_server()

    if args.convert:
        run_pipeline(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            hidden_dir=args.hidden_dir,
            fetch=False,
        )
        return

    required = [args.db, args.case, args.rubric, args.costs]
    if any(item is None for item in required):
        parser.error(
            "--db, --case, --rubric and --costs are required for a session"
        )

    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(message)s")

    if os.path.isdir(args.db):
        db = CaseDatabase.load_from_directory(args.db)
    elif args.db.endswith(".csv"):
        db = CaseDatabase.load_from_csv(args.db)
    else:
        db = CaseDatabase.load_from_json(args.db)

    gatekeeper = Gatekeeper(
        db,
        args.case,
        use_semantic_retrieval=args.semantic,
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
        if args.llm_provider == "ollama":
            client = OllamaClient()
        else:
            client = OpenAIClient()
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

    truth = db.get_case(args.case).summary
    result = evaluator.evaluate(
        orchestrator.final_diagnosis or "",
        truth,
        orchestrator.ordered_tests,
        visits=turn,
    )

    print(f"Final diagnosis: {orchestrator.final_diagnosis}")
    print(f"Total cost: ${result.total_cost:.2f}")
    print(f"Session score: {result.score}")
    print(f"Correct diagnosis: {result.correct}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        stats_main(sys.argv[2:])
    else:
        main()
