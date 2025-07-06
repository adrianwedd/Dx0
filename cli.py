"""Command line interface for running a demo diagnostic session."""

import argparse
import json
import logging
import os
from sdb import (
    CaseDatabase,
    Gatekeeper,
    CostEstimator,
    VirtualPanel,
    RuleEngine,
    LLMEngine,
    Orchestrator,
    Judge,
    Evaluator,
)


def main() -> None:
    """Run a demo diagnostic session using the virtual panel."""

    parser = argparse.ArgumentParser(
        description="Run a simple diagnostic session"
    )
    parser.add_argument(
        "--db",
        required=True,
        help="Path to case JSON, CSV or directory",
    )
    parser.add_argument("--case", required=True, help="Case identifier")
    parser.add_argument(
        "--rubric",
        required=True,
        help="Path to scoring rubric JSON",
    )
    parser.add_argument(
        "--costs",
        required=True,
        help="Path to test cost table CSV",
    )
    parser.add_argument(
        "--panel-engine",
        choices=["rule", "llm"],
        default="rule",
        help="Decision engine to use for the panel",
    )
    parser.add_argument(
        "--llm-model",
        choices=["gpt-4", "turbo"],
        default="gpt-4",
        help="Model name for LLM engine",
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

    gatekeeper = Gatekeeper(db, args.case)
    gatekeeper.register_test_result("complete blood count", "normal")

    cost_estimator = CostEstimator.load_from_csv(args.costs)

    with open(args.rubric, "r", encoding="utf-8") as fh:
        rubric = json.load(fh)

    judge = Judge(rubric)
    evaluator = Evaluator(judge, cost_estimator)

    if args.panel_engine == "rule":
        engine = RuleEngine()
    else:
        engine = LLMEngine(model=args.llm_model)

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
        orchestrator.final_diagnosis or "", truth, orchestrator.ordered_tests
    )

    print(f"Final diagnosis: {orchestrator.final_diagnosis}")
    print(f"Total cost: ${result.total_cost:.2f}")
    print(f"Session score: {result.score}")


if __name__ == "__main__":
    main()
