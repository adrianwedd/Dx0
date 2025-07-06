"""Command line interface for running a demo diagnostic session."""

import argparse
import os
from sdb import CaseDatabase, Gatekeeper, CostEstimator, VirtualPanel, Orchestrator
from sdb.cost_estimator import CptCost


def main() -> None:
    """Run a demo diagnostic session using the virtual panel."""

    parser = argparse.ArgumentParser(description="Run a simple diagnostic session")
    parser.add_argument("--db", required=True, help="Path to case JSON or directory")
    parser.add_argument("--case", required=True, help="Case identifier")
    args = parser.parse_args()

    if os.path.isdir(args.db):
        db = CaseDatabase.load_from_directory(args.db)
    else:
        db = CaseDatabase.load_from_json(args.db)

    gatekeeper = Gatekeeper(db, args.case)
    gatekeeper.register_test_result("complete blood count", "normal")

    cost_estimator = CostEstimator({"complete blood count": CptCost("100", 10.0)})
    panel = VirtualPanel()
    orchestrator = Orchestrator(panel, gatekeeper)

    turn = 0
    while not orchestrator.finished and turn < 10:
        response = orchestrator.run_turn("")
        print(f"Turn {turn+1}: {response}")
        turn += 1


if __name__ == '__main__':
    main()
