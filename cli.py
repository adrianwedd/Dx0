import argparse
from sdb import Case, CaseDatabase, Gatekeeper, CostEstimator, VirtualPanel, Orchestrator
from sdb.cost_estimator import TestCost


def main():
    parser = argparse.ArgumentParser(description="Run a simple diagnostic session")
    parser.add_argument('--case', default='1')
    args = parser.parse_args()

    # Sample case
    case = Case(id="1", summary="Patient complains of cough", full_text="History: patient has had a cough for 3 days.")
    db = CaseDatabase([case])
    gatekeeper = Gatekeeper(db, args.case)
    gatekeeper.register_test_result("complete blood count", "normal")

    cost_estimator = CostEstimator({"complete blood count": TestCost("100", 10.0)})
    panel = VirtualPanel()
    orchestrator = Orchestrator(panel, gatekeeper)

    turn = 0
    while not orchestrator.finished and turn < 10:
        response = orchestrator.run_turn("")
        print(f"Turn {turn+1}: {response}")
        turn += 1


if __name__ == '__main__':
    main()
