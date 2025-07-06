import json

from sdb.case_database import Case, CaseDatabase
from sdb.gatekeeper import Gatekeeper
from sdb.protocol import build_action, ActionType


def setup_gatekeeper():
    case = Case(id="1", summary="Patient complains of cough", full_text="History: patient has had a cough for 3 days.")
    db = CaseDatabase([case])
    gk = Gatekeeper(db, "1")
    gk.register_test_result("complete blood count", "normal")
    return gk


def test_load_results_from_json(tmp_path):
    case = Case(id="1", summary="s", full_text="f")
    db = CaseDatabase([case])
    gk = Gatekeeper(db, "1")

    data = {"cbc": "high"}
    path = tmp_path / "res.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    gk.load_results_from_json(str(path))
    q = build_action(ActionType.TEST, "cbc")
    res = gk.answer_question(q)
    assert res.content == "high"


def test_missing_fixture_file(tmp_path):
    case = Case(id="1", summary="s", full_text="f")
    db = CaseDatabase([case])
    gk = Gatekeeper(db, "1")
    gk.load_results_from_json(str(tmp_path / "missing.json"))
    q = build_action(ActionType.TEST, "cbc")
    res = gk.answer_question(q)
    assert res.synthetic is True


def test_question():
    gk = setup_gatekeeper()
    q = build_action(ActionType.QUESTION, "cough")
    res = gk.answer_question(q)
    assert "cough" in res.content.lower()
    assert res.synthetic is False


def test_test_query():
    gk = setup_gatekeeper()
    q = build_action(ActionType.TEST, "complete blood count")
    res = gk.answer_question(q)
    assert res.content == "normal"


def test_diagnosis_refusal():
    gk = setup_gatekeeper()
    q = build_action(ActionType.QUESTION, "what is the diagnosis?")
    res = gk.answer_question(q)
    assert res.synthetic is True


def test_case_insensitive_search():
    gk = setup_gatekeeper()
    q = build_action(ActionType.QUESTION, "COUGH FOR 3 DAYS")
    res = gk.answer_question(q)
    assert "cough for 3 days" in res.content.lower()
    assert res.synthetic is False


def test_invalid_xml():
    gk = setup_gatekeeper()
    res = gk.answer_question("<question>missing</question")
    assert res.synthetic is True
