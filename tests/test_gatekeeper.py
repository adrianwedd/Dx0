from sdb.case_database import Case, CaseDatabase
from sdb.gatekeeper import Gatekeeper
from sdb.protocol import build_action, ActionType


def setup_gatekeeper():
    case = Case(id="1", summary="Patient complains of cough", full_text="History: patient has had a cough for 3 days.")
    db = CaseDatabase([case])
    gk = Gatekeeper(db, "1")
    gk.register_test_result("complete blood count", "normal")
    return gk


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
