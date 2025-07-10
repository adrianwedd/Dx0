import json
import pytest

from sdb.case_database import Case, CaseDatabase
from sdb.gatekeeper import Gatekeeper
import sdb.gatekeeper as gatekeeper
from sdb.protocol import build_action, ActionType


def setup_gatekeeper(semantic: bool = False):
    case = Case(
        id="1",
        summary="Patient complains of cough",
        full_text="History: patient has had a cough for 3 days.",
    )
    db = CaseDatabase([case])
    gk = Gatekeeper(db, "1", use_semantic_retrieval=semantic)
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
    with pytest.raises(FileNotFoundError):
        gk.load_results_from_json(str(tmp_path / "missing.json"))


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


def test_semantic_retrieval_enabled():
    gk = setup_gatekeeper(semantic=True)
    q = build_action(ActionType.QUESTION, "cough")
    res = gk.answer_question(q)
    assert "cough" in res.content.lower()
    assert "context:" in res.content.lower()
    assert res.synthetic is False


def test_cross_encoder_name_passed(monkeypatch):
    captured = {}

    class DummyIndex:
        def __init__(
            self,
            docs,
            model_name="all-MiniLM-L6-v2",
            *,
            cross_encoder_name=None,
            rerank_k=5,
            plugin_name=None,
        ):
            captured["name"] = cross_encoder_name

    monkeypatch.setattr(
        gatekeeper,
        "load_retrieval_index",
        lambda docs, **k: DummyIndex(docs, **k),
    )

    case = Case(id="1", summary="s", full_text="t")
    db = CaseDatabase([case])
    Gatekeeper(db, "1", use_semantic_retrieval=True, cross_encoder_name="ce")
    assert captured["name"] == "ce"


def test_invalid_xml():
    gk = setup_gatekeeper()
    res = gk.answer_question("<question>missing</question")
    assert res.synthetic is True


def test_mixed_question_and_test_refused():
    gk = setup_gatekeeper()
    query = "<wrapper><question>a</question><test>b</test></wrapper>"
    res = gk.answer_question(query)
    assert res.synthetic is True


def test_unknown_xml_tag():
    """Unrecognized actions should return a synthetic reply."""
    gk = setup_gatekeeper()
    res = gk.answer_question("<foo>bar</foo>")
    assert res.synthetic is True
    assert "Invalid query" in res.content


def test_load_results_from_invalid_json(tmp_path):
    """Malformed JSON fixtures should raise a ``ValueError``."""
    case = Case(id="1", summary="s", full_text="f")
    db = CaseDatabase([case])
    gk = Gatekeeper(db, "1")

    path = tmp_path / "bad.json"
    path.write_text("{invalid}", encoding="utf-8")

    with pytest.raises(ValueError):
        gk.load_results_from_json(str(path))


def test_unexpected_nested_xml():
    """Nested tags mixing actions should be rejected."""
    gk = setup_gatekeeper()
    query = "<question>info<test>cbc</test></question>"
    res = gk.answer_question(query)
    assert res.synthetic is True
    assert "Invalid query" in res.content


def test_malicious_doctype():
    """XXE-style payloads should be rejected as invalid."""
    gk = setup_gatekeeper()
    payload = (
        "<!DOCTYPE foo [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]><question>&xxe;</question>"  # noqa: E501
    )
    res = gk.answer_question(payload)
    assert res.synthetic is True
    assert "Invalid query" in res.content
