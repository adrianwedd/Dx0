from sdb.judge import Judge
import pytest


class DummyClient:
    def chat(self, messages, model):
        text = messages[-1]["content"].lower()
        if "heart attack" in text and "myocardial infarction" in text:
            return "5"
        if "type ii" in text and "type 2" in text:
            return "5"
        if "influenza virus" in text and "influenza" in text:
            return "4"
        if "common cold" in text and "influenza" in text:
            return "2"
        if "viral pneumonia" in text and "bacterial pneumonia" in text:
            return "3"
        if "gastritis" in text and "myocardial infarction" in text:
            return "1"
        return "1"


class FailingClient:
    def chat(self, messages, model):
        return None


class BadResponseClient:
    def chat(self, messages, model):
        return "no score"


def test_judge_llm_synonyms():
    j = Judge({}, client=DummyClient())
    res = j.evaluate("heart attack", "myocardial infarction")
    assert res.score == 5
    res = j.evaluate("Influenza virus", "Influenza")
    assert res.score == 4
    res = j.evaluate("Common cold", "Influenza")
    assert res.score == 2


def test_judge_nuanced_synonyms():
    j = Judge({}, client=DummyClient())
    res = j.evaluate("Type II diabetes", "Diabetes mellitus type 2")
    assert res.score == 5
    res = j.evaluate("Viral pneumonia", "Bacterial pneumonia")
    assert res.score == 3


def test_judge_misdiagnosis():
    j = Judge({}, client=DummyClient())
    res = j.evaluate("Gastritis", "Myocardial infarction")
    assert res.score == 1


def test_judge_llm_failure():
    j = Judge({}, client=FailingClient())
    res = j.evaluate("foo", "bar")
    assert res.score == 1


def test_judge_bad_response():
    j = Judge({}, client=BadResponseClient())
    res = j.evaluate("foo", "bar")
    assert res.score == 1


class StaticClient:
    def __init__(self, reply):
        self.reply = reply

    def chat(self, messages, model):
        return self.reply


def test_llm_score_parses_numeric_reply():
    j = Judge({}, client=StaticClient("Score: 4"))
    assert j._llm_score("a", "b") == 4


def test_llm_score_none_reply():
    j = Judge({}, client=StaticClient(None))
    with pytest.raises(RuntimeError):
        j._llm_score("a", "b")


def test_llm_score_bad_reply():
    j = Judge({}, client=StaticClient("no digits"))
    with pytest.raises(ValueError):
        j._llm_score("a", "b")
