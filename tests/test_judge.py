from sdb.judge import Judge


class DummyClient:
    def chat(self, messages, model):
        text = messages[-1]["content"].lower()
        if "heart attack" in text and "myocardial infarction" in text:
            return "5"
        if "influenza virus" in text and "influenza" in text:
            return "4"
        if "common cold" in text and "influenza" in text:
            return "2"
        return "1"


def test_judge_llm_synonyms():
    j = Judge({}, client=DummyClient())
    res = j.evaluate("heart attack", "myocardial infarction")
    assert res.score == 5
    res = j.evaluate("Influenza virus", "Influenza")
    assert res.score == 4
    res = j.evaluate("Common cold", "Influenza")
    assert res.score == 2
