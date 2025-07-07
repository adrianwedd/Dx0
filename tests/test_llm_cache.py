import json
from sdb.llm_client import LLMClient


class DummyClient(LLMClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0

    def _chat(self, messages, model):
        self.calls += 1
        return f"reply {self.calls}"


def test_cache_hit(tmp_path):
    cache_file = tmp_path / "cache.jsonl"
    client = DummyClient(cache_path=str(cache_file), cache_size=2)
    msgs = [{"role": "user", "content": "hi"}]
    first = client.chat(msgs, "model")
    assert first == "reply 1"
    again = client.chat(msgs, "model")
    assert again == "reply 1"
    assert client.calls == 1
    with open(cache_file, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["value"] == "reply 1"


def test_cache_eviction(tmp_path):
    cache_file = tmp_path / "cache.jsonl"
    client = DummyClient(cache_path=str(cache_file), cache_size=1)
    m1 = [{"role": "user", "content": "a"}]
    m2 = [{"role": "user", "content": "b"}]
    client.chat(m1, "model")
    client.chat(m2, "model")
    client.chat(m1, "model")
    # m1 should have been evicted and fetched again
    assert client.calls == 3
