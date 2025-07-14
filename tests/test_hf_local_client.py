import json
from sdb.llm_client import HFLocalClient


class DummyPipe:
    def __call__(self, text, max_new_tokens=64):
        return [{"generated_text": text + " reply"}]


def test_hf_local_client(monkeypatch, tmp_path):
    monkeypatch.setattr("transformers.pipeline", lambda *a, **k: DummyPipe())
    cache_file = tmp_path / "cache.jsonl"
    client = HFLocalClient("model", cache_path=str(cache_file))
    msg = [{"role": "user", "content": "hello"}]
    out = client.chat(msg, "m")
    assert out == "reply"
    with open(cache_file, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["value"] == "reply"
