import json
from sdb.ingest.translate import translate_directory


class DummyClient:
    def __init__(self):
        self.calls = []

    def chat(self, messages, model):
        self.calls.append(messages[-1]["content"])
        return "T:" + messages[-1]["content"]


def test_translate_directory(tmp_path):
    src = tmp_path / "src"
    dest = tmp_path / "dest"
    src.mkdir()
    data = {"id": "case_1", "summary": "sum", "steps": [{"id": 1, "text": "txt"}]}
    (src / "case_1.json").write_text(json.dumps(data))

    client = DummyClient()
    translate_directory(str(src), "es", dest_dir=str(dest), client=client)

    out_path = dest / "case_1_es.json"
    out_data = json.loads(out_path.read_text())
    assert out_data["summary"].startswith("T:")
    assert out_data["steps"][0]["text"].startswith("T:")
    assert client.calls
