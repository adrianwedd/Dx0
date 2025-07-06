import csv
from types import SimpleNamespace
from sdb.cpt_lookup import lookup_cpt


def test_cache_hit(tmp_path):
    cache = tmp_path / "cache.csv"
    with open(cache, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["test_name", "cpt_code"])
        writer.writeheader()
        writer.writerow({"test_name": "cbc", "cpt_code": "85027"})
    assert lookup_cpt("cbc", cache_path=str(cache)) == "85027"


def test_llm_lookup_and_cache(tmp_path, monkeypatch):
    cache = tmp_path / "cache.csv"

    def fake_create(model, messages, max_tokens):
        choice = SimpleNamespace(message={"content": "12345"})
        return SimpleNamespace(choices=[choice])

    dummy_openai = SimpleNamespace(
        ChatCompletion=SimpleNamespace(create=fake_create),
        api_key=None,
    )
    import sdb.cpt_lookup as cl

    monkeypatch.setattr(cl, "openai", dummy_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    code = cl.lookup_cpt("bmp", cache_path=str(cache))
    assert code == "12345"
    with open(cache, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["cpt_code"] == "12345"
