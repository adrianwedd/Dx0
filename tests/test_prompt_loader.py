from sdb.prompt_loader import load_prompt


def test_load_prompt(tmp_path, monkeypatch):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "example.txt").write_text("hello")
    monkeypatch.setattr("sdb.prompt_loader.PROMPT_DIR", str(prompts))
    assert load_prompt("example") == "hello"
