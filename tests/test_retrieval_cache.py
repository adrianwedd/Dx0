import types

import sdb.retrieval as retrieval


class DummyIndex:
    def __init__(self):
        self.calls = 0

    def query(self, text: str, top_k: int = 1):
        self.calls += 1
        return [(f"{text}:{self.calls}", 1.0)]


def test_cache_hit(monkeypatch):
    base = DummyIndex()
    cache = retrieval.CachedRetrievalIndex(base, ttl=10)
    monkeypatch.setattr(retrieval.time, "time", lambda: 0)
    monkeypatch.setattr(retrieval.time, "perf_counter", lambda: 0)
    first = cache.query("q")
    second = cache.query("q")
    assert first == second
    assert base.calls == 1


def test_cache_expiry(monkeypatch):
    base = DummyIndex()
    now = [0]
    monkeypatch.setattr(retrieval.time, "time", lambda: now[0])
    monkeypatch.setattr(retrieval.time, "perf_counter", lambda: now[0])
    cache = retrieval.CachedRetrievalIndex(base, ttl=5)
    cache.query("q")
    now[0] = 6
    cache.query("q")
    assert base.calls == 2
