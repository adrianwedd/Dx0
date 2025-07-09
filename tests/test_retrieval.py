import sdb.retrieval as retrieval
from sdb.llm_client import LLMClient
import numpy as np
import pytest
from types import SimpleNamespace


class DummyClient(LLMClient):
    def _chat(self, messages, model):
        return None


def test_simple_embedding_index_returns_match():
    docs = ["patient has a cough", "chest pain"]
    index = retrieval.SimpleEmbeddingIndex(docs)
    results = index.query("cough")
    assert results
    assert results[0][0] == "patient has a cough"


def test_simple_embedding_index_without_numpy(monkeypatch):
    """SimpleEmbeddingIndex should still work when numpy is missing."""
    import sdb.retrieval as retrieval

    monkeypatch.setattr(retrieval, "np", None, raising=False)
    monkeypatch.setattr(retrieval, "NUMPY_AVAILABLE", False)
    docs = ["patient has a cough", "chest pain"]
    index = retrieval.SimpleEmbeddingIndex(docs)
    results = index.query("cough")
    assert results
    assert results[0][0] == "patient has a cough"


def test_sentence_transformer_index_fallback():
    """SentenceTransformerIndex should fall back when model is unavailable."""
    docs = ["patient has a cough", "chest pain"]
    index = retrieval.SentenceTransformerIndex(docs, model_name="all-MiniLM-L6-v2")
    results = index.query("cough")
    assert results


def test_sentence_transformer_index_requires_numpy(monkeypatch):
    """SentenceTransformerIndex should raise when numpy is missing."""
    import sdb.retrieval as retrieval

    monkeypatch.setattr(retrieval, "np", None, raising=False)
    monkeypatch.setattr(retrieval, "NUMPY_AVAILABLE", False)
    with pytest.raises(RuntimeError):
        retrieval.SentenceTransformerIndex(["doc"])


def test_count_tokens_bpe():
    """LLMClient should tokenize using BPE when available."""

    msgs = [{"role": "user", "content": "erythema"}]
    tokens = DummyClient._count_tokens(msgs)
    assert tokens in {1, 3}


def test_cross_encoder_reranking(monkeypatch):
    """SentenceTransformerIndex should use reranker when available."""

    import sdb.retrieval as retrieval

    docs = ["patient denies cough", "patient has cough"]

    class DummyModel:
        def encode(self, texts, normalize_embeddings=True):
            vecs = []
            for t in texts:
                if "denies" in t:
                    vecs.append([1.0, 0.0])
                elif "has cough" in t:
                    vecs.append([0.9, 0.1])
                else:  # query
                    vecs.append([1.0, 0.0])
            return np.array(vecs)

    class DummyReranker:
        def __init__(self, *a, **k):
            pass

        def rerank(self, query, passages):
            return [(passages[1], 1.0), (passages[0], 0.5)]

    monkeypatch.setattr(
        retrieval,
        "SentenceTransformer",
        lambda name: DummyModel(),
    )
    monkeypatch.setattr(
        retrieval,
        "CrossEncoderReranker",
        lambda name=None: DummyReranker(),
    )

    index = retrieval.SentenceTransformerIndex(
        docs, model_name="dummy", cross_encoder_name="dummy"
    )
    results = index.query("cough", top_k=1)
    assert results[0][0] == "patient has cough"


def test_faiss_index_requires_faiss(monkeypatch):
    import sdb.retrieval as retrieval

    monkeypatch.setattr(retrieval, "FAISS_AVAILABLE", False)
    with pytest.raises(RuntimeError):
        retrieval.FaissIndex(["doc"])  # type: ignore


def test_faiss_index_query(monkeypatch):
    import sdb.retrieval as retrieval

    docs = ["patient denies cough", "patient has cough"]

    class DummyModel:
        def encode(self, texts, normalize_embeddings=True):
            vecs = []
            for t in texts:
                if "denies" in t:
                    vecs.append([1.0, 0.0])
                elif "has cough" in t:
                    vecs.append([0.9, 0.1])
                else:  # query
                    vecs.append([1.0, 0.0])
            return np.array(vecs, dtype="float32")

    class DummyIndex:
        def __init__(self, dim):
            self.vecs = []
            self.dim = dim

        def add(self, arr):
            self.vecs.extend(arr)

        def search(self, q, k):
            vecs = np.vstack(self.vecs)
            scores = vecs.dot(q.T)
            idxs = np.argsort(scores)[::-1][:k]
            return np.array([scores[idxs]]), np.array([idxs])

    dummy_faiss = SimpleNamespace(IndexFlatIP=DummyIndex, write_index=lambda i, p: None)

    monkeypatch.setattr(retrieval, "faiss", dummy_faiss)
    monkeypatch.setattr(retrieval, "FAISS_AVAILABLE", True)
    monkeypatch.setattr(retrieval, "SentenceTransformer", lambda n: DummyModel())
    monkeypatch.setattr(retrieval, "TRANSFORMERS_AVAILABLE", True)

    index = retrieval.FaissIndex(docs, model_name="dummy")
    results = index.query("cough", top_k=1)
    assert results[0][0] == "patient has cough"
