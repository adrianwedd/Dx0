from sdb.retrieval import SimpleEmbeddingIndex, SentenceTransformerIndex
from sdb.llm_client import LLMClient


class DummyClient(LLMClient):
    def _chat(self, messages, model):
        return None


def test_simple_embedding_index_returns_match():
    docs = ["patient has a cough", "chest pain"]
    index = SimpleEmbeddingIndex(docs)
    results = index.query("cough")
    assert results
    assert results[0][0] == "patient has a cough"


def test_sentence_transformer_index_fallback():
    """SentenceTransformerIndex should fall back when model is unavailable."""
    docs = ["patient has a cough", "chest pain"]
    index = SentenceTransformerIndex(docs, model_name="all-MiniLM-L6-v2")
    results = index.query("cough")
    assert results


def test_count_tokens_bpe():
    """LLMClient should tokenize using BPE when available."""

    msgs = [{"role": "user", "content": "erythema"}]
    assert DummyClient._count_tokens(msgs) == 3
