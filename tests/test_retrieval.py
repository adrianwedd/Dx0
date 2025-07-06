from sdb.retrieval import SimpleEmbeddingIndex, SentenceTransformerIndex


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
