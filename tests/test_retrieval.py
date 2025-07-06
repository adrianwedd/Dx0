from sdb.retrieval import SimpleEmbeddingIndex


def test_simple_embedding_index_returns_match():
    docs = ["patient has a cough", "chest pain"]
    index = SimpleEmbeddingIndex(docs)
    results = index.query("cough")
    assert results
    assert results[0][0] == "patient has a cough"
