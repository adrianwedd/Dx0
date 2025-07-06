import re
from typing import List, Tuple
import numpy as np


def _tokenize(text: str) -> List[str]:
    """Simple whitespace and punctuation tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


class SimpleEmbeddingIndex:
    """Naive embedding index backed by NumPy vectors."""

    def __init__(self, documents: List[str]):
        self.documents = documents
        tokens_list = [_tokenize(doc) for doc in documents]
        vocab = sorted({tok for tokens in tokens_list for tok in tokens})
        self.vocab = {tok: i for i, tok in enumerate(vocab)}
        self.embeddings = np.zeros(
            (len(documents), len(self.vocab)), dtype=float
        )
        for i, tokens in enumerate(tokens_list):
            for tok in tokens:
                idx = self.vocab[tok]
                self.embeddings[i, idx] += 1.0
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / np.maximum(norms, 1e-8)

    def query(self, text: str, top_k: int = 1) -> List[Tuple[str, float]]:
        """Return top matching documents and similarity scores."""
        qvec = np.zeros(len(self.vocab), dtype=float)
        for tok in _tokenize(text):
            idx = self.vocab.get(tok)
            if idx is not None:
                qvec[idx] += 1.0
        norm = np.linalg.norm(qvec)
        if norm == 0:
            return []
        qvec /= norm
        scores = self.embeddings.dot(qvec)
        if scores.size == 0:
            return []
        indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in indices:
            if scores[i] > 0:
                results.append((self.documents[i], float(scores[i])))
        return results


class SentenceTransformerIndex:
    """Embedding index backed by a sentence-transformer model."""

    def __init__(
        self, documents: List[str], model_name: str = "all-MiniLM-L6-v2"
    ) -> None:
        self.documents = documents
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            self.embeddings = np.array(
                self.model.encode(documents, normalize_embeddings=True)
            )
            self.fallback = None
        except Exception:
            # Fall back to simple lexical embeddings if the library is missing
            self.model = None
            self.embeddings = None
            self.fallback = SimpleEmbeddingIndex(documents)

    def query(self, text: str, top_k: int = 1) -> List[Tuple[str, float]]:
        if self.model is None or self.embeddings is None:
            return self.fallback.query(text, top_k=top_k)

        qvec = self.model.encode([text], normalize_embeddings=True)[0]
        scores = self.embeddings.dot(qvec)
        indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in indices:
            if scores[i] > 0:
                results.append((self.documents[i], float(scores[i])))
        return results
