import re
from typing import List, Tuple

try:  # pragma: no cover - trivial import handling
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:  # pragma: no cover - numpy not installed
    np = None  # type: ignore
    NUMPY_AVAILABLE = False


def _tokenize(text: str) -> List[str]:
    """Simple whitespace and punctuation tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


class SimpleEmbeddingIndex:
    """Naive embedding index using NumPy when available."""

    def __init__(self, documents: List[str]):
        self.documents = documents
        tokens_list = [_tokenize(doc) for doc in documents]
        vocab = sorted({tok for tokens in tokens_list for tok in tokens})
        self.vocab = {tok: i for i, tok in enumerate(vocab)}

        if NUMPY_AVAILABLE:
            self.embeddings = np.zeros(
                (len(documents), len(self.vocab)), dtype=float
            )
            for i, tokens in enumerate(tokens_list):
                for tok in tokens:
                    idx = self.vocab[tok]
                    self.embeddings[i, idx] += 1.0
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self.embeddings = self.embeddings / np.maximum(norms, 1e-8)
        else:
            self.embeddings = []
            for tokens in tokens_list:
                vec = [0.0] * len(self.vocab)
                for tok in tokens:
                    vec[self.vocab[tok]] += 1.0
                norm = sum(v * v for v in vec) ** 0.5
                if norm > 1e-8:
                    vec = [v / norm for v in vec]
                self.embeddings.append(vec)

    def query(self, text: str, top_k: int = 1) -> List[Tuple[str, float]]:
        """Return top matching documents and similarity scores."""
        if NUMPY_AVAILABLE:
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
        else:
            qvec = [0.0] * len(self.vocab)
            for tok in _tokenize(text):
                idx = self.vocab.get(tok)
                if idx is not None:
                    qvec[idx] += 1.0
            norm = sum(v * v for v in qvec) ** 0.5
            if norm == 0:
                return []
            qvec = [v / norm for v in qvec]
            scores = [
                sum(e[i] * qvec[i] for i in range(len(qvec)))
                for e in self.embeddings
            ]
            if not scores:
                return []
            indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:top_k]
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

        if not NUMPY_AVAILABLE:
            raise RuntimeError("SentenceTransformerIndex requires numpy")

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
