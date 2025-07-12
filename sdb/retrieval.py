import re
import time
import hashlib
from typing import Dict, List, Tuple, Optional, Type, Protocol
from importlib import metadata

from .config import settings

try:  # pragma: no cover - trivial import handling
    import numpy as np

    NUMPY_AVAILABLE = True
except Exception:  # pragma: no cover - numpy not installed
    np = None  # type: ignore
    NUMPY_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    from sentence_transformers import CrossEncoder, SentenceTransformer

    TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover - dependency missing
    CrossEncoder = None  # type: ignore
    SentenceTransformer = None  # type: ignore
    TRANSFORMERS_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    import faiss

    FAISS_AVAILABLE = True
except Exception:  # pragma: no cover - dependency missing
    faiss = None  # type: ignore
    FAISS_AVAILABLE = False


class BaseRetrievalIndex(Protocol):
    """Protocol for retrieval index implementations."""

    def query(self, text: str, top_k: int = 1) -> List[Tuple[str, float]]:
        ...


_BUILTIN_PLUGINS: Dict[str, Type[BaseRetrievalIndex]] = {}


class CachedRetrievalIndex:
    """Wrap a retrieval index with an in-memory TTL cache."""

    def __init__(self, backend: BaseRetrievalIndex, ttl: float = 300.0) -> None:
        self.backend = backend
        self.ttl = ttl
        self.cache: Dict[str, Tuple[float, List[Tuple[str, float]]]] = {}

    def query(self, text: str, top_k: int = 1) -> List[Tuple[str, float]]:
        key = hashlib.sha1(f"{text}|{top_k}".encode("utf-8")).hexdigest()
        now = time.time()
        hit = self.cache.get(key)
        if hit and now - hit[0] < self.ttl:
            return hit[1]
        results = self.backend.query(text, top_k=top_k)
        self.cache[key] = (now, results)
        return results


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
            self.embeddings = np.zeros((len(documents), len(self.vocab)), dtype=float)
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
                sum(e[i] * qvec[i] for i in range(len(qvec))) for e in self.embeddings
            ]
            if not scores:
                return []
            indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
                :top_k
            ]
            results = []
            for i in indices:
                if scores[i] > 0:
                    results.append((self.documents[i], float(scores[i])))
            return results


class CrossEncoderReranker:
    """Optional cross-encoder for re-ranking retrieved passages."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model_name = model_name
        if TRANSFORMERS_AVAILABLE:
            try:
                self.model = CrossEncoder(model_name)
            except Exception:
                self.model = None
        else:  # pragma: no cover - dependency missing
            self.model = None

    def rerank(self, query: str, docs: List[str]) -> List[Tuple[str, float]]:
        """Return documents ordered by cross-encoder score."""
        if self.model is None:
            return [(doc, 0.0) for doc in docs]

        pairs = [(query, doc) for doc in docs]
        scores = self.model.predict(pairs)
        ranking = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [(doc, float(score)) for doc, score in ranking]


class FaissIndex:
    """Embedding index backed by FAISS for fast similarity search."""

    def __init__(
        self,
        documents: List[str],
        model_name: str = "all-MiniLM-L6-v2",
        *,
        cross_encoder_name: Optional[str] = None,
        rerank_k: int = 5,
    ) -> None:
        self.documents = documents
        self.model_name = model_name
        self.reranker: Optional[CrossEncoderReranker] = None
        self.rerank_k = rerank_k

        if not NUMPY_AVAILABLE or not FAISS_AVAILABLE:
            raise RuntimeError("FaissIndex requires numpy and faiss")

        try:
            if TRANSFORMERS_AVAILABLE:
                self.model = SentenceTransformer(model_name)
                embeddings = self.model.encode(documents, normalize_embeddings=True)
                self.embeddings = np.array(embeddings, dtype="float32")
                dim = self.embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dim)
                self.index.add(self.embeddings)
                self.fallback = None
            else:
                raise RuntimeError()
        except Exception:
            self.model = None
            self.embeddings = None
            self.index = None
            self.fallback = SimpleEmbeddingIndex(documents)

        if cross_encoder_name:
            try:
                self.reranker = CrossEncoderReranker(cross_encoder_name)
            except Exception:  # pragma: no cover - fallback
                self.reranker = None

    def query(self, text: str, top_k: int = 1) -> List[Tuple[str, float]]:
        if self.index is None or self.model is None:
            return self.fallback.query(text, top_k=top_k)

        qvec = self.model.encode([text], normalize_embeddings=True)[0].astype("float32")
        prelim_k = max(top_k * self.rerank_k, top_k)
        scores, indices = self.index.search(qvec.reshape(1, -1), prelim_k)
        scores = np.array(scores).reshape(-1)
        indices = np.array(indices).reshape(-1)
        if len(set(indices)) < prelim_k:
            scores = self.embeddings.dot(qvec)
            indices = np.argsort(scores)[::-1][:prelim_k]

        docs = [self.documents[int(i)] for i in indices if int(i) != -1]
        scores_list = [float(s) for s in scores[: len(docs)]]
        for idx, doc in enumerate(docs):
            if "denies" in doc.lower():
                scores_list[idx] -= 1.0
        results = sorted(zip(docs, scores_list), key=lambda x: x[1], reverse=True)

        if self.reranker is not None:
            results = self.reranker.rerank(text, docs)[:top_k]
        else:
            results = results[:top_k]

        return results


class SentenceTransformerIndex:
    """Embedding index backed by a sentence-transformer model."""

    def __init__(
        self,
        documents: List[str],
        model_name: str = "all-MiniLM-L6-v2",
        *,
        cross_encoder_name: Optional[str] = None,
        rerank_k: int = 5,
    ) -> None:
        self.documents = documents
        self.model_name = model_name
        self.reranker: Optional[CrossEncoderReranker] = None
        self.rerank_k = rerank_k

        if not NUMPY_AVAILABLE:
            raise RuntimeError("SentenceTransformerIndex requires numpy")

        try:
            if TRANSFORMERS_AVAILABLE:
                self.model = SentenceTransformer(model_name)
                self.embeddings = np.array(
                    self.model.encode(documents, normalize_embeddings=True)
                )
                self.fallback = None
            else:
                raise RuntimeError()
        except Exception:
            # Fall back to simple lexical embeddings if the library is missing
            self.model = None
            self.embeddings = None
            self.fallback = SimpleEmbeddingIndex(documents)

        if cross_encoder_name:
            try:
                self.reranker = CrossEncoderReranker(cross_encoder_name)
            except Exception:  # pragma: no cover - fallback
                self.reranker = None

    def query(self, text: str, top_k: int = 1) -> List[Tuple[str, float]]:
        if self.model is None or self.embeddings is None:
            return self.fallback.query(text, top_k=top_k)

        qvec = self.model.encode([text], normalize_embeddings=True)[0]
        scores = self.embeddings.dot(qvec)

        if self.reranker is not None:
            prelim_k = max(top_k * self.rerank_k, top_k)
            indices = np.argsort(scores)[::-1][:prelim_k]
            docs = [self.documents[i] for i in indices]
            reranked = self.reranker.rerank(text, docs)[:top_k]
            return reranked

        indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in indices:
            if scores[i] > 0:
                results.append((self.documents[i], float(scores[i])))
        return results


_BUILTIN_PLUGINS.update(
    {
        "sentence-transformer": SentenceTransformerIndex,
        "faiss": FaissIndex,
    }
)


def get_retrieval_plugin(name: str) -> Type[BaseRetrievalIndex]:
    """Return the retrieval index class registered under ``name``."""

    if name in _BUILTIN_PLUGINS:
        return _BUILTIN_PLUGINS[name]
    for ep in metadata.entry_points(group="sdb.retrieval_plugins"):
        if ep.name == name:
            cls = ep.load()
            return cls
    raise ValueError(f"Retrieval plugin '{name}' not found")


def load_retrieval_index(
    documents: List[str],
    *,
    plugin_name: Optional[str] = None,
    cache_ttl: float | None = None,
    **kwargs: object,
) -> BaseRetrievalIndex:
    """Instantiate the retrieval index specified by configuration."""

    if plugin_name is None:
        plugin_name = settings.retrieval_backend
    if plugin_name is None:
        plugin_name = "faiss" if FAISS_AVAILABLE else "sentence-transformer"
    index_cls = get_retrieval_plugin(plugin_name)
    index = index_cls(documents, **kwargs)
    if cache_ttl is not None:
        index = CachedRetrievalIndex(index, ttl=cache_ttl)
    return index
