from typing import List, Tuple
from sdb.retrieval import BaseRetrievalIndex
class ExampleIndex(BaseRetrievalIndex):
    """Minimal retrieval backend returning a fixed ranking."""

    def __init__(self, documents: List[str]):
        self.documents = documents

    def query(self, text: str, top_k: int = 1) -> List[Tuple[str, float]]:
        scores = [
            (doc, float(len(set(text.split()) & set(doc.split()))))
            for doc in self.documents
        ]
        scores.sort(key=lambda pair: pair[1], reverse=True)
        return scores[:top_k]
