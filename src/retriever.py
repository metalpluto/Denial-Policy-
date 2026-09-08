"""
retriever.py

TF-IDF + cosine similarity retrieval over the chunked policy documents.
Fully offline: no API key, no downloaded embedding model. This is a
sparse, lexical retrieval method — it matches on word overlap, not
semantic meaning — which is a deliberate tradeoff explained in the
README. It's the same technique (scikit-learn TfidfVectorizer) already
used in the Medical Claim Denial Analyzer project, kept consistent
across both.
"""

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunker import Chunk, load_and_chunk_all


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class Retriever:
    """
    Builds a TF-IDF index over a fixed set of chunks at construction
    time, then answers nearest-neighbor queries against that index.
    """

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        texts = [c.text for c in chunks]
        self.matrix = self.vectorizer.fit_transform(texts)

    def query(self, question: str, top_k: int = 3) -> list[RetrievedChunk]:
        """
        Return the top_k chunks most similar to the question, ranked
        by cosine similarity, highest first.
        """
        q_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self.matrix)[0]
        ranked_indices = scores.argsort()[::-1][:top_k]
        return [
            RetrievedChunk(chunk=self.chunks[i], score=float(scores[i]))
            for i in ranked_indices
        ]


def build_default_retriever() -> Retriever:
    """Load and chunk every policy document, then build a Retriever over it."""
    chunks = load_and_chunk_all()
    return Retriever(chunks)
