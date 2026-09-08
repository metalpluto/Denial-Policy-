"""
chunker.py

Splits policy documents into overlapping word-based chunks, so a
question can be matched against a focused passage rather than an
entire document. Pure text processing — no API, no network.
"""

from dataclasses import dataclass
from pathlib import Path

DOCS_DIR = Path(__file__).parent.parent / "data" / "policy_docs"

CHUNK_SIZE = 120   # words per chunk
CHUNK_OVERLAP = 30  # words shared between consecutive chunks


@dataclass
class Chunk:
    text: str
    source: str       # filename the chunk came from
    chunk_index: int  # position of this chunk within its source document

    def to_dict(self) -> dict:
        return {"text": self.text, "source": self.source, "chunk_index": self.chunk_index}


def chunk_text(text: str, source: str) -> list[Chunk]:
    """
    Split text into overlapping chunks of CHUNK_SIZE words, advancing
    by (CHUNK_SIZE - CHUNK_OVERLAP) words each step so consecutive
    chunks share CHUNK_OVERLAP words of context. Whitespace (including
    newlines) is normalized to single spaces during splitting.
    """
    words = text.split()
    if not words:
        return []

    step = CHUNK_SIZE - CHUNK_OVERLAP
    chunks = []
    idx = 0
    start = 0
    while start < len(words):
        piece = words[start:start + CHUNK_SIZE]
        chunks.append(Chunk(text=" ".join(piece), source=source, chunk_index=idx))
        idx += 1
        if start + CHUNK_SIZE >= len(words):
            break
        start += step
    return chunks


def load_and_chunk_all(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    """
    Read every .txt file in docs_dir and return all chunks from all
    documents combined into a single flat list, sorted by filename
    for a stable, reproducible order.
    """
    all_chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        all_chunks.extend(chunk_text(text, source=path.name))
    return all_chunks
