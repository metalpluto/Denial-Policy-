"""
pipeline.py

Ties together:
    1. retriever.py  -> top-k relevant policy chunks for a question (offline)
    2. generator.py  -> grounded answer citing sources (requires GOOGLE_API_KEY)

Retrieval always runs. Generation is optional so this pipeline works
fully offline with --no-generate.
"""

from .retriever import Retriever


def answer_question(question: str, retriever: Retriever, top_k: int = 3, use_generation: bool = True) -> dict:
    results = retriever.query(question, top_k=top_k)

    result = {
        "question": question,
        "retrieved": [
            {"source": r.chunk.source, "score": round(r.score, 3), "text": r.chunk.text}
            for r in results
        ],
        "answer": None,
    }

    if use_generation:
        from . import generator  # deferred import: only needed if generation runs
        try:
            result["answer"] = generator.generate_answer(question, results)
        except RuntimeError as e:
            result["answer"] = f"[skipped: {e}]"

    return result
