"""
generator.py

Optional generation step: takes the retrieved chunks and the user's
question, and asks an LLM to synthesize a direct answer grounded only
in those chunks, citing which source document each part came from.

This step requires GOOGLE_API_KEY. It is intentionally separated from
retriever.py so the retrieval half of this project stays fully offline
and independently testable — the same pattern used in the Medical
Claim Denial Analyzer project (extractor/classifier offline, agent
step requires an API key).
"""

import os

from langchain_google_genai import ChatGoogleGenerativeAI

from .retriever import RetrievedChunk

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", max_output_tokens=1024)


def generate_answer(question: str, results: list[RetrievedChunk]) -> str:
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY not set. Add it to a .env file or your "
            "environment before running generation."
        )

    context = "\n\n".join(
        f"[Source: {r.chunk.source}]\n{r.chunk.text}" for r in results
    )

    prompt = f"""You are a healthcare policy assistant. Answer the question
below using ONLY the policy excerpts provided as context. If the context
does not contain enough information to answer, say so plainly rather than
guessing.

Cite which source document supports each part of your answer, using the
filenames shown in the context (e.g. "per timely_filing.txt").

Context:
{context}

Question: {question}

Give a direct, plain-prose answer in under 120 words. No markdown
headers, no bullet points, no placeholder brackets."""

    response = llm.invoke(prompt)
    content = response.content
    if isinstance(content, list):
        content = "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return content
