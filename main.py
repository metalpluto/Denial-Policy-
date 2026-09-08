"""
main.py

CLI for the Denial Policy RAG Assistant.

Usage:
    python main.py --query "How long do I have to file an appeal?"
    python main.py --query "..." --no-generate   # retrieval only, fully offline
    python main.py --query "..." --top-k 5
"""

import argparse
import json

from src.pipeline import answer_question
from src.retriever import build_default_retriever


def main():
    parser = argparse.ArgumentParser(description="Denial Policy RAG Assistant")
    parser.add_argument("--query", required=True, help="Question to ask the policy assistant")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument("--no-generate", action="store_true", help="Skip LLM generation, retrieval only")
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of formatted output")
    args = parser.parse_args()

    retriever = build_default_retriever()
    result = answer_question(
        args.query,
        retriever,
        top_k=args.top_k,
        use_generation=not args.no_generate,
    )

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("QUESTION")
    print(" ", result["question"])
    print()
    print("RETRIEVED PASSAGES")
    for r in result["retrieved"]:
        print(f"  [{r['source']}  score={r['score']}]")
        print(f"    {r['text'][:600]}{'...' if len(r['text']) > 600 else ''}")
        print()

    if result["answer"] is not None:
        print("ANSWER")
        print(" ", result["answer"])


if __name__ == "__main__":
    main()
