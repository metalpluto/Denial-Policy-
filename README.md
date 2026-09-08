# Denial Policy RAG Assistant

A retrieval-augmented generation system for answering questions about
medical claim denial policy, grounded in a small local knowledge base
of policy documents. Ask a question e.g. *"How long do I have to
file an appeal?"* and it retrieves the most relevant policy passages,
then (optionally) generates a direct answer citing which document it
came from

## Why this design

The pipeline has two stages, deterministic, fully offline core, and an optional
LLM step that requires an API key.

**Retrieval (offline, no API key needed).** Documents are split into
overlapping 120-word chunks (30-word overlap, so a fact split across a
chunk boundary is still findable) and indexed with a TF-IDF vectorizer
the same technique already used in the Medical Claim Denial Analyzer
project. Given a question, chunks are ranked by cosine similarity to
the query.

TF-IDF is a **sparse, lexical** retrieval method it matches based on
shared vocabulary between the query and the text, not semantic
meaning. This is a deliberate scope decision, not an oversight: it
keeps the whole retrieval step dependency-light, deterministic, and
testable with no downloaded embedding model and no network call. The
tradeoff is real and is covered honestly below.

**Generation (optional, requires `GOOGLE_API_KEY`).** The retrieved
chunks are handed to an LLM with an explicit instruction to answer
only from the provided context, cite the source document, and say so
plainly if the context is insufficient rather than filling gaps from
the model's own general knowledge

## Setup

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY=your_key_here   # only needed for generation
```

## Usage

```bash
python main.py --query "How long do I have to file an appeal?"
python main.py --query "Do I need authorization before an MRI?" --no-generate
python main.py --query "..." --top-k 5 --json
```

## Running with Docker

Build the image:

```bash
docker build -t denial-policy-rag .
```

Run it retrieval only, fully offline, no API key needed:

```bash
docker run denial-policy-rag --query "Do I need authorization before an MRI?" --no-generate
```

Run it with generation enabled  pass the API key at run time with
`-e`, never bake it into the image or the Dockerfile:

```bash
docker run -e GOOGLE_API_KEY=your_key_here denial-policy-rag --query "How long do I have to file an appeal?"
```

`tests/` and `README.md` are excluded from the built image via
`.dockerignore`, since they're not needed at runtime this keeps the
image focused on just what's required to actually answer a query

## Testing

Chunking and retrieval are both pure, offline functions, so they're
fully covered by unit tests no API key or network access needed.

```bash
python -m pytest tests/ -v
```

The generation step isn't unit tested here since it depends on a live
LLM call tests focus on the deterministic retrieval core.

**Verified retrieval accuracy:** across five representative questions
(one per policy category), the correct source document appeared in the
top-3 retrieved chunks for all five. Top-1 accuracy was lower see
Known Limitations.

## Known limitations

- **TF-IDF retrieval is lexical, not semantic.** It matches on shared
  words, not meaning. A query like *"How many days do I have to submit
  a claim?"* retrieved `timely_filing.txt` only at rank 3 (score
  0.237, top-3), not rank 1 because the query's phrasing shares
  little exact vocabulary with the document's wording ("120 days,"
  "filing window"). At `top_k=2`, the same query for *"What happens
  if I file a claim late?"* misses `timely_filing.txt` entirely. A
  production system would likely use dense embeddings (e.g. a
  sentence-transformer model) or a hybrid sparse+dense approach to
  close this gap that's the direct next step this project doesn't
  yet take, by design, to keep the retrieval core dependency-free.

- **Small, synthetic knowledge base.** Five documents, twelve total
  chunks. Real payer policy documents are longer, more numerous, and
  more inconsistent in phrasing across payers a real deployment
  would need a much larger and more varied corpus, at which point
  TF-IDF's lexical-matching weakness would compound further.

- **No re-ranking step.** Chunks are returned in raw cosine-similarity
  order with no secondary relevance check (e.g. an LLM-based
  re-ranker), which is a common addition in production RAG systems to
  correct exactly the kind of lexical-mismatch cases described above.

- **Generation is not fact-checked against the retrieved context.**
  The prompt instructs the model to answer only from the provided
  chunks, but nothing in this project verifies that the generated
  answer doesn't drift from what the context actually supports.

## Project structure

```
denial-policy-rag/
├── data/policy_docs/           five synthetic policy documents
│   ├── timely_filing.txt
│   ├── medical_necessity.txt
│   ├── prior_authorization.txt
│   ├── coding_compliance.txt
│   └── appeals_process.txt
├── src/
│   ├── chunker.py              overlapping word-chunk splitting
│   ├── retriever.py            TF-IDF + cosine similarity search
│   ├── generator.py            optional grounded-answer generation
│   └── pipeline.py             ties retrieval + generation together
├── tests/
│   ├── test_chunker.py         offline
│   └── test_retriever.py       offline
├── main.py                     CLI entrypoint
└── requirements.txt
```

## Possible extensions

- Swap or hybridize TF-IDF with dense embeddings for semantic matching
- Add an LLM-based re-ranking step over the top-N retrieved chunks
- Expand the knowledge base with a larger, more realistic policy corpus
- Add answer-groundedness checking against the retrieved context
