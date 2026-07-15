# AI Engineering Core — RAG & Vector Systems (Hands-On Reference)

> Principal/FDE-depth handbook. Each topic follows the same spine:
> **What it is → Why it matters → Hands-on code → Real-world use case → Production notes.**
> Code targets Python 3.11+. Library APIs (LangChain, RAGAS) move fast — pin versions and treat the imports as *shapes*, not gospel. Where a "best model" is implied, check the current **MTEB leaderboard** instead of trusting any name printed here.

---

## 0. The mental model (read this first)

A RAG system is a **search engine bolted to a text generator**. Every quality problem you'll ever debug lives in one of five layers:

```
[ Ingest ] → [ Chunk ] → [ Embed + Index ] → [ Retrieve ] → [ Generate ]
   parse       split         vectors            search        LLM answer
```

Rule of thumb for triage: **80% of "the LLM is dumb" complaints are actually retrieval failures.** If the right chunk never enters the context window, no prompt engineering saves you. Debug retrieval before you touch the prompt.

---

## 1. Embeddings and Vector Databases

**What it is.** An embedding is a fixed-length float vector (typically 384–3072 dims) that encodes *meaning*. Semantically similar text lands close together in the vector space; "loan default" and "borrower failed to repay" sit near each other even with zero shared words.

**Why it matters.** Keyword search matches strings; embeddings match *concepts*. This is what lets a user ask "what happens if I miss a payment?" and retrieve a clause titled "Consequences of Delinquency."

**Hands-on — generate and compare embeddings.**

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Small, fast, strong baseline. Swap for a larger model when recall matters.
model = SentenceTransformer("BAAI/bge-small-en-v1.5")   # 384 dims

docs = [
    "Borrower failed to repay the loan on the due date.",
    "The recipe calls for two cups of flour and one egg.",
    "Late repayment triggers a penalty and affects credit score.",
]
query = "What happens if I default on my loan?"

# normalize_embeddings=True => cosine similarity == dot product (faster, cleaner)
doc_vecs = model.encode(docs, normalize_embeddings=True)
q_vec    = model.encode(query, normalize_embeddings=True)

scores = doc_vecs @ q_vec            # dot product across all docs
for d, s in sorted(zip(docs, scores), key=lambda x: -x[1]):
    print(f"{s:.3f}  {d}")
# The two loan sentences rank far above the recipe — that's semantics working.
```

**Dense vs sparse (know both).**
- **Dense** (what's above): learned, captures meaning, poor at exact tokens (SKU codes, IDs, rare acronyms like "BaFin §25a").
- **Sparse** (BM25, SPLADE): lexical, nails exact terms, blind to synonyms.
- Production answer is almost always **hybrid** (§9).

**Distance metrics.**
| Metric | Formula intuition | Use when |
|---|---|---|
| Cosine | angle between vectors | text embeddings (normalize first) |
| Dot product | cosine × magnitudes | same as cosine *if* normalized |
| Euclidean (L2) | straight-line distance | some image/audio models |

Mismatching the metric to how the model was trained silently wrecks recall. bge/e5/OpenAI → cosine.

**Real-world use case.** Support-ticket dedup: embed every incoming ticket, and if it's within cosine 0.92 of an existing open ticket, auto-link them. Cut duplicate triage volume without a single keyword rule.

**Production notes.**
- **Model choice drives everything downstream.** A weak embedder caps your ceiling; no reranker recovers information the embedder threw away.
- **Instruction-tuned embedders** (e5, bge, Voyage) expect a prefix like `"query: ..."` vs `"passage: ..."`. Forgetting it costs several points of recall. Read the model card.
- **Version-lock the embedder.** Re-embedding 50M chunks because someone bumped a model minor version is a very bad Tuesday. The embedding model is part of your schema.
- **Dimensionality is a cost lever.** 3072-dim vectors are 8× the RAM of 384-dim. Matryoshka embeddings (OpenAI `text-embedding-3`, Nomic) let you truncate dims with graceful quality loss — store 512, not 3072, if recall holds.

---

## 2. Chunking Strategies

**What it is.** Splitting source documents into retrievable units. The chunk is the atomic unit of retrieval — you retrieve chunks, not documents.

**Why it matters.** Chunk too big → you retrieve a 4-page section to answer a one-line question, diluting the signal and burning context tokens. Chunk too small → you sever a clause from the condition that governs it and retrieve a half-truth. This single knob moves answer quality more than almost anything else.

**The five strategies, in order of sophistication.**

### 2a. Fixed-size (baseline — don't ship this)
```python
def fixed_chunks(text, size=800, overlap=100):
    step = size - overlap
    return [text[i:i+size] for i in range(0, len(text), step)]
```
Splits mid-word, mid-sentence. Only acceptable as a smoke test.

### 2b. Recursive character (the sane default)
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    # Tries to split on the biggest boundary that fits, in order:
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_text(long_document)
```
Respects paragraph → sentence → word boundaries. Ships in 80% of real systems.

### 2c. Document-structure-aware (use the format's grammar)
Markdown headers, HTML tags, PDF sections, code AST. Keep a header trail as metadata so a chunk knows it belongs to "Article 6 › Prohibited Practices."
```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
)
# Each chunk carries {"h1": "...", "h2": "..."} — priceless for filtering & citations.
docs = splitter.split_text(markdown_text)
```

### 2d. Semantic chunking (split where meaning shifts)
Embed sentences, cut where adjacent-sentence similarity drops below a percentile threshold.
```python
import numpy as np
from sentence_transformers import SentenceTransformer

def semantic_chunks(sentences, model, pct=90):
    vecs = model.encode(sentences, normalize_embeddings=True)
    dists = [1 - float(vecs[i] @ vecs[i+1]) for i in range(len(vecs)-1)]
    breakpoint = np.percentile(dists, pct)     # split at biggest topic shifts
    chunks, cur = [], [sentences[0]]
    for i, d in enumerate(dists):
        if d > breakpoint:
            chunks.append(" ".join(cur)); cur = []
        cur.append(sentences[i+1])
    chunks.append(" ".join(cur))
    return chunks
```
Great for prose that wanders (research papers, transcripts). Costs an embedding pass at ingest.

### 2e. Parent-document / sentence-window (retrieve small, return big)
Embed *small* precise chunks for retrieval accuracy, but feed the LLM the *surrounding* parent chunk for context. Best of both worlds.
```python
# Store: child chunks (embedded) each pointing to a parent_id.
# Retrieve on children -> dereference to unique parents -> send parents to LLM.
retrieved_child_ids = search(query, k=8)
parent_ids = {store[c]["parent_id"] for c in retrieved_child_ids}
context = [parent_store[p] for p in parent_ids]
```

**Choosing per document type.**
| Content | Strategy | Why |
|---|---|---|
| Legal / regulatory (RBI, EU AI Act) | structure-aware + parent-doc | clauses must stay whole; cite the article |
| Chat / call transcripts | semantic | topic drifts; no headers |
| Source code | AST/function-level | never split a function body |
| Product docs / wikis | markdown-header | structure is already there |
| Scientific PDFs | semantic + table extraction | dense, wandering prose + tables |

**Real-world use case.** A loan-agreement Q&A bot: structure-aware chunking keeps "Section 7: Default & Acceleration" intact and tags it, so when a borrower asks about missed payments you retrieve the *whole* enforceable clause plus a clean citation ("Agreement §7.2") — not a fragment that omits the 30-day cure period.

**Production notes.**
- **Overlap** (10–20% of chunk size) is cheap insurance against boundary-severed facts.
- **Attach metadata at chunk time**, not later: `source`, `page`, `section`, `created_at`, `doc_version`. You cannot retrofit context you didn't capture.
- **Tables and figures break naive chunkers.** Extract tables separately (see §8) and store a text serialization + the header trail.
- Re-chunking = re-embedding = re-indexing. Treat chunk parameters as a versioned decision.

---

## 3. Indexing Methods (ANN algorithms)

**What it is.** How the database *organizes* vectors so it can find nearest neighbors without comparing the query to all N vectors. This is Approximate Nearest Neighbor (ANN) search — you trade a sliver of recall for orders-of-magnitude speed.

**Why it matters.** Exact search over 10M × 768-dim vectors is ~30 GB of dot products *per query*. ANN turns that into sub-millisecond lookups. The index type is the single biggest lever on the latency/recall/RAM triangle.

**The methods.**

| Method | Idea | Recall | Speed | RAM | When |
|---|---|---|---|---|---|
| **Flat** | brute force, compare all | 100% | slow at scale | high | <100k vectors, or a ground-truth baseline |
| **IVF** | cluster space; search nearest clusters | tunable | fast | medium | millions, batch-ish |
| **HNSW** | navigable small-world graph | high | very fast | high | low-latency online serving (default) |
| **PQ** | compress vectors to codes | lower | fast | very low | RAM-constrained, huge N |
| **IVF-PQ** | cluster + compress | good | fast | low | 100M+ on commodity hardware |
| **DiskANN** | graph on SSD | high | fast | low RAM | billion-scale, budget RAM |

**Hands-on — the same data, three indexes (FAISS).**
```python
import faiss, numpy as np

d = 384
xb = np.random.rand(200_000, d).astype("float32")   # stand-in corpus
xq = np.random.rand(5, d).astype("float32")

# 1) Flat — exact, your correctness oracle
flat = faiss.IndexFlatIP(d)          # inner product (normalize for cosine)
flat.add(xb)

# 2) HNSW — the online-serving workhorse
hnsw = faiss.IndexHNSWFlat(d, 32)    # M=32 neighbors/node
hnsw.hnsw.efConstruction = 200       # build-time quality
hnsw.add(xb)
hnsw.hnsw.efSearch = 64              # query-time recall/latency dial
D, I = hnsw.search(xq, k=5)

# 3) IVF-PQ — memory-frugal for huge corpora
quantizer = faiss.IndexFlatIP(d)
ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist=1024, m=48, nbits=8)
ivfpq.train(xb)                      # PQ must learn codebooks first
ivfpq.add(xb)
ivfpq.nprobe = 16                    # clusters to scan (recall/speed dial)
D, I = ivfpq.search(xq, k=5)
```

**The two knobs you'll actually tune.**
- **HNSW:** `M` (graph connectivity, ↑ = better recall, more RAM) and `efSearch` (↑ = better recall, slower). Raise `efSearch` at query time until recall plateaus, then stop.
- **IVF:** `nlist` (number of clusters ≈ √N) and `nprobe` (clusters scanned per query).

**Real-world use case.** Codebase semantic search across a 4M-function monorepo on a single 32 GB box: IVF-PQ compresses vectors ~16×, fitting in RAM, and returns "find the function that validates IBAN checksums" in under 10 ms — impossible with a flat index at that scale.

**Production notes.**
- **Always keep a flat index on a data sample** to measure real recall@k. Without ground truth you're tuning blind.
- HNSW **inserts are cheap, deletes are not** — tombstone + periodic rebuild. High-churn corpora hate HNSW.
- PQ throws away precision. If your embeddings are already low-dim, PQ can crater recall — measure, don't assume.

---

## 4. Vector Databases

**What it is.** A datastore purpose-built to index vectors *and* filter on metadata *and* handle updates, replication, and multi-tenancy — everything raw FAISS makes you build yourself.

**Why it matters.** FAISS is a library; a vector DB is a system. Persistence, concurrent writes, metadata filters, hybrid search, and horizontal scale are the difference between a notebook demo and a service.

**The landscape (choose deliberately).**
| DB | Sweet spot | Watch-out |
|---|---|---|
| **pgvector** (Postgres) | you already run Postgres; want SQL joins + vectors in one place; strong consistency | tune HNSW params; scaling = scaling Postgres |
| **Qdrant** | rich filtering, Rust performance, great DX, self-host or cloud | newer ecosystem |
| **Weaviate** | built-in hybrid search, modules, GraphQL | opinionated schema |
| **Milvus** | billion-scale, GPU indexing | operationally heavy |
| **Pinecone** | fully managed, zero ops | cost, vendor lock-in, data residency |
| **Chroma** | prototyping, local-first | not for heavy prod |
| **Redis (vector)** | already using Redis; low latency | RAM-bound |

**For your world (BFSI / EU data residency):** default to **pgvector or self-hosted Qdrant**. Fully managed clouds complicate BaFin/DPDP data-residency and audit stories; keeping vectors inside your own Postgres/Qdrant in-region is the path of least regulatory resistance.

**Hands-on — Qdrant end-to-end.**
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = QdrantClient(":memory:")          # swap for url="http://localhost:6333"

client.create_collection(
    "policies",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

docs = [
    {"text": "Missed EMI incurs a late fee after a 15-day grace period.", "category": "loans", "region": "IN"},
    {"text": "KYC must be re-verified every 24 months for high-risk accounts.", "category": "compliance", "region": "IN"},
]
client.upsert("policies", points=[
    PointStruct(id=i, vector=model.encode(d["text"]).tolist(),
                payload=d)                    # payload = filterable metadata
    for i, d in enumerate(docs)
])

# Vector search WITH a metadata pre-filter (see §10)
hits = client.query_points(
    "policies",
    query=model.encode("what if I miss a payment?").tolist(),
    query_filter=Filter(must=[FieldCondition(key="category", match=MatchValue(value="loans"))]),
    limit=3,
).points
for h in hits:
    print(h.score, h.payload["text"])
```

**Hands-on — pgvector (SQL-native).**
```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
    id         bigserial PRIMARY KEY,
    content    text,
    category   text,
    doc_version int,
    embedding  vector(384)
);

-- HNSW index for cosine distance
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Retrieval: filter in WHERE, order by vector distance. One query, transactional.
SELECT content
FROM chunks
WHERE category = 'loans' AND doc_version = 3
ORDER BY embedding <=> $1        -- <=> is cosine distance
LIMIT 5;
```

**Real-world use case.** Multi-tenant SaaS knowledge base: each customer's docs isolated by a `tenant_id` payload filter in Qdrant. One collection, hard tenant isolation at query time, no cross-tenant leakage — the filter *is* the security boundary.

**Production notes.**
- **"Which DB" matters less than you think** once you clear ~1M vectors with metadata filters. Pick on ops fit (do you run Postgres already?) and compliance, not benchmarks.
- **The metadata schema is as important as the vectors.** Design it up front (§10).
- Multi-tenancy: prefer **one collection + tenant filter** over collection-per-tenant until you have thousands of tenants.

---

## 5. Vector DB Performance Optimization

**What it is.** Getting acceptable **recall** at target **latency** and **cost** — the three-way tug-of-war every vector system fights.

**Why it matters.** A RAG API that answers in 3 seconds feels broken; one that misses the right chunk 20% of the time *is* broken. You optimize both simultaneously.

**The levers, most-to-least impactful.**

**1. Index parameters.**
```python
# HNSW: raise efSearch until recall@10 plateaus, then back off for latency.
for ef in (16, 32, 64, 128, 256):
    index.hnsw.efSearch = ef
    recall, p95_ms = benchmark(index, queries, ground_truth)
    print(ef, recall, p95_ms)      # pick the knee of the curve
```

**2. Quantization (RAM ÷ 4 to ÷ 32).**
- **Scalar (int8):** ~4× smaller, tiny recall loss. Almost always worth it.
- **Binary:** ~32× smaller, big speedup; pair with a float rerank pass to recover precision.

**3. Filter *before* you search** (pre-filtering). Searching 5M vectors then discarding 4.9M by category is wasteful and can starve top-k. Push the filter into the ANN traversal (§10).

**4. Hybrid search** (dense + BM25). Free recall on exact terms (SKUs, statute numbers) the dense vector fumbles (§9).

**5. Cache.** Embedding the same query twice is pure waste.
```python
from functools import lru_cache

@lru_cache(maxsize=10_000)
def embed_cached(text: str) -> tuple:
    return tuple(model.encode(text, normalize_embeddings=True).tolist())
```
Also cache full (query → retrieved-ids) for hot queries; invalidate on re-index.

**6. Sharding & replication.** Shard by vector count for throughput; replicate for QPS + HA. Route filtered queries to relevant shards when you shard by a metadata key (e.g., region).

**The one discipline that matters: measure recall against ground truth.**
```python
def recall_at_k(approx_ids, exact_ids, k=10):
    hits = len(set(approx_ids[:k]) & set(exact_ids[:k]))
    return hits / k
# exact_ids come from a Flat index over a query sample. No oracle => flying blind.
```

**Real-world use case.** A customer-facing policy assistant had a p95 of 1.8 s and 74% recall@5. Int8 quantization (−0.5% recall, −60% RAM) + `efSearch` retuned to the knee + query-embedding cache dropped p95 to 240 ms at 91% recall — same hardware, users stopped complaining.

**Production notes.**
- **Optimize in this order:** correct metric → right index → tune params → quantize → cache → shard. Don't shard to fix what a parameter tweak solves.
- **p95/p99, not mean.** Tail latency is what users feel.
- Warm the index after a deploy; cold HNSW graphs are slow on the first queries.

---

## 6. What is RAG?

**What it is.** Retrieval-Augmented Generation: fetch relevant text from *your* corpus at query time and inject it into the LLM's prompt so the answer is grounded in that text rather than the model's frozen, generic memory.

**Why it matters.** It's the cheapest, fastest way to make an LLM answer about *your* private, current, proprietary data — and to make it **cite sources** and **say "I don't know"** instead of hallucinating. No retraining, updates land the moment you re-index.

**Hands-on — a complete minimal RAG in ~25 lines.**
```python
from sentence_transformers import SentenceTransformer
import numpy as np, anthropic

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = anthropic.Anthropic()

corpus = [
    "EMI late fee applies after a 15-day grace period, capped at 2% of the instalment.",
    "Foreclosure of a floating-rate retail loan carries no prepayment penalty.",
    "KYC re-verification is required every 24 months for high-risk accounts.",
]
C = model.encode(corpus, normalize_embeddings=True)

def rag_answer(question, k=2):
    q = model.encode(question, normalize_embeddings=True)
    top = np.argsort(-(C @ q))[:k]                      # retrieve
    context = "\n".join(f"[{i}] {corpus[i]}" for i in top)
    prompt = (
        "Answer ONLY from the context. If it's not there, say you don't know. "
        f"Cite the [number].\n\nContext:\n{context}\n\nQuestion: {question}"
    )
    msg = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text

print(rag_answer("Is there a penalty for prepaying my home loan?"))
# → grounded answer citing [1], not a generic guess.
```

**The grounding contract (this is the whole point).**
1. Instruct: *answer only from context*.
2. Instruct: *if the context doesn't cover it, say so*.
3. Instruct: *cite*.
Together these convert an eloquent bluffer into an auditable answerer.

**Real-world use case.** An internal compliance copilot over 4,000 pages of RBI circulars and internal SOPs. An officer asks "what's the current co-lending exposure cap?"; RAG retrieves the exact revised clause and answers with a citation an auditor can follow back to the source — a use case where a hallucinated number is a regulatory incident, not a bug ticket.

**Production notes.**
- **RAG shrinks but does not eliminate hallucination.** The model can still misread retrieved text. Faithfulness must be *measured* (§12), not assumed.
- **Garbage retrieval → confident garbage answer.** The generator is downstream of everything in §1–5.
- Always return the sources alongside the answer. Ungrounded answers users can't verify are a liability, especially in regulated domains.

---

## 7. RAG vs Fine-Tuning

**What it is.** Two different tools for two different problems. RAG changes *what the model knows*; fine-tuning changes *how the model behaves*.

**Why it matters.** Teams routinely reach for the expensive one (fine-tuning) to solve a knowledge problem (which RAG solves better and cheaper), then wonder why the model still gets facts wrong.

**The decision matrix.**
| Need | Use | Why |
|---|---|---|
| Answer about private/current facts | **RAG** | inject knowledge at query time; update by re-indexing |
| Enforce a tone, format, persona | **Fine-tune** | bake style into weights |
| Cite sources / audit trail | **RAG** | retrieved chunks are the citation |
| Teach a narrow output schema (e.g., always emit valid FpML) | **Fine-tune** | reliable structure |
| Knowledge changes weekly | **RAG** | re-index, no retraining |
| Specialized skill/reasoning pattern | **Fine-tune** (or both) | shift the model's behavior |
| Reduce prompt length / latency | **Fine-tune** | move instructions into weights |

**The honest summary:** **RAG for knowledge; fine-tuning for style, format, and skill.** They compose — fine-tune a model to *speak like your compliance team and always cite*, then RAG feeds it the current regulations.

**Cost & operational reality.**
| | RAG | Fine-tuning |
|---|---|---|
| Upfront cost | low (build a pipeline) | high (data curation, training) |
| Update cost | re-index (minutes) | retrain (hours/days + $) |
| Knowledge freshness | real-time | frozen at training |
| Hallucination control | strong (grounding + cite) | weak (still confabulates) |
| Latency | +retrieval hop | none added |
| Data residency | data stays in your DB | training data leaves for the trainer |

**Real-world use case.** A multilingual loan-servicing assistant: **fine-tune** a small model so it always replies in the customer's language with the mandated regulatory disclaimer and a fixed JSON envelope; **RAG** feeds it the borrower's current account terms. Fine-tuning owns *how it talks*, RAG owns *what's true today*.

**Production notes.**
- Try **RAG + prompt engineering first.** Fine-tune only when you've proven a style/format/skill gap RAG can't close.
- Fine-tuning on facts is a trap: the model memorizes fuzzily and can't cite, and the facts go stale the day training ends.
- **Both together** is the mature endgame, not the starting point.

---

## 8. Data Ingestion

**What it is.** The pipeline that turns messy sources — PDFs, databases, APIs, web pages, scanned images — into clean, chunked, metadata-tagged, embedded records. **PDFs, DBs, APIs, web pages → chunks → vectors.**

**Why it matters.** This is where most RAG projects quietly die. A brilliant retriever over garbage-parsed PDFs (headers/footers bleeding into text, tables flattened into word soup) produces confident nonsense. Ingestion quality caps system quality.

**Hands-on — a multi-source ingestion pipeline.**
```python
import hashlib, requests, io
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)

def load_pdf(path):
    reader = PdfReader(path)
    for pageno, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if text:
            yield {"text": text, "source": path, "page": pageno}

def load_web(url):
    from bs4 import BeautifulSoup
    html = requests.get(url, timeout=15).text
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["nav", "footer", "script", "style"]):   # strip chrome
        tag.decompose()
    yield {"text": soup.get_text(" ", strip=True), "source": url}

def load_db(conn):                                            # SQL → docs
    for row in conn.execute("SELECT id, body, updated_at FROM articles"):
        yield {"text": row["body"], "source": f"db://articles/{row['id']}",
               "updated_at": str(row["updated_at"])}

def to_chunks(records):
    seen = set()
    for rec in records:
        for chunk in splitter.split_text(rec["text"]):
            h = hashlib.sha256(chunk.encode()).hexdigest()   # dedup identical chunks
            if h in seen:
                continue
            seen.add(h)
            yield {**{k: v for k, v in rec.items() if k != "text"},
                   "text": chunk, "chunk_hash": h}
```

**The hard parts nobody warns you about.**
- **PDF tables & multi-column layouts:** naive extractors scramble reading order and flatten tables. Use layout-aware tools (Unstructured, LlamaParse, Docling, Amazon Textract) and store tables as separate, clearly-typed chunks.
- **Scanned docs need OCR** (Tesseract, PaddleOCR, or a cloud OCR). Detect image-only PDFs and route them through OCR before chunking.
- **Incremental / delta ingestion:** re-embedding everything nightly doesn't scale. Track a content hash per source; only re-process changed items; propagate deletes.
```python
def sync(source_id, new_hash, store):
    old = store.get_hash(source_id)
    if old == new_hash:
        return "unchanged"
    store.delete_chunks(source_id)      # remove stale vectors first
    return "reprocess"                  # then re-ingest
```
- **Dedup** (hash-level and near-dup via embedding) stops the same boilerplate disclaimer from flooding every retrieval.

**Real-world use case.** Ingesting a regulator's PDF circulars: layout-aware parsing preserves the numbered clause structure, tables of rate caps become queryable table-chunks, and a per-file content hash means when the regulator revises one circular you re-embed *that file only* — and old vectors are deleted so the bot never cites a superseded rule.

**Production notes.**
- **Ingestion is a data pipeline, not a script.** Give it idempotency, retries, dead-letter handling, and observability (docs in, chunks out, failures logged).
- Capture provenance (`source`, `page`, `doc_version`, `ingested_at`) — it's your citation and your audit trail.
- **Test parsing on your ugliest real documents**, not clean samples. The 12-column scanned table is the one that matters.

---

## 9. Retrieval Strategies

**What it is.** How you select which chunks reach the LLM. **Top-K** (nearest neighbors), **MMR** (nearest *and diverse*), and **re-ranking** (retrieve broadly, then reorder with a stronger model).

**Why it matters.** Plain top-k floods the context with five near-duplicate chunks and misses the one complementary fact. Diversity and reranking are what turn "retrieved *something* relevant" into "retrieved the *right* set."

### 9a. Top-K (baseline)
```python
top_ids = np.argsort(-(C @ q))[:k]     # k nearest. Simple, and the floor, not ceiling.
```

### 9b. MMR — Maximal Marginal Relevance (kill redundancy)
Balances relevance to the query against novelty vs already-picked chunks.
```python
def mmr(query_vec, doc_vecs, k=5, lambda_=0.6):
    selected, candidates = [], list(range(len(doc_vecs)))
    sim_q = doc_vecs @ query_vec
    while len(selected) < k and candidates:
        best, best_score = None, -1e9
        for c in candidates:
            redundancy = max((doc_vecs[c] @ doc_vecs[s]) for s in selected) if selected else 0
            score = lambda_ * sim_q[c] - (1 - lambda_) * redundancy
            if score > best_score:
                best, best_score = c, score
        selected.append(best); candidates.remove(best)
    return selected
# λ→1 pure relevance; λ→0 pure diversity. ~0.5–0.7 is the usual sweet spot.
```
Use when your corpus has heavy repetition (FAQs, versioned docs).

### 9c. Hybrid search + Reciprocal Rank Fusion (dense meets lexical)
```python
def rrf(dense_ranked, sparse_ranked, k=60):     # merge two ranked lists
    scores = {}
    for ranked in (dense_ranked, sparse_ranked):
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)
# dense finds meaning; BM25 finds exact tokens (statute numbers, IBANs). RRF fuses both.
```

### 9d. Re-ranking (the highest-ROI upgrade)
Retrieve 30–50 cheaply, then a cross-encoder scores each (query, chunk) *jointly* — far more accurate than the bi-encoder that fetched them. Keep the top 5.
```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-base")

def retrieve_rerank(query, candidates, keep=5):     # candidates: list[str] from ANN
    pairs = [(query, c) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = [c for c, _ in sorted(zip(candidates, scores), key=lambda x: -x[1])]
    return ranked[:keep]
```

**The strategy that wins in practice:** hybrid retrieve top-40 → RRF → cross-encoder rerank → top-5 → (optional MMR for diversity). Sounds heavy; the rerank pass is milliseconds and buys the biggest single jump in answer quality after fixing chunking.

**Real-world use case.** A legal search where a paralegal queries "penalty for delayed KYC under the 2025 rules." Dense retrieval nails the *concept* of penalties; BM25 guarantees the exact token "2025" and the rule number aren't lost; RRF fuses them; the cross-encoder floats the one clause that mentions both delay *and* penalty *and* 2025 to the top. Any single method alone drops one of those three constraints.

**Production notes.**
- **Reranking is usually the best bang-for-buck** after chunking. Add it before you try anything exotic in §11.
- Retrieve **wider than you keep** (fetch 40, feed 5) — you can't rerank what you never retrieved.
- Watch the **context budget**: more chunks ≠ better. Beyond ~5–8 focused chunks you invite "lost in the middle" and distraction. Precision > volume.

---

## 10. Metadata Filtering

**What it is.** Constraining retrieval with structured attributes — **filter by date, category, author, region, tenant, doc_version** — *before* (or alongside) the vector search.

**Why it matters.** Semantics alone can't express "only the *current* version," "only *this tenant's* docs," or "only *IN-region* policies." Metadata filtering is how you enforce correctness, recency, access control, and multi-tenancy. It's frequently the difference between "technically relevant" and "actually usable."

**Pre- vs post-filtering (know which your DB does).**
- **Pre-filter:** restrict the candidate set, *then* search vectors. Correct top-k, but naive implementations scan more. Good vector DBs push the filter into the ANN traversal.
- **Post-filter:** search vectors, then drop non-matching results. Can return *fewer than k* (or zero) if the filter is selective and matches got pruned early. Dangerous with tight filters.

**Hands-on — filtered retrieval (Qdrant).**
```python
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range, MatchAny

flt = Filter(
    must=[
        FieldCondition(key="tenant_id", match=MatchValue(value="acme")),      # isolation
        FieldCondition(key="region",    match=MatchValue(value="IN")),        # residency
        FieldCondition(key="doc_version", match=MatchValue(value=3)),         # current only
        FieldCondition(key="effective_date", range=Range(gte=1704067200)),    # since 2024
    ],
    should=[FieldCondition(key="category", match=MatchAny(any=["loans", "compliance"]))],
)

hits = client.query_points("policies", query=q_vec, query_filter=flt, limit=5).points
```

**Hands-on — filter in SQL (pgvector).**
```sql
SELECT content, source, page
FROM chunks
WHERE tenant_id = $1
  AND region = 'IN'
  AND doc_version = (SELECT MAX(doc_version) FROM chunks WHERE tenant_id = $1)  -- always current
ORDER BY embedding <=> $2
LIMIT 5;
```

**Designing the metadata schema (do this at chunk time).**
```python
{
  "text": "...chunk...",
  "source": "rbi_circular_2025_07.pdf",
  "page": 12,
  "section": "Co-Lending › Exposure Limits",
  "category": "compliance",
  "region": "IN",
  "tenant_id": "acme",
  "doc_version": 3,
  "effective_date": 1719792000,
  "author": "RBI",
  "supersedes": "rbi_circular_2024_11.pdf"
}
```
Index the high-cardinality/frequently-filtered fields. You can't filter on what you didn't store.

**Real-world use case.** A version-aware policy bot: every chunk carries `doc_version` and `effective_date`. When a regulation is revised, new chunks land as `version=4` and the query filters to the max version — so the bot **cannot** cite last year's superseded cap even though those vectors are still similar. The filter enforces correctness the embedding space can't.

**Production notes.**
- **Prefer pre-filtering** with a DB that pushes filters into ANN traversal; avoid post-filtering with selective filters (silent recall collapse).
- **Metadata filtering is your access-control boundary** in multi-tenant systems. Test it like a security control — a leaked `tenant_id` filter is a data breach, not a bug.
- Beware **over-filtering to zero results.** Handle the empty-set case gracefully (relax filters, tell the user) instead of hallucinating.

---

## 11. Advanced RAG

Techniques for when baseline RAG (chunk → embed → top-k → generate) plateaus. Add them **surgically**, driven by eval metrics (§12) — each adds latency and moving parts.

### 11a. Self-Query (let the LLM write the metadata filter)
The model parses a natural question into a semantic query **plus** structured filters.
```python
# User: "penalties in RBI circulars issued after 2024 about co-lending"
# LLM extracts →
{
  "semantic_query": "penalties for co-lending violations",
  "filters": {"author": "RBI", "category": "compliance",
              "effective_date": {"gte": "2024-01-01"}}
}
# You then run a filtered vector search (§10) from the extracted structure.
```
**Use when:** users naturally phrase constraints (dates, authors, categories) in their questions.

### 11b. HyDE — Hypothetical Document Embeddings
Embed the *question* and a real *answer passage* often live in different regions of vector space. So first have the LLM **hallucinate a plausible answer**, embed *that*, and search with it — the fake answer sits nearer real answers than the question does.
```python
def hyde_retrieve(question, k=5):
    draft = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=200,
        messages=[{"role": "user",
                   "content": f"Write a short passage that would answer: {question}"}],
    ).content[0].text
    q = model.encode(draft, normalize_embeddings=True)   # search with the hypothesis
    return np.argsort(-(C @ q))[:k]
```
**Use when:** queries are short/vague and recall is weak. **Cost:** an extra LLM call per query. (The hallucination is fine — it's never shown, only used to steer retrieval.)

### 11c. Corrective RAG (CRAG) — grade, then act
Grade retrieved chunks for relevance; if they're weak, *correct* (rewrite the query, widen search, or fall back to web/other source) before generating.
```python
def crag(question):
    chunks = retrieve(question, k=5)
    grade = llm_grade_relevance(question, chunks)     # LLM returns high/ambiguous/low
    if grade == "low":
        chunks = web_search(question)                 # fallback source
    elif grade == "ambiguous":
        chunks = retrieve(rewrite_query(question), k=8)  # try harder
    return generate(question, chunks)
```
**Use when:** your corpus has coverage gaps and you'd rather widen the net than confidently answer from irrelevant context.

### 11d. RAPTOR — recursive hierarchical summaries
Cluster chunks, summarize each cluster, cluster the summaries, summarize again — a **tree** from raw chunks (leaves) to document-wide themes (root). Retrieve across levels, so broad questions hit summary nodes and specific ones hit leaves.
```
Level 2:        [ whole-corpus themes ]
Level 1:   [ topic summary ] [ topic summary ]
Level 0: chunk chunk chunk   chunk chunk chunk
```
**Use when:** questions span granularities — "summarize our entire lending policy" (needs the root) *and* "what's the exact grace period?" (needs a leaf).

### 11e. Contextual Retrieval (prepend context before embedding)
Each chunk is embedded *with* a one-line LLM-generated description of where it sits in the document, so an isolated "…this cap is 5%…" becomes "In the co-lending exposure section of the 2025 RBI circular, this cap is 5%…" — vastly more retrievable and unambiguous.
```python
def contextualize(chunk, full_doc):
    ctx = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=80,
        messages=[{"role": "user", "content":
            f"Give a one-sentence context locating this chunk in the document.\n\n"
            f"DOC:\n{full_doc[:4000]}\n\nCHUNK:\n{chunk}"}],
    ).content[0].text
    return f"{ctx}\n\n{chunk}"      # embed this enriched text
```
**Use when:** chunks lose meaning out of context (pronouns, "this section," dangling references) — extremely common in legal/technical corpora.

**Choosing what to add (don't stack them all):**
| Symptom | Reach for |
|---|---|
| Users phrase filters in the question | Self-Query |
| Short/vague queries, weak recall | HyDE |
| Corpus has coverage gaps | Corrective RAG |
| Questions mix broad + narrow scope | RAPTOR |
| Chunks meaningless in isolation | Contextual Retrieval |

**Real-world use case.** A regulatory copilot combined **Contextual Retrieval** (so each clause knew its parent circular) with **Self-Query** (so "co-lending caps changed after Nov 2024" auto-filtered on date + topic). Faithfulness climbed because chunks stopped being ambiguous and stale versions stopped surfacing — measured, not guessed (§12).

**Production notes.**
- **Every technique here adds latency and failure modes.** Add one, measure the delta with RAGAS, keep it only if the numbers move.
- Order of ROI in most systems: **fix chunking → add reranking → contextual retrieval → the rest.** Exotic first is premature optimization.

---

## 12. RAG Evaluation

**What it is.** Measuring RAG quality along axes that isolate *where* it fails: **Faithfulness** (is the answer supported by retrieved context?), **Answer Relevance** (does it address the question?), **Context Precision/Recall** (did retrieval fetch the right chunks?). RAGAS is the common framework.

**Why it matters.** "It looks good in the demo" is not a metric. Without evaluation you can't tell a retrieval bug from a generation bug, can't safely ship changes, and can't put a number in front of a risk/compliance committee. **You cannot improve what you don't measure** — this is the difference between a demo and a product.

**The metrics decoded.**
| Metric | Question it answers | A low score means |
|---|---|---|
| **Faithfulness** | Is every claim grounded in the retrieved context? | the LLM is hallucinating beyond its sources |
| **Answer Relevance** | Does the answer actually address the question? | on-topic-ish rambling or evasion |
| **Context Precision** | Are the *top* retrieved chunks the relevant ones? | reranking / ordering problem |
| **Context Recall** | Did retrieval fetch *all* needed chunks? | chunking / embedding / top-k problem |

**The diagnostic power is in combining them:**
- High faithfulness + low answer relevance → grounded but unhelpful (wrong chunks retrieved).
- Low faithfulness + high context recall → right chunks fetched, LLM ignored them (prompt/model issue).
- Low context recall → fix §2/§3/§9, not the prompt.

**Hands-on — RAGAS evaluation.**
```python
# API surface changes across RAGAS versions — treat as a shape, pin your version.
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

eval_data = Dataset.from_dict({
    "question":     ["Is there a prepayment penalty on floating-rate home loans?"],
    "answer":       ["No. Floating-rate retail loan foreclosure carries no penalty. [1]"],
    "contexts":     [["Foreclosure of a floating-rate retail loan carries no prepayment penalty."]],
    "ground_truth": ["There is no prepayment penalty on floating-rate retail home loans."],
})

result = evaluate(
    eval_data,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)
print(result)   # e.g. {'faithfulness': 1.0, 'answer_relevancy': 0.95, ...}
```

**Build a golden dataset (the real work).**
```python
# 30–200 curated (question, ground_truth_answer, ground_truth_context) triples
# drawn from REAL user questions + edge cases + known-hard queries.
# This is your regression suite. Grow it every time prod produces a bad answer.
golden = [
    {"question": "...", "ground_truth": "...", "ground_truth_context": "..."},
    # ...
]
```

**Wire it into CI (the discipline that compounds).**
```python
def gate(results, thresholds):
    failures = {m: v for m, v in results.items() if v < thresholds.get(m, 0)}
    assert not failures, f"RAG quality regression: {failures}"

gate(result, {"faithfulness": 0.90, "answer_relevancy": 0.85,
              "context_recall": 0.80, "context_precision": 0.75})
# Now a chunking tweak that quietly tanks recall fails the build instead of prod.
```

**LLM-as-judge (complement, cross-check).** RAGAS itself uses an LLM judge under the hood; you can also roll targeted judges (e.g., "does this answer contain a regulatory citation?"). Validate judges against human labels on a sample — judges have biases (length, position) and drift with model updates.

**Real-world use case.** Before shipping a compliance assistant, the team ran 120 golden questions nightly. A model upgrade *raised* answer relevance but *dropped* faithfulness from 0.94 to 0.88 — it started smoothing over gaps in the regulations with plausible-sounding filler. The CI gate caught it; they pinned the model and added stricter grounding instructions. In a regulated setting that catch prevented an auditable false statement from reaching a user.

**Production notes.**
- **Evaluate retrieval and generation separately.** Context recall/precision isolate retrieval; faithfulness/relevance isolate generation. Blended "it's bad" tells you nothing actionable.
- **Golden datasets decay.** As the corpus and users evolve, refresh them — a stale eval set gives false confidence.
- **Log production queries and sample them for offline eval.** Real failures live in the long tail you never imagined at design time.
- Track metrics over time; a dashboard that shows faithfulness trending down *before* users revolt is worth more than any single benchmark.

---

## Appendix — the end-to-end path, in order

When you build (or debug) a RAG system, walk these in sequence:

1. **Ingest** cleanly (§8) — layout-aware parsing, provenance, dedup, delta sync.
2. **Chunk** to fit the content's grammar (§2) — structure-aware + parent-doc for regulated text.
3. **Embed** with an appropriate, version-locked model (§1); design **metadata** now (§10).
4. **Index** for your scale/latency (§3) in a DB that fits your ops & compliance (§4).
5. **Retrieve** hybrid → rerank → (MMR) (§9), with **metadata filters** for correctness & isolation (§10).
6. **Generate** with a strict grounding contract and citations (§6).
7. **Measure** faithfulness / relevance / context recall & precision, gated in CI (§12).
8. Only then add **advanced** techniques (§11), one at a time, kept only if the metrics move.
9. **Optimize** performance last (§5) — metric → index → params → quantize → cache → shard.

> The single highest-leverage habit: **measure retrieval before you blame the model.** Most "the LLM is wrong" tickets are chunking and retrieval tickets wearing a costume.
