# Core-CS Components: Hybrid Retrieval Engine + Legal Authority Graph

Two algorithmic components were added on top of the AI Legal Council so the system
demonstrates *owned* computer-science work (Information Retrieval and Graph Algorithms),
not just API orchestration. Both are **off by default** — the production pipeline is
unchanged unless the feature flags are flipped.

---

## 1. Hybrid Retrieval Engine (Information Retrieval)

Instead of relying solely on Qdrant's vector search, retrieval now fuses two retrievers
and reranks the result with algorithms implemented in-house:

- **BM25** — a lexical scorer over an inverted index we build from the corpus
  (`scripts/build_bm25_index.py`), one index per collection.
- **Reciprocal Rank Fusion (RRF)** — fuses the dense (semantic) and BM25 (lexical)
  rankings: `score(d) = Σ 1/(k + rank_i(d))`.
- **Maximal Marginal Relevance (MMR)** — greedy diversity-aware reranking (implemented,
  benchmarked, and — based on the results below — left **off** for this corpus).

### Measured results (TREC-style pooled-relevance evaluation, 20 labelled queries)

| Method | nDCG@10 | MAP | nDCG@5 |
|--------|:-------:|:---:|:------:|
| Dense (vector only) — *previous system* | 0.916 | 0.573 | 0.931 |
| BM25 (lexical only) | 0.946 | 0.668 | 0.978 |
| **Hybrid (RRF)** — *new default* | **0.961** | **0.904** | 0.959 |
| Hybrid + MMR | 0.911 | 0.470 | 0.896 |

**Headline:** hybrid RRF beats the previous dense-only retriever on every ranking
metric — most dramatically on **MAP (0.573 → 0.904)**, because fusing lexical and
semantic signals surfaces *all* relevant provisions, not just the top one.

**A real finding to defend:** MMR *reduced* ranking quality here (MAP 0.470). Its
diversity penalty demotes near-duplicate legal provisions — exactly the documents we
want clustered at the top — so we measured it and chose RRF for production. (This is
rigorous evaluation, not a failure: we let the data pick the algorithm.)

Reproduce: `venv/Scripts/python.exe -m scripts.eval_retrieval`
→ `output/retrieval_metrics.{md,csv}`.

---

## 2. Legal Authority Graph (Graph Algorithms)

The case corpus has **no stored case-to-case citations**, so we build a **case ↔ statute
bipartite graph** (a case links to every statute it applies, from `summary.laws_applied`)
and analyse it (`scripts/build_authority_graph.py`).

| Graph | Value |
|-------|-------|
| Nodes / Edges | 56,596 / 93,509 |
| Cases / distinct statutes | 38,806 / 17,790 |
| Communities (Louvain) | 1,522 |

- **PageRank** → per-case *authority* score (used to re-rank retrieved cases).
- **HITS** → statute authority ranking. Top results: **Constitution of India → Indian
  Penal Code → Code of Criminal Procedure → Code of Civil Procedure → Evidence Act** —
  i.e. the algorithm independently identifies the foundational statutes (face validity).
- **Louvain community detection** rediscovered the structure of Indian law unsupervised:
  distinct clusters for **criminal** (IPC/CrPC/Evidence), **civil & property**
  (CPC/Transfer of Property/Limitation/Contract), **constitutional**, **tax**
  (Income Tax/Finance/Wealth Tax), **labour** (Industrial Disputes/Factories/ESI), and
  **land acquisition** law.

**Design choice worth highlighting:** PageRank/HITS run directly on the *bipartite*
graph, which avoids the O(n²) edge explosion a case-case projection would create from
"hub" statutes like the Constitution (applied in 9,728 cases).

**Demo:** with the authority blend on, a query for "right to privacy" surfaces the actual
landmark precedents (*Puttaswamy*, *Kharak Singh*, *PUCL*) and promotes the
higher-authority judgments. Note: the authority blend does not move the aggregate IR
metrics above, because the case queries already retrieve relevant docs at the top
(binary relevance saturates); its value is corpus analysis + breaking ties toward
landmark precedent.

Reproduce: `venv/Scripts/python.exe -m scripts.build_authority_graph`
→ `data/graph_stats.json`, `data/authority_scores.json`, `data/case_graph.graphml`
(open the GraphML in Gephi for a slide visual).

---

## How it plugs in (and why nothing breaks)

`council.py` now calls `retrieval_service` instead of `qdrant_service`. The dispatcher
branches on config flags in `app/config.py`:

| Flag | Default | Effect |
|------|:-------:|--------|
| `RETRIEVAL_MODE` | `"dense"` | `"dense"` = exact previous behaviour; `"hybrid"` = the engine above |
| `USE_MMR` | `False` | enable MMR diversity reranking (benchmarked as worse here) |
| `AUTHORITY_WEIGHT` | `0.0` | `>0` blends PageRank authority into case ranking |

With defaults, output is **byte-for-byte identical** to the previous system (verified by
a regression check). The returned chunk shape `{rank, score, text, metadata}` is
unchanged; hybrid adds an additive `metadata.retrieval` provenance block
(`dense_rank`, `bm25_rank`, `rrf_score`, `authority`) that the frontend safely ignores.
BM25 indexes and authority scores load lazily; if the artefacts are absent, the system
falls back to dense.

### One-time build (offline, no LLM/API cost)
```bash
cd backend-ai-lawyer
venv/Scripts/python.exe -m scripts.build_bm25_index        # ~2.5 min, builds data/bm25_*.pkl
venv/Scripts/python.exe -m scripts.build_authority_graph   # ~1 min, builds data/authority_scores.json
```

### Enable for a live demo
Set in `.env`: `RETRIEVAL_MODE=hybrid` and (optional) `AUTHORITY_WEIGHT=0.3`.

## New / changed files
```
app/services/retrieval.py        dispatcher (dense | hybrid)
app/services/hybrid.py           BM25 + RRF + MMR + authority blend
app/services/bm25_index.py       tokeniser, text synthesis, lazy index loader
app/services/authority_graph.py  lazy PageRank-score loader
app/services/qdrant.py           + query_with_embedding(), fetch_vectors() (additive)
app/services/council.py          two retrieval calls swapped to retrieval_service
app/config.py                    retrieval flags (safe defaults)
scripts/build_bm25_index.py      offline BM25 builder
scripts/build_authority_graph.py offline graph builder (PageRank/HITS/Louvain)
scripts/eval_retrieval.py        IR benchmark (pooled relevance)
eval_data/retrieval_eval.json    20 labelled queries with relevance refs
```
```
output/retrieval_metrics.md      IR results table (the jury number)
data/graph_stats.json            graph stats (top statutes, communities)
```
