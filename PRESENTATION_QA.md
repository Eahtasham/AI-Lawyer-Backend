# Project Defence — 50 Questions & Answers (Surface → Deep)

Prepared for the jury viva. Ordered from surface-level to deep technical. Read the
**bold answer first** (your spoken reply); the italics/extra lines are backup detail if
they press. Pair this with `UNDERSTANDING.md` (which has the worked examples).

Sections:
A. Project & RAG basics (Q1–8) · B. Embeddings, vectors & HNSW (Q9–18) ·
C. Smart Chunker (Q19–26) · D. BM25 & lexical retrieval (Q27–33) ·
E. Hybrid fusion: RRF & MMR (Q34–40) · F. Evaluation / IR metrics (Q41–45) ·
G. Authority graph: PageRank / HITS / Louvain (Q46–50).

---

## A. Project & RAG basics

**Q1. In one line, what is your project?**
An AI legal assistant for Indian law that answers questions using **RAG** — it retrieves
the most relevant statute/case passages from a 96,000-chunk legal database and a
multi-agent "council" of LLMs reasons over them to produce a grounded answer.

**Q2. What is RAG and why use it instead of just asking an LLM?**
**Retrieval-Augmented Generation = retrieve relevant documents, then let the LLM answer
using them as context.** It reduces hallucination (answers are grounded in real legal
text), lets us cite sources, and keeps knowledge current without retraining. A bare LLM
would invent plausible-sounding but wrong sections.

**Q3. Walk us through the full pipeline for one query.**
Query → **Clerk** (classifies legal/non-legal, rewrites the query, routes to statutes
and/or cases) → **Retrieval** (hybrid: dense + BM25 over Qdrant) → **Council** (4
specialist agents each give an opinion) → **Chairman** (synthesises a final answer) →
streamed to the frontend with the retrieved sources.

**Q4. What are the major components and the tech stack?**
FastAPI backend, Next.js frontend, **Qdrant** vector DB (two collections: ~53k statute
chunks, ~43k Supreme Court case chunks), **Gemini** for embeddings and generation,
Supabase for chat storage.

**Q5. Why a multi-agent "council" instead of a single model?**
Different legal lenses matter: a **Constitutional Expert**, a **Statutory Analyst**, a
**Case-Law Researcher**, and a **Devil's Advocate** each analyse the same context, then a
**Chairman** reconciles them. This surfaces counter-arguments and reduces one-shot bias.

**Q6. What is the "Clerk" and why is it needed?**
A lightweight router. It does **contextual query rewriting** (resolves "it"/"that case"
from history), classifies legal vs. non-legal (so chit-chat skips the expensive council),
and decides whether to search statutes, cases, or both. It saves cost and improves
retrieval targeting.

**Q7. What did *you* build vs. what comes from libraries?**
We built the retrieval pipeline (dense+lexical fusion, reranking, authority re-ranking),
the chunking strategy, the multi-agent orchestration, the graph construction and
analysis, and the full evaluation harness. We use **reference implementations** for inner
kernels — BM25 scoring (`rank_bm25`), PageRank's linear algebra (`networkx`/`scipy`),
HNSW (Qdrant) — which is standard engineering practice. *We own the design, integration,
indexing, and evaluation.*

**Q8. What's the single biggest technical contribution?**
Turning retrieval from a black-box vector call into a measured **hybrid IR engine**
(lexical + semantic fusion) plus a **legal authority graph** — and proving the
improvement with standard IR metrics (MAP rose 0.57 → 0.90).

---

## B. Embeddings, vector search & HNSW

**Q9. What is an embedding?**
A vector of numbers (ours have **2048 dimensions**) representing the *meaning* of text.
Similar meanings → nearby vectors, so "stealing" lands near "theft" even with no shared
words. Produced by Gemini's `gemini-embedding-001`.

**Q10. Why 2048 dimensions? Isn't that a lot?**
It's the dimensionality the embedding model outputs for high-fidelity retrieval. More
dimensions capture finer semantic distinctions but cost memory and compute; 2048 is the
model's configured output size, a balance the provider chose for retrieval-quality tasks.

**Q11. How do you measure similarity between vectors?**
**Cosine similarity** — the cosine of the angle between them: `A·B / (|A||B|)`. It ignores
vector magnitude and compares direction (meaning). Range −1…1; for our embeddings ~0…0.8.

**Q12. Why cosine and not Euclidean distance?**
Cosine is scale-invariant — two texts about the same topic are "close" regardless of
length/magnitude. For normalized embeddings cosine and Euclidean rank identically, but
cosine is the convention for semantic similarity.

**Q13. How does the vector database find nearest neighbours so fast?**
Qdrant uses **HNSW** — Hierarchical Navigable Small World graphs — an *approximate*
nearest-neighbour index. Instead of comparing the query to all 96k vectors, it navigates
a graph in roughly **logarithmic** time.

**Q14. Explain HNSW in depth.**
It's a **multi-layer graph of vectors**. The bottom layer contains every point densely
connected to its nearest neighbours; higher layers are sparse "express lanes" with
long-range links. Search starts at an entry point in the **top** layer and **greedily**
hops to the neighbour closest to the query; when it can't get closer, it **drops a layer**
and repeats, until the bottom layer yields the nearest neighbours. The sparse top layers
let you cover huge distances quickly (like skip-lists for graphs); the dense bottom layer
gives precision. Search is ~**O(log N)** vs **O(N)** brute force.

**Q15. What are HNSW's key parameters and their trade-offs?**
- **M** — max neighbours per node. Higher M = better recall, more memory.
- **ef_construction** — candidate-list size while *building*; higher = better graph, slower build.
- **ef_search** — candidate-list size at *query time*; higher = better recall, slower query.
*ef_search is the live recall↔latency dial.*

**Q16. Why "approximate"? Don't you want the exact nearest neighbours?**
Exact search is `O(N·d)` per query (here ~96k × 2048 ≈ 200M multiply-adds) and scales
linearly — fine now, painful at millions of vectors. HNSW trades a tiny recall loss
(typically >95% recall) for orders-of-magnitude speed-up. For RAG, near-perfect recall is
plenty because the LLM tolerates an occasional sub-optimal passage.

**Q17. Did you implement HNSW yourself?**
No — Qdrant provides it, and re-implementing a production ANN index would be reinventing
the wheel. Our owned algorithmic work is the **lexical index (BM25), the fusion (RRF),
reranking, and the graph algorithms**, where we add measurable value on top of the vector
store.

**Q18. How is the index stored / how big is it?**
Vectors live in Qdrant (two collections, ~96k points total, 2048-dim each). Our additional
**BM25 lexical indexes** are ~590 MB on disk (built offline), and the **authority scores**
are a 1.2 MB lookup table.

---

## C. The Smart Chunker (and why cases differ)

**Q19. Do you chunk statutes and cases the same way?**
No — and the asymmetry is deliberate. **Statutes ARE chunked** by section (Q20). **Cases are
NOT text-chunked** — each judgment is summarised by an LLM into a structured headnote and
embedded as a *single* vector (Q25). One act → many vectors; one case → one vector.

**Q20. How does the statute Smart Chunker work?**
It measures every section in **tokens** with TikToken and uses an **aggressive-merge +
split** strategy with a **600-token budget**:
1. **Merge** — walk the act's sections in order and keep combining consecutive sections into
   one chunk until adding the next would exceed 600 tokens. This avoids tiny one-line-section
   chunks (which embed poorly). *(That's why a chunk's `section_numbers` is a list — it can
   hold several merged sections.)*
2. **Split** — if a *single* section is itself larger than 600 tokens, split it into
   ≤600-token pieces **at sentence boundaries** (regex on `.!?`), never mid-sentence.
Each chunk = one-or-several small sections (or a slice of a big one), ≤600 tokens, tagged
with all its section numbers/titles, then embedded.

**Q21. Why count tokens instead of characters or words?**
Because the embedding/LLM models think in **tokens** (sub-word units via Byte-Pair
Encoding). Token counts map directly to model context limits and embedding quality;
character counts don't (a 1000-char clause and a 1000-char list tokenize very differently).

**Q22. Why a 600-token budget, and why *merge* small sections?**
The token budget aligns chunks with the embedding model. **Merging** solves a real problem in
bare acts: many sections are a single line ("Short title", a one-clause definition). Embedding
each tiny fragment alone gives weak, ambiguous vectors; merging neighbours into a ~600-token
chunk gives a richer, more retrievable unit while staying precise. *(We profiled 400/600/800
in `analyze_chunks.py`; 600 was the balance.)*

**Q23. Why split at sentence boundaries instead of fixed-size windows?**
Cutting mid-sentence destroys meaning — especially in law, where a clause's force depends on
its full wording. Sentence-boundary splitting keeps each chunk grammatically and legally
coherent, so its embedding is meaningful and the LLM doesn't read half a rule.

**Q24. Do chunks overlap?**
The statute chunker does **not** add sliding-window overlap. It relies on **section integrity**
(whole sections stay together via merging) and **sentence-boundary splits** so no chunk ends
mid-thought. Overlap (repeating text across boundaries) is the alternative technique — we'd
consider it as an improvement, trading storage for extra boundary safety.

**Q25. How are *cases* handled if they're not chunked?**
Each judgment PDF is OCR'd, then **Gemini produces a structured summary** —
`executive_summary`, `facts`, `issues`, arguments, `judgment` (held), `laws_applied`,
`legal_principles`, `keywords`, `outcome`. We concatenate the key fields into one text
("Summary… Facts… Judgment Held…") and create **one embedding per case**. *(Large PDFs are
chunked only to fit Gemini's input window during extraction — a processing step, not the
stored unit.)*

**Q26. Why summarise cases instead of chunking them like statutes?**
A statute is **modular** — sections are self-contained units you cite directly, so chunking by
section is natural. A judgment is a **narrative**; chunking it produces fragments that lose the
ratio/holding. An LLM headnote captures facts + issue + ratio + outcome in one coherent,
retrievable unit. Trade-off: retrieval depends on summary quality, and we don't keep the
verbatim full text — which is why a case's raw `text` field is empty and content lives in the
`summary` (and why the hybrid BM25 indexer synthesises case text from those summary fields).

---

## D. BM25 & lexical retrieval

**Q27. What problem does BM25 solve that vector search doesn't?**
Vector search matches **meaning** but can miss **exact tokens** — section numbers
("Section 302"), case names ("Puttaswamy"), rare legal terms. BM25 is a **lexical** scorer
that nails exact-term matching. They're complementary.

**Q28. What is BM25, precisely?**
"Best Match 25" — a ranking function scoring how well a document's *words* match a query,
over an **inverted index**. It combines three ideas: **term frequency** (with saturation),
**inverse document frequency** (rare words weigh more), and **document-length
normalization**.

**Q29. Give the formula and explain each term.**
`BM25(D,Q) = Σ_t IDF(t) · [ f(t,D)·(k1+1) ] / [ f(t,D) + k1·(1 − b + b·|D|/avgdl) ]`
- `f(t,D)` term frequency in the doc; `|D|` doc length; `avgdl` average length.
- `k1` (~1.5) controls **TF saturation** — the 10th occurrence adds less than the 1st.
- `b` (~0.75) controls **length normalization** — long docs don't unfairly win.
- `IDF(t) = ln((N − n_t + 0.5)/(n_t + 0.5) + 1)` — rare terms (small `n_t`) get high weight.

**Q30. How is BM25 different from plain TF-IDF?**
TF-IDF grows linearly with term frequency and doesn't normalize length well. BM25 adds
**TF saturation** (`k1`) and **principled length normalization** (`b`), which is why it
consistently outperforms TF-IDF and became the IR standard.

**Q31. What data structure powers it, and how did you build the index?**
An **inverted index** (term → postings list of documents + frequencies). We built it
**offline** over all 96k chunks (`scripts/build_bm25_index.py`), tokenizing each chunk
(for cases we synthesise text from the structured summary since the raw `text` field is
empty) and fitting a `BM25Okapi` scorer, then persisted it.

**Q32. BM25 is keyword-based — doesn't it fail on synonyms?**
Yes — "stealing" won't lexically match "theft". That's exactly why we **fuse** it with
dense (semantic) retrieval, which handles synonyms. Neither alone is enough; together they
cover both axes.

**Q33. What's BM25's complexity / cost at query time?**
It scores the query against candidate documents using the inverted index; in our
implementation it evaluates the corpus per query (~2–5 s over 53k/43k docs). For
production scale you'd use a posting-list engine (SQLite FTS5, Tantivy, or Qdrant sparse
vectors) that only touches documents containing the query terms.

---

## E. Hybrid fusion: RRF & MMR

**Q34. You have two rankings (dense, BM25) with different score scales. How do you combine them?**
With **Reciprocal Rank Fusion (RRF)** — it ignores raw scores and uses only **rank
positions**, so incompatible scales (cosine ≈ 0.7 vs BM25 ≈ 5) don't matter.

**Q35. Give the RRF formula and the intuition.**
`RRF(d) = Σ_lists 1/(k + rank_list(d))`, with `k = 60`. A document ranked #1 in either list
scores high; a document ranked highly by **both** gets two contributions and rises to the
top — the "agreement bonus." *Example: ranked 2nd and 4th → 1/62 + 1/64 = 0.0318, beating a
sole #1 at 1/61 = 0.0164.*

**Q36. Why the constant k = 60?**
It softens the dominance of the very top ranks so lower-but-consistent results still
matter. 60 is the value from the original RRF paper and is the community default; it's
robust and we kept it.

**Q37. What is MMR and why did you build it?**
**Maximal Marginal Relevance** reranks results to balance relevance against **redundancy**:
`argmax_d [ λ·sim(d,q) − (1−λ)·max_{s∈S} sim(d,s) ]`. It greedily picks results that are
relevant but *different* from what's already chosen, to diversify the top-k.

**Q38. You said MMR is off by default — why keep a feature you disabled?**
Because we **measured** it. For legal queries the relevant passages are near-identical
provisions you *want* clustered at the top; MMR's diversity penalty scattered them and
dropped **MAP from 0.90 to 0.47**. So we benchmarked it and disabled it — an evidence-based
decision, which is itself a result worth presenting.

**Q39. After fusion, how do you produce the final score shown to the user?**
We take the top RRF pool, then score each by **cosine relevance to the query** for display
(kept in [0,1] so the UI's "match %" is meaningful), and attach provenance
(`dense_rank`, `bm25_rank`, `rrf_score`, `authority`). For cases we optionally blend the
authority score (next section).

**Q40. How do you know the hybrid is actually better — could it be luck?**
We ran a controlled IR benchmark (next section). Hybrid beat dense on every ranking metric
across 20 labelled queries; the largest gain (MAP 0.57→0.90) is far outside noise. It's
reproducible: `python -m scripts.eval_retrieval`.

---

## F. Evaluation / IR metrics

**Q41. How did you evaluate retrieval without paying humans to label everything?**
**TREC-style pooling**: pool the candidate documents from all methods, judge each against
the query's reference labels (correct act/section/case), and treat that as the relevance
set. It's the standard way to evaluate retrieval with limited labelling.

**Q42. Define the metrics you report.**
- **Precision@k** = relevant in top-k / k. **Recall@k** = relevant in top-k / total relevant.
- **MRR** = mean of 1/(rank of first relevant) — how high the first hit is.
- **nDCG@k** = DCG/IDCG; rewards relevant docs ranked high, with a log position discount;
  normalized to [0,1].
- **MAP** = mean Average Precision; strictest — rewards ranking *all* relevant docs high.

**Q43. Walk us through nDCG with a tiny example.**
Top-5 relevance pattern `[1,0,1,1,0]`: `DCG = 1/log2(2)+1/log2(4)+1/log2(5) = 1.931`. Ideal
order `[1,1,1,0,0]`: `IDCG = 2.131`. `nDCG = 1.931/2.131 = 0.906` → 90.6% of perfect.

**Q44. What were the headline numbers?**
nDCG@10: dense 0.916 → **hybrid 0.961**. MAP: dense 0.573 → **hybrid 0.904**. BM25 alone was
strong on exact-match queries (nDCG@5 0.978). MMR variant was worse (MAP 0.470).

**Q45. What are the limitations of your evaluation?**
20 queries give *directional*, not statistically tight, numbers; relevance labels are
heuristic (substring match), not adjudicated by lawyers; and generation quality is judged
by an LLM, which has its own biases (citation accuracy, though, is deterministic). We'd
expand the labelled set and add human adjudication for a publication-grade result.

---

## G. Authority graph: PageRank / HITS / Louvain

**Q46. Why build a graph, and what are its nodes and edges?**
To rank **authoritative** precedent. The corpus has **no stored case-to-case citations**,
so we build a **bipartite graph**: nodes are **cases** (~38.8k) and **statutes** (~17.8k);
an edge connects a case to each statute it applies (from the case summary's
`laws_applied`). ~56.6k nodes, ~93.5k edges.

**Q47. Explain PageRank and how you use it.**
PageRank scores node importance via the **random-surfer** model: a node is important if
important nodes connect to it, and importance flows across edges.
`PR(u) = (1−d)/N + d·Σ_{v→u} PR(v)/deg(v)`, damping `d=0.85`, solved by **power iteration**
(start uniform, iterate to convergence). We run it on the bipartite graph; a **case's**
PageRank is its **authority score**, used to promote landmark cases in re-ranking.

**Q48. What's HITS and what did it show?**
HITS gives each node a **hub** and an **authority** score that reinforce each other
(`authority = Σ hubs pointing in`, `hub = Σ authorities pointed to`). On our graph,
**statutes** accumulate authority. The top results were the **Constitution of India →
IPC → CrPC → CPC → Evidence Act** — the genuinely foundational laws, which validates the
method (face validity).

**Q49. Why bipartite instead of connecting cases that share a statute directly?**
A "projection" connecting all cases that share a statute creates `m(m−1)/2` edges per
statute. The Constitution is applied by 9,728 cases → ~**47 million edges from one node** →
`O(n²)` blow-up. Running PageRank/HITS **directly on the bipartite graph** (93.5k edges)
gives the same authority signal without the explosion. *If you did project, you'd
IDF-weight shared-statute links so ubiquitous statutes count less.*

**Q50. What is Louvain / community detection and what did you find?**
Louvain partitions the graph to **maximize modularity** (far more edges inside groups than
random chance), greedily and fast. On our legal graph it found 1,522 communities whose
largest ones are coherent **legal domains we never labelled** — criminal (IPC/CrPC/
Evidence), civil & property (CPC/Transfer of Property/Contract), constitutional, tax,
labour, land acquisition. Unsupervised community detection effectively **rediscovered the
structure of Indian law**.

---

## Bonus — likely curveballs

**B1. Isn't this just gluing APIs together?**
The orchestration is engineering, yes — but the retrieval engine (BM25 + RRF + reranking),
the graph algorithms (PageRank/HITS/Louvain), and the IR evaluation are owned algorithmic
work with measured results. We can derive every formula and defend every design choice.

**B2. How do you prevent hallucination?**
Grounding (answers must use retrieved context), a strict "answer from context" prompt, the
Devil's Advocate agent, and source display so users can verify. We also measure faithfulness.

**B3. Why Gemini? Could you swap models?**
Large context window, strong reasoning, cost-effective, and good embeddings. The model is
behind a config flag — swapping providers is a config change, not a rewrite.

**B4. Biggest weakness / what would you do with more time?**
No real citation graph (we proxy via shared statutes); BM25 is in-memory (would move to a
disk index for cloud scale); small eval set; add a learned cross-encoder reranker and
topic-sensitive (personalized) PageRank.

**B5. How does it scale to millions of documents?**
Vector side scales via HNSW (log-time). Lexical side moves to a posting-list engine
(SQLite FTS5 / Tantivy / Qdrant sparse). The graph is precomputed offline; authority is a
cheap lookup at query time.
