# Understanding the Two Core-CS Features — A Complete Guide

This document teaches you **every algorithm** in the two features we added, from scratch.
You know general CS (what a graph is, what indexing means) but not these specific
algorithms. By the end you'll be able to explain each one to the jury with confidence.

Read it top to bottom once. Each algorithm follows the same pattern:
**(1) the problem it solves → (2) the intuition → (3) the formula explained symbol by
symbol → (4) a tiny worked example → (5) where it lives in our code → (6) how to say it
out loud.**

---

# PART 0 — The Big Picture

## What the system does
Our app answers Indian-law questions. It uses **RAG (Retrieval-Augmented Generation)**:

1. **Retrieve**: find the most relevant law/case passages from a database of ~96,000
   legal text chunks.
2. **Augment**: paste those passages into the AI's prompt as context.
3. **Generate**: the AI (Gemini) answers using that context.

The quality of the final answer depends massively on step 1. *Garbage in, garbage out.*
So step 1 — **retrieval** — is where we added real algorithms.

## What was there before vs. what we added
- **Before:** retrieval was 100% "dense vector search" done by a library (Qdrant). We
  just called it. That's engineering, not core CS.
- **Feature 1 (Hybrid Retrieval):** we *built our own* retrieval algorithms (BM25,
  Reciprocal Rank Fusion, MMR) and **measured** them with standard Information Retrieval
  metrics. We now own the retrieval logic and can prove it's better.
- **Feature 2 (Legal Authority Graph):** we built a graph of the legal corpus and ran
  classic graph algorithms (PageRank, HITS, Louvain) to find the most *authoritative*
  cases and statutes, and to discover the structure of Indian law.

Two CS fields: **Information Retrieval** and **Graph Algorithms**.

---

# PART 1 — Foundations You Need First

Before the new algorithms, three building blocks.

## 1.1 Embeddings (vectors that capture meaning)
An **embedding** is a list of numbers (a vector) that represents the *meaning* of a piece
of text. Our embeddings have **2048 numbers** each (2048 "dimensions"), produced by a
Gemini model.

The key property: **texts with similar meaning have similar vectors**, even if they use
different words. "stealing" and "theft" land close together; "theft" and "marriage" land
far apart. This is *semantic* matching (by meaning).

Think of it as a point in 2048-dimensional space. We can't picture 2048D, but the math
works the same as 2D/3D.

## 1.2 Cosine similarity (how close two vectors are)
To measure how similar two vectors A and B are, we use the **cosine of the angle**
between them:

```
cosine(A, B) =  (A · B) / (|A| × |B|)

  A · B   = dot product = a1*b1 + a2*b2 + ... + a2048*b2048
  |A|     = length of A = sqrt(a1^2 + a2^2 + ... )
```

- Result ranges from −1 (opposite) to +1 (identical direction).
- For our embeddings it's usually between ~0 and ~0.8.
- **Why cosine and not plain distance?** It ignores vector *length* and only cares about
  *direction* (meaning), which is what we want.

**Tiny worked example (2D):** A = [1, 1], B = [2, 0].
```
A · B = 1*2 + 1*0 = 2
|A| = sqrt(2) ≈ 1.41,  |B| = 2
cosine = 2 / (1.41 × 2) = 0.71      → fairly similar (45° apart)
```

In our code this is `_cosine(a, b)` in `app/services/hybrid.py`.

## 1.3 Dense (vector) retrieval — what Qdrant already did
"Dense retrieval" = **embed the query into a vector, then find the database chunks whose
vectors are most cosine-similar.** Those are the semantically closest passages.

Searching 96,000 vectors exactly would be slow, so Qdrant uses an *approximate* method
called **HNSW** (a navigable graph of vectors). We did **not** build HNSW — Qdrant does
it. Just know: "dense search = nearest vectors by cosine, made fast by HNSW." That's the
*semantic* retriever.

## 1.4 Inverted index — the classic *lexical* index
You know indexing generally. The IR-specific structure is an **inverted index**: a map
from each **word → the list of documents that contain it** (plus how often).

```
"theft"   -> [doc5 (3 times), doc12 (1), doc88 (2), ...]
"murder"  -> [doc2 (1), doc7 (4), ...]
```

This lets you instantly find documents that literally contain the query words. This is
*lexical* matching (by exact words), the opposite philosophy to embeddings. **BM25**
(next) is the scoring formula that runs on top of an inverted index.

---

# PART 2 — FEATURE 1: The Hybrid Retrieval Engine

The core idea: **two retrievers see different things, so combine them.**
- **Dense (vectors):** great at *meaning* ("stealing" finds "theft") but can miss exact
  terms, section numbers, rare names.
- **BM25 (words):** great at *exact terms* ("Section 302", "Puttaswamy") but blind to
  synonyms ("stealing" won't match "theft").

We run both, **fuse** their rankings (RRF), and optionally **rerank** for diversity (MMR).

## 2.1 BM25 — lexical relevance scoring

### The problem
Given a query and a document, how well do the *words* match? Naively you'd count shared
words, but that's too crude. BM25 (Best Match 25) is the industry-standard formula that
fixes three problems with naive counting.

### The three ideas inside BM25
1. **Term Frequency (TF):** a word appearing more often in a document means that document
   is more about it — *but with diminishing returns*. The 10th occurrence of "theft"
   matters less than the 1st. BM25 **saturates** TF.
2. **Inverse Document Frequency (IDF):** rare words are more informative. "theft" is a
   strong signal; "the" is useless. IDF gives rare words higher weight.
3. **Length normalization:** long documents naturally contain more words, so they'd
   unfairly win. BM25 penalizes length.

### The formula
For a document `D` and query `Q` (a set of terms `t`):

```
                    f(t,D) · (k1 + 1)
BM25(D,Q) = Σ  IDF(t) · ─────────────────────────────────────
            t∈Q         f(t,D) + k1 · (1 − b + b · |D|/avgdl)
```

Symbol by symbol:
- `f(t,D)` = how many times term `t` appears in document `D`  (term frequency).
- `|D|`    = length of document `D` (number of words).
- `avgdl`  = average document length across the whole corpus.
- `k1`     = TF saturation knob (we use ~1.5). Higher = TF keeps mattering longer.
- `b`      = length-normalization knob (we use ~0.75). 0 = ignore length, 1 = full.
- `IDF(t)` = `ln( (N − n_t + 0.5) / (n_t + 0.5) + 1 )`
   - `N`   = total number of documents in the corpus.
   - `n_t` = number of documents that contain term `t`.
   - So if `t` is in *few* documents, `n_t` is small, IDF is large → rare word, big weight.

You **sum** this over every query term. Higher total = better lexical match.

### Why each knob exists (the intuition again)
- The `(k1+1)·f / (f + k1·...)` shape rises fast for the first few occurrences then
  flattens → **saturation**.
- `b · |D|/avgdl` makes the denominator bigger for long docs → **length penalty**.
- `IDF` multiplies each term → **rare terms dominate**.

### Tiny worked example
Query = "theft". Corpus has N = 1000 docs. "theft" appears in n_t = 50 docs.
```
IDF("theft") = ln((1000 − 50 + 0.5)/(50 + 0.5) + 1) = ln(19.8) ≈ 2.99
```
Document D mentions "theft" 3 times, |D| = 100 words, avgdl = 120, k1 = 1.5, b = 0.75:
```
numerator   = 3 · (1.5 + 1) = 7.5
denominator = 3 + 1.5·(1 − 0.75 + 0.75·100/120) = 3 + 1.5·(0.875) = 4.31
BM25 = 2.99 · (7.5 / 4.31) ≈ 5.20
```
A document that says "theft" once would score lower; one that says it 20 times barely
beats 3 times (saturation); a 1000-word document scores lower (length penalty).

### In our code
- We build the inverted index + BM25 statistics offline over all 96k chunks:
  `scripts/build_bm25_index.py` (saves `data/bm25_*.pkl`, ~590 MB total).
- At query time, `BM25Store.top_n()` in `app/services/bm25_index.py` returns the
  top lexical matches. We use the `rank_bm25` library's `BM25Okapi` for the core formula
  (this is the standard reference implementation; we own the indexing, tokenisation, and
  how it's used).

### Say it out loud
> "BM25 is the classic lexical relevance score. It rewards query words that appear often
> in a document (with diminishing returns), weights rare words more heavily via IDF, and
> normalizes for document length. It runs on an inverted index we built over all 96,000
> legal chunks."

## 2.2 Reciprocal Rank Fusion (RRF) — combining two ranked lists

### The problem
Dense gives a ranking; BM25 gives a ranking. Their **scores are on totally different
scales** (cosine ≈ 0.7 vs BM25 ≈ 5.2) — you can't just add them. How do you merge two
rankings fairly?

### The idea
**Ignore the raw scores; use only the rank positions.** A document ranked #1 by either
method is trusted; being ranked highly by *both* is best of all.

### The formula
```
            ⎛        1              1         ⎞
RRF(d) = Σ  ⎜ ─────────────  or  ─────────────  ⎟   over each list the doc appears in
            ⎝  k + rank_dense     k + rank_bm25 ⎠

  rank = the document's position in that list (1 = top)
  k    = a constant, we use 60 (standard; softens the difference between top ranks)
```

Add the contributions from each list. Sort everything by total RRF score, descending.

### Why it works
- Rank-based → immune to incompatible score scales.
- `1/(k+rank)` decays gently, so a doc that's #1 in one list and #15 in the other still
  gets a healthy combined score (it's not all-or-nothing).
- A doc found by **both** retrievers gets two contributions → naturally rises to the top.
  That's the "agreement bonus."

### Tiny worked example
k = 60. Document X is ranked **2nd by dense** and **4th by BM25**:
```
RRF(X) = 1/(60+2) + 1/(60+4) = 1/62 + 1/64 = 0.01613 + 0.01563 = 0.03176
```
Document Y is ranked **1st by dense** but **not found by BM25**:
```
RRF(Y) = 1/(60+1) = 0.01639
```
So X (found by both) beats Y (found by one), even though Y was someone's #1. **This is
exactly what we saw in our results** — the chunk found by both dense *and* BM25 rose to
the top.

### In our code
`hybrid_search()` in `app/services/hybrid.py`, the "Reciprocal Rank Fusion over the
union" block.

### Say it out loud
> "RRF merges the two rankings using only positions, not raw scores, so the
> incompatible scales don't matter. Documents that both retrievers rank highly get
> rewarded twice and rise to the top."

## 2.3 Maximal Marginal Relevance (MMR) — diversity reranking

### The problem
The top results might all be **near-duplicates** (e.g. five copies of the same section).
The user might prefer a *diverse* set covering different angles. MMR re-orders results to
balance **relevance** against **redundancy**.

### The idea
Build the result list one item at a time. At each step pick the document that is
**relevant to the query** but **different from what you've already picked**.

### The formula
Pick the next document `d` (from remaining set R, given already-selected set S) that
maximizes:
```
MMR(d) = λ · sim(d, query)  −  (1 − λ) · max sim(d, s)
                                         s∈S
  sim(d, query) = relevance (cosine of doc to the query)
  max sim(d, s) = how similar d is to the MOST similar already-picked doc (redundancy)
  λ (lambda)    = the knob. λ=1 → pure relevance (ignore diversity).
                            λ=0 → pure diversity (ignore relevance).
```
First pick = the most relevant doc (S is empty). Then each subsequent pick is penalized
for looking like something already chosen.

### Tiny worked example
λ = 0.7. We already picked doc A. Candidate B: relevance 0.80, similarity-to-A 0.90.
Candidate C: relevance 0.70, similarity-to-A 0.10.
```
MMR(B) = 0.7·0.80 − 0.3·0.90 = 0.56 − 0.27 = 0.29
MMR(C) = 0.7·0.70 − 0.3·0.10 = 0.49 − 0.03 = 0.46   ← C wins
```
C is slightly less relevant but much more *novel*, so MMR picks it next.

### Why we measured it and then TURNED IT OFF (important — a strength, not a weakness)
In law, the relevant passages are often near-identical provisions, and you *want* them
clustered at the top. MMR's diversity penalty **pushed relevant duplicates down**, which
**hurt** our ranking metrics (MAP fell from 0.90 to 0.47 — see Part 3). So we
**implemented it, benchmarked it, and the data told us to leave it off** for this corpus.
That is exactly the kind of evidence-based engineering decision a jury wants to see.

### In our code
`_mmr_order()` in `app/services/hybrid.py`; toggled by the `USE_MMR` flag (default
`False`).

### Say it out loud
> "MMR reranks for diversity — it picks results that are relevant but not redundant.
> We benchmarked it, but for legal text the relevant provisions are similar to each other,
> so diversity hurt ranking quality. We measured that and left it off — RRF won."

## 2.4 The full hybrid pipeline (how the pieces connect)
For one query (per collection):
```
1. Embed the query once (one vector).
2. DENSE: ask Qdrant for the top 30 by cosine (semantic).
3. BM25:  ask our index for the top 30 by lexical score.
4. RRF:   fuse the two rankings into one combined ranking.
5. Take the top ~20 (the "pool").
6. (optional) MMR rerank for diversity — off by default.
7. (optional) Authority blend — promote landmark cases (Feature 2).
8. Return the top 5 chunks, each with a score in [0,1] and a provenance record
   {dense_rank, bm25_rank, rrf_score, authority}.
```

---

# PART 3 — Measuring Retrieval (the IR metrics behind the table)

You must be able to explain the results table. To grade a ranking you need **(a) which
documents are truly relevant** and **(b) metrics that score the ranking**.

## 3.1 Relevance judgments + pooling (how we got "ground truth")
For each test query we wrote down **reference labels** (e.g. for "theft": the words
"theft", "section 378", "section 379"). A retrieved chunk is judged **relevant** if its
text/metadata contains a label.

**Pooling (the TREC method):** we gather the candidate documents from *all* methods into
one pool, judge each one, and use that as the relevance set. This is the standard way to
evaluate retrieval without paying humans to read all 96k documents. (`scripts/eval_retrieval.py`.)

Below, a ranked list is written as `1 = relevant`, `0 = not`, e.g. `[1,0,1,1,0]`.

## 3.2 Precision@k and Recall@k
```
Precision@k = (relevant items in top k) / k          "of what I showed, how much was good?"
Recall@k    = (relevant items in top k) / (total relevant)   "of all the good, how much did I find?"
```
Example, top-5 = `[1,0,1,1,0]`, and there are 8 relevant docs total:
```
Precision@5 = 3/5 = 0.60
Recall@5    = 3/8 = 0.375
```

## 3.3 MRR — Mean Reciprocal Rank
"How high was the **first** correct answer?" For one query, RR = 1/(rank of first
relevant). MRR = average RR over all queries.
```
First relevant at position 1 → RR = 1.00
First relevant at position 3 → RR = 0.33
```
Great metric for "did we get *a* right answer near the top." (Ours = 1.000 → every query
had a relevant doc at rank 1.)

## 3.4 nDCG — normalized Discounted Cumulative Gain (the main ranking metric)
Rewards putting relevant docs **high**, with a logarithmic discount for lower positions.

```
DCG@k  = Σ (rel_i / log2(i + 1))   for i = 1..k     (rel_i = 1 if relevant else 0)
IDCG@k = the DCG of the BEST possible ordering (all relevant docs first)
nDCG@k = DCG@k / IDCG@k             → a number in [0,1], 1 = perfect ordering
```
Worked example, top-5 = `[1,0,1,1,0]`:
```
DCG  = 1/log2(2) + 0/log2(3) + 1/log2(4) + 1/log2(5) + 0/log2(6)
     = 1/1 + 0 + 1/2 + 1/2.32 + 0 = 1 + 0.5 + 0.431 = 1.931
Ideal ordering would be [1,1,1,0,0]:
IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1 + 0.631 + 0.5 = 2.131
nDCG = 1.931 / 2.131 = 0.906
```
So this ranking is 90.6% of perfect. **This is the headline metric** because it cares
about *order*, not just presence.

## 3.5 MAP — Mean Average Precision
The strictest single number. **Average Precision (AP)** for one query = average of
"Precision@k" computed *at each position where a relevant doc appears*, divided by total
relevant. MAP = mean AP across queries.

Worked example, ranking = `[1,0,1,1,0]`, total relevant = 3:
```
Relevant docs are at positions 1, 3, 4.
Precision at pos 1 = 1/1 = 1.00
Precision at pos 3 = 2/3 = 0.67
Precision at pos 4 = 3/4 = 0.75
AP = (1.00 + 0.67 + 0.75) / 3 = 0.806
```
MAP rewards getting **all** the relevant docs ranked high and contiguous. This is why
**MMR tanked MAP** — by scattering relevant near-duplicates apart, the precision at later
relevant positions dropped.

## 3.6 Reading OUR results table
```
| Method                | nDCG@10 | MAP   |
| Dense (old system)    | 0.916   | 0.573 |
| BM25 (lexical only)   | 0.946   | 0.668 |
| Hybrid (RRF) — NEW    | 0.961   | 0.904 |   ← best
| Hybrid + MMR          | 0.911   | 0.470 |   ← MMR hurt
```
What to say:
- **"Hybrid RRF beats the old dense-only system on every metric."**
- **"The biggest jump is MAP: 0.57 → 0.90. MAP measures whether we rank *all* relevant
  passages highly — fusing lexical + semantic surfaces relevant docs that either method
  alone missed."**
- **"We also tested MMR; it reduced quality for legal text, so we left it off. That's an
  evidence-based decision, not a guess."**

---

# PART 4 — FEATURE 2: The Legal Authority Graph

## 4.1 Why a graph, and the data problem
Lawyers care about **authoritative** precedent — landmark cases everyone cites. Normally
you'd build a **citation graph** (case A cites case B) and rank by citations. **But our
data has no stored case-to-case citations** (the field was empty for every case). What we
*do* have, for ~91% of cases, is `laws_applied` — which statutes each case used.

So we build a different graph and derive authority from it.

## 4.2 Bipartite graph (the structure we built)
A **bipartite graph** has **two types of nodes**, and edges only ever connect *across*
types — never within a type.

- Type 1: **cases** (~38,806 of them).
- Type 2: **statutes** (~17,790 distinct acts/codes).
- An **edge** connects a case to each statute it applies.

```
   CASES                         STATUTES
   Puttaswamy ───────────────►  Constitution of India
   Kharak Singh ─────────────►  Constitution of India
   Some Criminal Case ───────►  Indian Penal Code
                      └───────►  Code of Criminal Procedure
```
Result: **56,596 nodes, 93,509 edges.** This graph encodes "which cases lean on which
laws," and that lets us compute authority.

(Before building edges we **normalize statute names** — "Indian Penal Code, 1860", "IPC",
"Indian Penal Code" all collapse to one node — so the graph isn't fragmented. That's the
`canonical_statute()` function.)

## 4.3 PageRank — ranking node importance (Google's algorithm)

### The problem
Which nodes are "important"? Not just "most connected" — importance should **flow** from
important neighbours. A case linked to central statutes should itself rank higher.

### The intuition: the "random surfer"
Imagine someone randomly walking the graph: from a node they follow a random edge, over
and over. Occasionally (probability `1−d`) they teleport to a random node. After a long
time, **the fraction of time spent at each node = that node's PageRank.** Important nodes
get visited more.

### The formula
```
PR(u) = (1 − d)/N  +  d · Σ  PR(v) / degree(v)
                          v∈neighbours(u)

  N        = total number of nodes
  d        = damping factor = 0.85 (chance of following an edge vs teleporting)
  (1−d)/N  = the teleport term (everyone gets a small baseline)
  Σ ...    = importance flowing in from neighbours, split by their degree
```
A node gains rank from its neighbours, and each neighbour shares its own rank **divided
by how many connections it has** (so a neighbour that points everywhere gives each less).

### How it's computed: power iteration
1. Start: every node gets equal rank `1/N`.
2. Apply the formula to update every node's rank (one "iteration").
3. Repeat. The numbers **converge** to stable values (usually within ~50–100 iterations).

### Tiny worked example (intuition)
Three nodes, A↔B, A↔C, B↔C (a triangle): all symmetric → equal PageRank (1/3 each).
Now add a 4th node D connected only to A. A is now the most-connected hub → A's PageRank
rises, because rank flows into it from B, C, and D.

### In our system
- We run PageRank on the **bipartite graph** (`nx.pagerank` in
  `scripts/build_authority_graph.py`).
- A **case's** PageRank = its **authority score** (saved to `data/authority_scores.json`,
  normalized to 0–1). Cases that apply many central statutes score high.
- Used optionally to **re-rank retrieved cases** so landmark precedents surface first.

### Say it out loud
> "PageRank is the random-surfer algorithm Google used: a node is important if important
> nodes connect to it, and importance flows across edges with an 85% damping factor. We
> run it on the case–statute graph to score how authoritative each case is."

## 4.4 HITS — Hubs and Authorities

### The problem / idea
HITS gives every node **two** scores:
- **Authority**: high if **many good hubs point to it** (it's a destination everyone
  references).
- **Hub**: high if **it points to many good authorities** (it's a good index/collector).

They reinforce each other: good hubs point to good authorities, and good authorities are
pointed to by good hubs.

### The formula (mutual reinforcement, iterated)
```
authority(p) = Σ hub(q)      for every q pointing to p
hub(p)       = Σ authority(q) for every q that p points to
   (normalize after each round; repeat until stable)
```

### On our bipartite graph
- **Statutes** accumulate **authority** (pointed to by many cases).
- **Cases** accumulate **hub** scores (point to many authoritative statutes).

### Our actual result (great sanity check)
Top statute authorities came out as:
**Constitution of India → Indian Penal Code → Code of Criminal Procedure → Code of Civil
Procedure → Evidence Act.** The algorithm, with zero hand-tuning, identified the most
foundational laws in the Indian system. That's *face validity* — proof the method works.

### Say it out loud
> "HITS gives each node a hub score and an authority score that reinforce each other.
> On our graph, statutes accumulate authority and cases act as hubs. The top authorities
> came out as the Constitution, IPC, CrPC, CPC, and Evidence Act — exactly the
> foundational statutes, which validates the approach."

## 4.5 Why a bipartite graph instead of a "case-to-case" graph (key design point)
You might ask: "why not connect two cases directly if they share a statute?" That's a
**projection**, and it explodes:

- If a statute is used by `m` cases, connecting all those cases pairwise creates
  `m·(m−1)/2` edges (a *clique*).
- The **Constitution of India** is applied by **9,728 cases** → that one statute alone
  would create about **9,728 × 9,727 / 2 ≈ 47 million edges**.
- Across all hub statutes you'd get hundreds of millions of edges — `O(n²)` blow-up,
  too big to store or compute on.

**Solution:** run the algorithms **directly on the bipartite graph**, which has only
93,509 edges. Same authority signal, no explosion. *(If you ever did want a case-to-case
graph, you'd weight shared-statute links by `IDF` — rare shared statutes count more than
ubiquitous ones — and drop the hub statutes. We mention this as the principled fix.)*

This is a great point to raise yourself in the presentation — it shows algorithmic
maturity (thinking about complexity, not just correctness).

## 4.6 Louvain — community detection (finding clusters)

### The problem
Does the graph have natural **groups** — clusters of cases/statutes that hang together?
In law these would be *legal domains* (criminal, tax, labour...).

### The measure: modularity (Q)
**Modularity** scores a grouping: it's high when there are **many more edges inside
groups than you'd expect by random chance**. We want the grouping that maximizes Q.
Roughly:
```
Q = (fraction of edges that fall inside communities)
      − (expected fraction if edges were placed randomly)
```
High Q = dense inside groups, sparse between groups = good communities.

### The Louvain algorithm (fast, greedy, two repeating phases)
1. **Local phase:** start with every node in its own community. For each node, try moving
   it into a neighbour's community; keep the move if it **increases modularity Q** the
   most. Repeat until no move helps.
2. **Aggregation phase:** collapse each community into a single "super-node" (edges
   become weighted links between communities). 
3. **Repeat** phases 1–2 on the smaller graph until Q stops improving.

It's popular because it's **fast** (handles millions of nodes) and needs no preset number
of clusters.

### Our actual result (the showpiece)
Louvain found **1,522 communities**; the largest ones are coherent legal domains:
- **Criminal law** — IPC, CrPC, Evidence Act
- **Civil & property** — CPC, Transfer of Property Act, Limitation Act, Contract Act
- **Constitutional** — Constitution of India, Contempt of Courts Act
- **Tax** — Income Tax Act, Finance Act, Wealth Tax Act
- **Labour** — Industrial Disputes Act, Factories Act, ESI, Minimum Wages
- **Land acquisition** — Land Acquisition Act and related planning acts

**We did not tell it these categories.** Unsupervised community detection *rediscovered
the structure of Indian law* from co-application patterns. That's a striking result.

### Say it out loud
> "Louvain finds communities by maximizing modularity — groups with far more internal
> connections than random. Run on our legal graph, it rediscovered the branches of
> Indian law on its own: criminal, civil, constitutional, tax, labour, land. We never
> labelled those; the structure emerged from which statutes cases apply together."

---

# PART 5 — How It All Plugs Into the App (and why nothing breaks)

`council.py` now calls a `retrieval_service` instead of calling Qdrant directly. That
service reads three config flags (`app/config.py`):

| Flag | Default | Meaning |
|------|:-------:|---------|
| `RETRIEVAL_MODE` | `"dense"` | `dense` = old behaviour exactly; `hybrid` = our new engine |
| `USE_MMR` | `False` | turn on MMR diversity reranking (benchmarked as worse here) |
| `AUTHORITY_WEIGHT` | `0.0` | `> 0` blends PageRank authority into case ranking |

- With defaults, retrieval is **identical** to before (we verified this with a regression
  test) — so the existing system is safe.
- The heavy indexes/graph are built **offline** by two scripts with **no API cost**, and
  load lazily; if missing, the system falls back to plain dense search.
- For a live demo you set `RETRIEVAL_MODE=hybrid` (and optionally `AUTHORITY_WEIGHT=0.3`).

Final score blending when authority is on (cases):
```
final = (1 − AUTHORITY_WEIGHT) · relevance  +  AUTHORITY_WEIGHT · authority
```
e.g. with weight 0.3, ranking is 70% "matches the query" + 30% "is a landmark case."

---

# PART 6 — Glossary (one-line each)

- **Embedding** — a 2048-number vector representing a text's meaning.
- **Cosine similarity** — angle-based closeness of two vectors (−1..1).
- **Dense retrieval** — find nearest vectors by cosine (semantic search).
- **HNSW** — the fast approximate-nearest-neighbour graph Qdrant uses (we didn't build it).
- **Inverted index** — word → list of documents containing it (lexical search).
- **BM25** — lexical relevance score (TF saturation + IDF + length normalization).
- **TF / IDF** — term frequency / inverse document frequency (rare words weigh more).
- **RRF** — fuse two rankings using positions, not raw scores.
- **MMR** — rerank for relevance *and* diversity (we benchmarked, left off).
- **Precision@k / Recall@k** — fraction of top-k that's relevant / fraction of all relevant found.
- **MRR** — how high the first relevant result is.
- **nDCG** — ranking quality with position discount, normalized to [0,1].
- **MAP** — mean average precision; strictest, rewards ranking all relevant docs high.
- **Pooling** — TREC method: judge the union of all systems' candidates.
- **Bipartite graph** — two node types, edges only across types (cases ↔ statutes).
- **PageRank** — node importance via random-surfer; importance flows across edges.
- **Damping factor (d=0.85)** — chance the surfer follows an edge vs teleports.
- **Power iteration** — repeatedly applying the update until scores converge.
- **HITS** — gives each node a hub score and an authority score (mutually reinforcing).
- **Authority / Hub** — pointed-to-by-many-good-hubs / points-to-many-good-authorities.
- **Louvain** — fast community detection by maximizing modularity.
- **Modularity (Q)** — how much denser edges are inside groups than random chance.

---

# PART 7 — Likely Jury Questions (with answers)

**Q: Why not just use the vector database? Isn't that enough?**
> Vector search is purely semantic — it can miss exact terms like "Section 302" or rare
> case names. Lexical BM25 catches those. Each is blind to the other's strengths, so we
> fuse them. Our metrics show the fusion beats either alone (MAP 0.57 → 0.90).

**Q: Did you implement these algorithms or use libraries?**
> We implemented the retrieval *pipeline*, the fusion (RRF), the MMR reranker, the graph
> construction, and the full evaluation ourselves. We use reference implementations for
> the inner kernels (BM25 scoring, PageRank's linear algebra) — the standard practice —
> but we own the design, integration, indexing, and analysis.

**Q: Why did MMR make things worse?**
> MMR optimizes for diversity. Legal answers often need several near-identical
> provisions ranked together, so penalizing similarity scattered relevant results and
> lowered MAP. We measured it and disabled it — an evidence-based decision.

**Q: Your graph has no real citations — isn't the authority fake?**
> We're explicit about this. With no stored citations, we derive authority from which
> statutes cases apply. The HITS authorities came out as the Constitution, IPC, CrPC,
> CPC, Evidence Act — the genuinely foundational laws — which validates the signal. It's
> an authority *proxy*, and a defensible one.

**Q: How do you know retrieval actually improved? Did you just eyeball it?**
> No — we used a labelled query set and standard IR metrics (Precision, Recall, MRR,
> nDCG, MAP) with TREC-style pooled relevance judgments. The numbers are reproducible:
> `python -m scripts.eval_retrieval`.

**Q: Does this slow down or risk the existing app?**
> No. Everything is behind feature flags that default to the original behaviour
> (verified by a regression test). The heavy indexes/graph are built offline with no API
> cost and load lazily, falling back to dense search if absent.

**Q: What would you do with more time?**
> Add a learned cross-encoder reranker; obtain real citation data to build a true
> precedent graph; expand the labelled evaluation set for tighter statistical confidence;
> and personalize PageRank to the query (topic-sensitive authority).
