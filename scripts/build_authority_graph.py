"""
Build the Legal Authority Graph (offline, no API cost).

The case corpus has no stored case-to-case citations, so authority is derived from a
**case <-> statute bipartite graph**: an edge connects a case to each statute it applies
(from summary.laws_applied). On this graph we run:

  * PageRank  -> per-CASE authority score (used to re-rank retrieved cases)
  * HITS      -> statute authority ranking (most-central statutes) + case hub scores
  * Louvain   -> communities (legal domains), characterised by their top statutes

Running PageRank/HITS on the bipartite graph directly avoids the O(n^2) edge explosion
that a case-case projection would create from ubiquitous "hub" statutes such as the
Constitution of India (applied by hundreds of cases) — a deliberate design choice.

Outputs (settings.BM25_INDEX_DIR / settings.AUTHORITY_SCORES_PATH):
  data/authority_scores.json   {case_id: normalised_pagerank}
  data/graph_stats.json        summary stats for the report / slides
  data/case_graph.graphml      focused subgraph (top cases + statutes) for Gephi

Usage (from backend-ai-lawyer/):
    venv/Scripts/python.exe -m scripts.build_authority_graph
"""
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import networkx as nx          # noqa: E402
from tqdm import tqdm           # noqa: E402

from app.config import settings           # noqa: E402
from app.services.qdrant import qdrant_service  # noqa: E402

# Statute-name canonicalisation -------------------------------------------------
_ABBREV = {
    "ipc": "indian penal code", "crpc": "code of criminal procedure",
    "cpc": "code of civil procedure", "bns": "bharatiya nyaya sanhita",
    "constitution": "constitution of india",
}
_DROP_PREFIX = re.compile(r"^(section|sections|sec|s|order|rule|rules|article|articles|art)\b[\s.]*\d*", re.I)


def canonical_statute(item) -> str:
    """Reduce a laws_applied entry to an act-level canonical name (or '')."""
    if isinstance(item, dict):
        # Prefer the act-level reference if it names an Act/Code/Constitution.
        ref = (item.get("reference") or "").strip()
        name = (item.get("name") or "").strip()
        raw = ref if re.search(r"act|code|constitution|sanhita|adhiniyam", ref, re.I) else (name or ref)
    else:
        raw = str(item or "")
    raw = raw.lower().strip()
    if not raw:
        return ""
    raw = re.sub(r",?\s*\d{4}\b", "", raw)                # drop years
    raw = _DROP_PREFIX.sub("", raw).strip(" .,-")          # drop "Section 302" -> ""
    raw = re.sub(r"[^a-z0-9 ]", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    raw = _ABBREV.get(raw, raw)
    # require something act-like and not a bare number/short token
    if len(raw) < 4 or raw.isdigit():
        return ""
    return raw


def load_corpus():
    client = qdrant_service.client
    cases = {}          # case_id -> {title, year, statutes:set, keywords:list}
    statute_df = Counter()
    next_off = None
    pbar = tqdm(desc="scroll cases", unit="doc")
    while True:
        points, next_off = client.scroll(
            collection_name=settings.QDRANT_COLLECTION_CASES, limit=1000,
            offset=next_off, with_payload=True, with_vectors=False)
        if not points:
            break
        for p in points:
            pl = p.payload or {}
            summ = pl.get("summary", {}) or {}
            laws = summ.get("laws_applied", []) or [] if isinstance(summ, dict) else []
            statutes = {s for s in (canonical_statute(l) for l in laws) if s}
            if not statutes:
                continue  # isolated node contributes nothing to the graph
            cid = str(p.id)
            cases[cid] = {
                "title": pl.get("title", "Untitled"),
                "year": pl.get("year", ""),
                "statutes": statutes,
                "keywords": (summ.get("keywords", []) or [])[:6] if isinstance(summ, dict) else [],
            }
            for s in statutes:
                statute_df[s] += 1
        pbar.update(len(points))
        if next_off is None:
            break
    pbar.close()
    return cases, statute_df


def main():
    t0 = time.time()
    print("=== Building Legal Authority Graph ===")
    cases, statute_df = load_corpus()
    print(f"  cases with statutes: {len(cases):,} | distinct statutes: {len(statute_df):,}")

    # Bipartite graph: case nodes + statute nodes
    B = nx.Graph()
    for cid, c in cases.items():
        B.add_node(("c", cid), kind="case")
        for s in c["statutes"]:
            B.add_node(("s", s), kind="statute")
            B.add_edge(("c", cid), ("s", s))
    print(f"  graph: {B.number_of_nodes():,} nodes, {B.number_of_edges():,} edges")

    print("  PageRank...")
    pr = nx.pagerank(B, alpha=0.85, max_iter=200)
    case_pr = {cid: pr.get(("c", cid), 0.0) for cid in cases}
    mx = max(case_pr.values()) or 1.0
    authority_scores = {cid: round(v / mx, 6) for cid, v in case_pr.items()}  # normalised 0..1

    print("  HITS...")
    try:
        hubs, auth = nx.hits(B, max_iter=500, normalized=True)
    except Exception as e:
        print(f"   HITS did not converge ({e}); using degree as fallback.")
        auth = {n: B.degree(n) for n in B}
    statute_auth = sorted(((s, auth.get(("s", s), 0.0), statute_df[s]) for s in statute_df),
                          key=lambda x: x[1], reverse=True)

    print("  Louvain communities...")
    communities = nx.community.louvain_communities(B, seed=42)
    comm_summaries = []
    for comm in sorted(communities, key=len, reverse=True)[:10]:
        statutes_in = Counter()
        n_cases = 0
        for node in comm:
            if node[0] == "s":
                statutes_in[node[1]] += statute_df[node[1]]
            else:
                n_cases += 1
        comm_summaries.append({
            "size": len(comm), "cases": n_cases,
            "top_statutes": [s for s, _ in statutes_in.most_common(5)],
        })

    # Top authority cases
    top_cases = sorted(cases.keys(), key=lambda c: authority_scores[c], reverse=True)[:25]
    top_cases_out = [{"case_id": c, "title": cases[c]["title"], "year": cases[c]["year"],
                      "authority": authority_scores[c], "n_statutes": len(cases[c]["statutes"])}
                     for c in top_cases]

    degrees = [d for _, d in B.degree() if isinstance(_, tuple)]
    stats = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_cases": len(cases), "n_statutes": len(statute_df),
        "n_nodes": B.number_of_nodes(), "n_edges": B.number_of_edges(),
        "avg_degree": round(sum(degrees) / len(degrees), 2) if degrees else 0,
        "n_communities": len(communities),
        "top_authority_cases": top_cases_out,
        "top_authority_statutes": [{"statute": s, "hits_authority": round(a, 6), "applied_in_cases": df}
                                   for s, a, df in statute_auth[:20]],
        "community_summaries": comm_summaries,
    }

    # Persist
    os.makedirs(settings.BM25_INDEX_DIR, exist_ok=True)
    with open(ROOT / settings.AUTHORITY_SCORES_PATH, "w", encoding="utf-8") as f:
        json.dump(authority_scores, f)
    with open(os.path.join(settings.BM25_INDEX_DIR, "graph_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Focused GraphML for visualisation: top cases + their statutes
    focus_nodes = set()
    for c in top_cases:
        focus_nodes.add(("c", c))
        focus_nodes |= {("s", s) for s in cases[c]["statutes"]}
    sub = B.subgraph(focus_nodes).copy()
    for node in sub.nodes:
        sub.nodes[node]["label"] = (cases[node[1]]["title"][:40] if node[0] == "c" else node[1])
        sub.nodes[node]["kind"] = "case" if node[0] == "c" else "statute"
    sub = nx.relabel_nodes(sub, {n: f"{n[0]}:{n[1]}" for n in sub.nodes})
    nx.write_graphml(sub, os.path.join(settings.BM25_INDEX_DIR, "case_graph.graphml"))

    print(f"\n  Saved authority_scores.json ({len(authority_scores):,} cases), graph_stats.json, "
          f"case_graph.graphml")
    print(f"  Top authority statutes: {[s for s,_,_ in statute_auth[:5]]}")
    print(f"  Top authority case: {top_cases_out[0]['title'][:60]} (auth={top_cases_out[0]['authority']})")
    print(f"  Done in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
