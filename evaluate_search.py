#!/usr/bin/env python3
"""Search quality evaluation script.

Reproduces the Hit Rate @3 benchmark reported in RESULTS.md across the
12 starter notes using both the Keyword Baseline and TF-IDF Hybrid Search.
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import json
from pathlib import Path
from src.config import STARTER_NOTES_JSON, SEARCH_QUERIES_JSON
from src.search import KeywordSearchBaseline, TfidfHybridSearchEngine


def main():
    print("=" * 70)
    print("SMART NOTES ASSISTANT - SEARCH QUALITY BENCHMARK (Hit Rate @3)")
    print("=" * 70)

    # Load starter notes and test queries
    with open(STARTER_NOTES_JSON, 'r', encoding='utf-8') as f:
        notes = json.load(f)

    with open(SEARCH_QUERIES_JSON, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    zero_overlap_count = sum(1 for q in queries if q.get("no_overlap", False))
    print(f"Total benchmark queries: {len(queries)}")
    print(f"Queries with zero content-word overlap: {zero_overlap_count}")
    print(f"Evaluated against {len(notes)} starter notes (N001-N012)\n")

    baseline_engine = KeywordSearchBaseline()
    production_engine = TfidfHybridSearchEngine(min_score=0.05)

    engines = [
        ("Keyword Baseline (Token Overlap)", baseline_engine),
        ("TF-IDF Hybrid (Word + Subword Char n-gram)", production_engine)
    ]

    benchmark_summary = {}

    for name, engine in engines:
        print("-" * 70)
        print(f"EVALUATING: {name}")
        print("-" * 70)

        hits = 0
        missed = []

        for item in queries:
            q = item["query"]
            expected = item["expected"]
            results = engine.search(q, notes, top_k=3)
            retrieved_ids = [r["id"] for r in results]

            is_hit = any(exp in retrieved_ids for exp in expected)
            hits += int(is_hit)
            status = "HIT " if is_hit else "MISS"

            zero_flag = " [ZERO-OVERLAP]" if item.get("no_overlap") else ""
            print(f"  [{status}]{zero_flag:15s} Query: \"{q}\"")
            print(f"         Expected: {expected} | Retrieved: {retrieved_ids}")

            if not is_hit:
                missed.append({
                    "query": q,
                    "expected": expected,
                    "retrieved": retrieved_ids,
                    "zero_overlap": item.get("no_overlap", False)
                })

        hit_rate = hits / len(queries)
        benchmark_summary[name] = {
            "hit_rate": hit_rate,
            "hits": hits,
            "total": len(queries),
            "missed": missed
        }
        print(f"\n=> Hit Rate @3 for '{name}': {hit_rate:.4f} ({hits}/{len(queries)})\n")

    print("=" * 70)
    print("COMPARATIVE SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Search Approach':45s} | {'Hit Rate @3':15s}")
    print("-" * 65)
    for name, stats in benchmark_summary.items():
        rate_str = f"{stats['hit_rate'] * 100:.1f}% ({stats['hits']}/{stats['total']})"
        print(f"{name:45s} | {rate_str:15s}")

    print("\n[OK] Search benchmark evaluation finished successfully.\n")


if __name__ == '__main__':
    main()
