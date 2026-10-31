"""Evaluate retrieval and answers separately, so you can tell which half broke.

  python evals.py retrieval   offline, deterministic, no API key: recall@k and MRR per search mode
  python evals.py live        needs ANTHROPIC_API_KEY: plain RAG vs agentic RAG on the same questions

Retrieval is scored at DOCUMENT level (did a chunk from the right document appear in the top k?).
Questions with no gold document are skipped for retrieval metrics: they test abstention instead.
With 14 cases these numbers are illustrative, not statistically meaningful. Grow the set before you trust it.
"""

import sys

from corpus import CASES, Case, build_chunks
from index import HybridIndex


def doc_ranking(index: HybridIndex, case: Case, k: int, mode: str) -> list[str]:
    seen = []
    for hit in index.search(case.question, k=k, roles=case.roles, mode=mode):
        if hit.chunk.doc_id not in seen:
            seen.append(hit.chunk.doc_id)
    return seen


def retrieval_report(index: HybridIndex, cases=CASES, k: int = 3, modes=("bm25", "vector", "hybrid")) -> dict:
    scored = [c for c in cases if c.gold_docs]
    report = {}
    for mode in modes:
        hits, rr, precision, misses = 0, 0.0, 0.0, []
        for case in scored:
            ranking = doc_ranking(index, case, k, mode)
            returned = index.search(case.question, k=k, roles=case.roles, mode=mode)
            if returned:  # share of returned chunks that come from a gold document: how much noise the model must ignore
                precision += sum(h.chunk.doc_id in case.gold_docs for h in returned) / len(returned)
            positions = [i for i, d in enumerate(ranking) if d in case.gold_docs]
            if positions:
                hits += 1
                rr += 1.0 / (positions[0] + 1)
            else:
                misses.append(case.question)
        report[mode] = {"recall@k": hits / len(scored), "mrr": rr / len(scored),
                        "precision": precision / len(scored), "misses": misses}
    return report


def check_answer(result, case: Case, index: HybridIndex) -> tuple[bool, str]:
    """Pass/fail for one answer. `result` is a RagResult from agent.answer_question."""
    text = result.answer.answer.lower()
    for bad in case.forbidden:
        if bad.lower() in text:
            return False, f"leaked forbidden string {bad!r}"
    if not case.gold_docs:  # unanswerable (or not permitted): must abstain
        return (not result.answer.answerable), ("ok" if not result.answer.answerable else "answered an unanswerable question")
    if not result.answer.answerable:
        return False, "abstained but the answer was available"
    if not result.verified:
        return False, "citations failed verification"
    cited_docs = {index.by_id[c.chunk_id].doc_id for c in result.answer.citations if c.chunk_id in index.by_id}
    if not cited_docs & case.gold_docs:
        return False, f"cited {sorted(cited_docs)}, expected one of {sorted(case.gold_docs)}"
    if case.must_contain and case.must_contain.lower() not in text:
        return False, f"missing fact {case.must_contain!r}"
    return True, "ok"


def run_answer_eval(client, index: HybridIndex, mode: str, cases=CASES, **kwargs) -> dict:
    from agent import answer_question

    rows = []
    for case in cases:
        try:
            result = answer_question(client, case.question, index, case.roles, mode=mode, **kwargs)
            passed, detail = check_answer(result, case, index)
            rows.append({"question": case.question, "passed": passed, "detail": detail,
                         "searches": len(result.searches), "tokens": result.input_tokens + result.output_tokens})
        except Exception as exc:  # a crash is a failed case, not a crashed eval
            rows.append({"question": case.question, "passed": False, "detail": f"error: {exc}", "searches": 0, "tokens": 0})
    n = len(rows)
    return {"pass_rate": sum(r["passed"] for r in rows) / n, "avg_searches": sum(r["searches"] for r in rows) / n,
            "avg_tokens": sum(r["tokens"] for r in rows) / n, "rows": rows}


if __name__ == "__main__":
    index = HybridIndex(build_chunks())
    which = sys.argv[1] if len(sys.argv) > 1 else "retrieval"
    if which == "retrieval":
        for k in (1, 3):
            print(f"--- retrieval, top {k} chunks ---")
            for mode, r in retrieval_report(index, k=k).items():
                print(f"{mode:7s} recall@{k}={r['recall@k']:.2f} mrr={r['mrr']:.2f} precision={r['precision']:.2f}")
                if k == 3:
                    for q in r["misses"]:
                        print(f"          missed: {q}")
    elif which == "live":
        import anthropic

        client = anthropic.Anthropic()
        for mode in ("plain", "agentic"):
            r = run_answer_eval(client, index, mode)
            print(f"\n=== {mode}: pass {r['pass_rate']:.0%}, avg searches {r['avg_searches']:.1f}, avg tokens {r['avg_tokens']:.0f}")
            for row in r["rows"]:
                print(("PASS" if row["passed"] else "FAIL"), "|", row["question"], "|", row["detail"])
    else:
        print("usage: python evals.py [retrieval|live]")
