"""Local document search that is willing to say "nothing relevant".

The lesson's `search_docs` returns the top 2 chunks for ANY query (`n_results=2`, no relevance threshold). Ask it about something its documents never mention and it still hands the
model two documents, which the model will happily cite. Here a result must share a minimum fraction of the query's meaningful words, otherwise the answer is an explicit "no relevant
documentation found", which the model can act on.

This is BM25 with a coverage threshold, in the standard library: no embedding model, no download, no telemetry, and the index can be saved to disk and loaded back (the lesson's
diagram says "on disk" but its code uses Chroma's in-memory client). An embedding model could be added behind the same `search` shape. I have not measured retrieval quality on real
documents, only on the small corpora in the tests.
"""

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

STOP = frozenset("a an the is are was were be been do does did how what why when where which who whom of in on at to for from by with and or but if it its this that these those i you we they me my your our can could should would will shall may might not no yes as about into than then so".split())
K1, B = 1.5, 0.75


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9_]+", text.lower())
    return [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words if w not in STOP]


def chunk_text(text: str, max_chars: int = 500) -> list[str]:
    """Paragraphs first; a paragraph longer than max_chars is split on sentence ends, then by length."""
    chunks = []
    for para in re.split(r"\n\s*\n", text.strip()):
        para = " ".join(para.split())
        while len(para) > max_chars:
            cut = max(para.rfind(". ", 0, max_chars), para.rfind("; ", 0, max_chars))
            cut = cut + 1 if cut > max_chars // 3 else max_chars
            chunks.append(para[:cut].strip())
            para = para[cut:].strip()
        if para:
            chunks.append(para)
    return chunks


@dataclass(frozen=True)
class Hit:
    doc_id: str
    chunk: int
    score: float
    coverage: float
    text: str


class BM25Index:
    def __init__(self):
        self._chunks: list[dict] = []  # {"doc", "n", "text", "tf"}

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, doc_id: str, text: str, max_chars: int = 500) -> int:
        added = 0
        for n, chunk in enumerate(chunk_text(text, max_chars)):
            toks = tokenize(chunk)
            if toks:
                self._chunks.append({"doc": doc_id, "n": n, "text": chunk, "tf": dict(Counter(toks))})
                added += 1
        return added

    def search(self, query: str, k: int = 3, min_coverage: float = 0.5) -> list[Hit]:
        q = list(dict.fromkeys(tokenize(query)))
        if not q or not self._chunks:
            return []
        N = len(self._chunks)
        lengths = [sum(c["tf"].values()) for c in self._chunks]
        avg = sum(lengths) / N
        df = Counter(t for c in self._chunks for t in c["tf"])
        hits = []
        for c, length in zip(self._chunks, lengths):
            matched = [t for t in q if t in c["tf"]]
            coverage = len(matched) / len(q)
            if coverage < min_coverage:
                continue
            score = sum(math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5)) * (c["tf"][t] * (K1 + 1)) / (c["tf"][t] + K1 * (1 - B + B * length / avg)) for t in matched)
            hits.append(Hit(c["doc"], c["n"], round(score, 4), round(coverage, 3), c["text"]))
        hits.sort(key=lambda h: (-h.score, h.doc_id, h.chunk))
        return hits[:k]

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self._chunks), encoding="utf8")

    @classmethod
    def load(cls, path) -> "BM25Index":
        data = json.loads(Path(path).read_text(encoding="utf8"))
        if not isinstance(data, list) or any(not isinstance(c, dict) or set(c) != {"doc", "n", "text", "tf"} for c in data):
            raise ValueError("not a saved index")
        idx = cls()
        idx._chunks = data
        return idx


def search_text(index: BM25Index, query: str, k: int = 3, min_coverage: float = 0.5) -> str:
    hits = index.search(query, k, min_coverage)
    if not hits:
        return "No relevant documentation found. Do not guess; say the documentation does not cover it."
    return "\n".join(f"[{h.doc_id}] {h.text}" for h in hits)
