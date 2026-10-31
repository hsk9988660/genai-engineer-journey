"""Keyword (BM25), vector, and hybrid search, with permission filtering.

All pure Python so it runs offline. The vector side is a pluggable `embedder`: the default
`HashingEmbedder` is a deterministic character-n-gram model. It is NOT semantic: it matches word
shapes ("cancel" ~ "cancellation", typos), not meaning ("8 year old" ~ "unaccompanied minor").
For real semantic search, pass an object with `embed(list[str]) -> list[list[float]]` that calls an
embedding provider and returns unit-length vectors.
"""

import math
import re
import zlib
from collections import Counter
from dataclasses import dataclass

from chunker import Chunk

STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i if in is it me my of on or the this that to "
    "was we what when where which who will with you your get".split()
)
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")  # keeps codes like fare-204 and ins-200 as one token


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    for suffix in ("ing", "ed"):
        if len(token) > 5 and token.endswith(suffix):
            return token[: -len(suffix)]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str, stem: bool = True) -> list[str]:
    tokens = [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]
    return [_stem(t) for t in tokens] if stem else tokens


class BM25:
    """Scores a chunk by the query words it contains, weighting rare words more (idf)."""

    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(t) for t in docs_tokens]
        self.lengths = [len(t) for t in docs_tokens]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if docs_tokens else 1.0
        df = Counter()
        for t in docs_tokens:
            df.update(set(t))
        n = len(docs_tokens)
        self.idf = {term: math.log(1 + (n - c + 0.5) / (c + 0.5)) for term, c in df.items()}

    def score(self, query_tokens: list[str], i: int) -> float:
        tf, total = self.tf[i], 0.0
        for term in set(query_tokens):
            f = tf.get(term)
            if f:
                norm = f + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_len)
                total += self.idf[term] * f * (self.k1 + 1) / norm
        return total


class HashingEmbedder:
    """Deterministic character-trigram embeddings (crc32, not hash(), which is randomized per process)."""

    def __init__(self, dim: int = 512, n: int = 3):
        self.dim, self.n = dim, n

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for token in tokenize(text, stem=False):
            padded = f" {token} "
            for i in range(max(1, len(padded) - self.n + 1)):
                h = zlib.crc32(padded[i : i + self.n].encode())
                v[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Merge ranked lists by rank position only, so BM25 and cosine scores need no common scale."""
    fused: Counter = Counter()
    for ranking in rankings:
        for position, item in enumerate(ranking):
            fused[item] += 1.0 / (k + position + 1)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class HybridIndex:
    def __init__(self, chunks: list[Chunk], embedder=None, min_cosine: float = 0.3):
        # 0.3 was chosen by measurement on the 14 demo questions (at 0.2 the trigram embedder let in noise and
        # hybrid precision fell below BM25). Re-tune for any real embedder and any real corpus.
        self.chunks = list(chunks)
        self.by_id = {c.id: c for c in self.chunks}
        self.embedder, self.min_cosine = embedder or HashingEmbedder(), min_cosine
        self.bm25 = BM25([tokenize(c.indexed_text) for c in self.chunks])
        self.vectors = self.embedder.embed([c.indexed_text for c in self.chunks])

    def _bm25_ranking(self, query: str, allowed: list[int]) -> list[int]:
        q = tokenize(query)
        scored = [(self.bm25.score(q, i), i) for i in allowed]
        return [i for s, i in sorted(scored, key=lambda x: (-x[0], x[1])) if s > 0]

    def _vector_ranking(self, query: str, allowed: list[int]) -> list[int]:
        qv = self.embedder.embed([query])[0]
        sims = [(sum(a * b for a, b in zip(qv, self.vectors[i])), i) for i in allowed]
        return [i for s, i in sorted(sims, key=lambda x: (-x[0], x[1])) if s >= self.min_cosine]

    def search(self, query: str, k: int = 4, roles=("public",), mode: str = "hybrid") -> list[Hit]:
        """Return the top-k chunks the caller is allowed to see.

        Permissions are applied BEFORE ranking: a chunk the user may not read is never a candidate, so it
        cannot reach the model, however the query is worded.
        """
        roles = frozenset(roles)
        allowed = [i for i, c in enumerate(self.chunks) if c.access & roles]
        if mode == "bm25":
            ranked = [(i, 0.0) for i in self._bm25_ranking(query, allowed)]
        elif mode == "vector":
            ranked = [(i, 0.0) for i in self._vector_ranking(query, allowed)]
        elif mode == "hybrid":
            ranked = reciprocal_rank_fusion([self._bm25_ranking(query, allowed), self._vector_ranking(query, allowed)])
        else:
            raise ValueError(f"unknown mode: {mode}")
        return [Hit(self.chunks[i], score) for i, score in ranked[:k]]
