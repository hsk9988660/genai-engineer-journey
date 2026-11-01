"""Tool loadout selection: offer the model the few tools likely needed for THIS step, not all of them.

Too many tools in the context hurts tool choice (the lesson's "context confusion"). The fix is retrieval over tool descriptions, but
retrieval can MISS the tool you need, so this module also provides:
  * `always` tools that are included regardless of the query
  * a `search_tools` fallback the model can call to discover tools that are not in its current loadout
  * `recall_at_k`, a measurement of how often the needed tools are actually selected

Retrieval here is lexical (word overlap weighted by rarity). Lexical search has a vocabulary gap: "get around Paris" does not share
words with "public_transport_info". Tags and good descriptions are how you close it, and the tests show both the gap and the fix.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field

# "get" and "find" are stopwords on purpose: they appear in tool NAMES (get_visa_rules, find_tours), so a user saying "get around" or
# "find a hotel" would otherwise pull in unrelated tools. Measuring per-case recall is how this noise was noticed.
STOP = frozenset("a an and are as at be by for from get find how i in is it me my of on or the to what with you your".split())


def _stem(word: str) -> str:
    """Light plural stemming, so 'hotel' matches 'hotels'. Without it a plain singular query misses the plural description."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _tokens(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower().replace("_", " ")) if t not in STOP]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    tags: tuple = ()  # extra search words: synonyms people will actually type

    @property
    def search_text(self) -> str:
        return f"{self.name} {self.description} {' '.join(self.tags)}"


@dataclass
class Loadout:
    selected: list = field(default_factory=list)
    always: list = field(default_factory=list)
    dropped: list = field(default_factory=list)  # names considered but not offered

    @property
    def names(self) -> list[str]:
        return [t.name for t in self.always] + [t.name for t in self.selected]


class ToolIndex:
    def __init__(self, tools: list[ToolSpec]):
        self.tools = list(tools)
        self._tf = [Counter(_tokens(t.search_text)) for t in self.tools]
        df = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self.tools)
        self._idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def _score(self, query_tokens: list[str], i: int) -> float:
        tf = self._tf[i]
        return sum(self._idf[w] * (1 + math.log(tf[w])) for w in set(query_tokens) if w in tf)

    def ranked(self, query: str) -> list[tuple[ToolSpec, float]]:
        q = _tokens(query)
        scored = [(self.tools[i], self._score(q, i)) for i in range(len(self.tools))]
        return sorted((s for s in scored if s[1] > 0), key=lambda s: (-s[1], s[0].name))

    def select(self, query: str, k: int = 5, always: tuple = ()) -> Loadout:
        """The `always` tools plus the top-k by relevance. Everything else is recorded as dropped."""
        must = [t for t in self.tools if t.name in always]
        chosen = [t for t, _ in self.ranked(query) if t.name not in always][:k]
        offered = {t.name for t in must + chosen}
        return Loadout(chosen, must, [t.name for t in self.tools if t.name not in offered])

    def search_tools(self, query: str, k: int = 5) -> str:
        """The fallback meta-tool: lets the model discover tools that were not in its loadout."""
        hits = [t for t, _ in self.ranked(query)][:k]
        return "\n".join(f"{t.name}: {t.description}" for t in hits) or "No matching tools."


def recall_at_k(index: ToolIndex, cases: list[tuple[str, set]], k: int, always: tuple = ()) -> float:
    """Of the tools each query REQUIRES, what fraction made it into the loadout? Averaged over cases."""
    total = 0.0
    for query, required in cases:
        offered = set(index.select(query, k, always).names)
        total += len(required & offered) / len(required)
    return total / len(cases)
