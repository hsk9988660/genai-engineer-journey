"""Tool results are usually the biggest source of context bloat: web pages, file dumps, database rows.

Two tools against it:
  OFFLOAD  a result over a token threshold is stored OUTSIDE the context. The model gets a short preview, a handle, and a way
           to fetch it page by page (the "sandbox / offload" pattern from the lesson). Handles are unguessable and every page
           fetch is size-limited.
  STUB     once a result has been used, an older one is replaced by a one-line stub (what it was, how big, where the full text is).

Offloading is for the model's benefit only. The full text is kept, so nothing is lost.
"""

import secrets
from dataclasses import dataclass

from tokens import estimate_tokens


@dataclass
class ToolItem:
    id: str
    tool: str
    text: str  # what goes into the context (possibly a preview)
    handle: str | None = None
    full_tokens: int = 0  # size of the ORIGINAL result


def stub_text(item: ToolItem) -> str:
    where = f"; full output stored as {item.handle}" if item.handle else ""
    return f"[tool result cleared: {item.tool}, about {item.full_tokens} tokens{where}]"


class ToolResultStore:
    def __init__(self, offload_over_tokens: int = 500, preview_chars: int = 240, page_chars: int = 2000, count=estimate_tokens):
        self.offload_over, self.preview_chars, self.page_chars, self.count = offload_over_tokens, preview_chars, page_chars, count
        self._store: dict[str, str] = {}
        self._n = 0

    def add(self, tool: str, text: str) -> ToolItem:
        self._n += 1
        tokens = self.count(text)
        item_id = f"tr{self._n}"
        if tokens <= self.offload_over:
            return ToolItem(item_id, tool, text, None, tokens)
        handle = "res_" + secrets.token_hex(8)  # unguessable: one session cannot read another's results by guessing
        self._store[handle] = text
        preview = text[:self.preview_chars].replace("\n", " ")
        note = (f"[large result from {tool}: about {tokens} tokens, stored as {handle}. Preview: {preview}... "
                f"Call fetch_result(handle, offset) to read more.]")
        return ToolItem(item_id, tool, note, handle, tokens)

    def fetch(self, handle: str, offset: int = 0, length: int | None = None) -> str:
        """One bounded page of an offloaded result."""
        text = self._store.get(handle)
        if text is None:
            raise KeyError("unknown handle")
        if offset < 0:
            raise ValueError("offset must be >= 0")
        size = min(length if length is not None else self.page_chars, self.page_chars)  # a caller cannot ask for the whole thing at once
        return text[offset:offset + size]
