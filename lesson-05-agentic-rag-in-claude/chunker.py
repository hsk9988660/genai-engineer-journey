"""Split documents into chunks that keep their metadata.

Chunking is where many RAG systems quietly fail. Too big and relevance gets diluted. Too small and a
chunk loses the context needed to be understood. This chunker packs whole sentences up to `max_chars`
and repeats the last sentence(s) at the start of the next chunk (overlap), so a fact split across a
boundary still appears intact in one chunk.
"""

import re
from dataclasses import dataclass

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    id: str  # "<doc_id>#<n>": what the model cites
    doc_id: str
    source: str  # human-readable title, shown to the model and used in citations
    text: str  # the raw words: what quotes are verified against
    access: frozenset  # roles allowed to see this chunk, checked at retrieval time

    @property
    def indexed_text(self) -> str:
        # "Contextual header": prefix the source title so a chunk from the middle of a document
        # still mentions what it is about.
        return f"{self.source}. {self.text}"


def chunk_document(doc_id: str, source: str, text: str, access, max_chars: int = 260,
                   overlap_sentences: int = 1) -> list[Chunk]:
    sentences = [s for s in _SENTENCE_END.split(" ".join(text.split())) if s]
    chunks: list[Chunk] = []
    current: list[str] = []

    def flush():
        if current:
            chunks.append(Chunk(f"{doc_id}#{len(chunks)}", doc_id, source, " ".join(current), frozenset(access)))

    for sentence in sentences:
        if current and len(" ".join(current + [sentence])) > max_chars:
            flush()
            tail = current[-overlap_sentences:] if overlap_sentences else []
            # Drop the overlap if it would push the new chunk over the limit.
            current = tail if len(" ".join(tail + [sentence])) <= max_chars else []
        current = current + [sentence]  # a single over-long sentence still becomes its own chunk
    flush()
    return chunks
