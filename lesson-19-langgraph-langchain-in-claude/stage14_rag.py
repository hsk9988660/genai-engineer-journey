"""Stage 14: retrieval (RAG) with LangChain. Documents, splitting, embeddings, a vector store, and answers you can check.

RAG means: before the model answers, look up relevant text and put it in the prompt. The model then answers from that
text instead of from memory. The LangChain pieces, in the order the data flows:

    Document ──► text splitter ──► chunks ──► Embeddings ──► vector store ──► retriever / search ──► prompt ──► model

| piece                        | job                                                                          |
|------------------------------|------------------------------------------------------------------------------|
| `Document`                   | `page_content` (text) plus `metadata` (a dict: id, audience, source...)       |
| `RecursiveCharacterTextSplitter` | cuts long text into chunks, preferring paragraph and sentence breaks. Chunks keep the metadata |
| `Embeddings`                 | turns text into a vector. Similar meaning should give a close vector         |
| `InMemoryVectorStore`        | stores chunk vectors; `similarity_search_with_score(query, k, filter=...)`   |
| retriever                    | a vector store wrapped as a runnable: `store.as_retriever(search_kwargs=...)` |

Three rules that matter more than the choice of vector store (they are the lessons of lesson 05, now on LangChain):

1. FILTER BEFORE YOU RANK. A staff-only article must never be a candidate for a customer, even if it is the best match.
   The store's `filter=` runs before ranking, so a restricted chunk cannot even take a slot in the top k.
2. SAY "NOTHING RELEVANT". A vector store always returns its k nearest chunks, however far away. A score threshold turns
   "nearest" into "relevant", and an empty result means the model is not called and does not guess.
3. VERIFY CITATIONS IN CODE. The model is asked to cite chunk ids. Code checks every cited id was really retrieved.
   A citation proves the model SAID where it got something, not that the source supports it.

The embeddings here are a word-hashing stand-in, so the lesson needs no model and no API key. They match shared words,
not meanings. A real embedding model also matches synonyms. See the test that shows the difference.

Run the offline demo:  python stage14_rag.py
"""

import hashlib
import math
import re
from dataclasses import dataclass, field

from langchain.tools import ToolRuntime, tool
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from fakes import ScriptedChatModel, say
from tools import DeskContext

ARTICLES = [
    {"id": "shipping", "audience": "public", "text": (
        "Standard shipping takes 3 to 5 business days. Express shipping takes 1 to 2 business days and costs extra. "
        "Orders placed after 2 pm ship the next business day. You receive a tracking link by email when the parcel leaves our warehouse.")},
    {"id": "returns", "audience": "public", "text": (
        "You can return an item within 30 days of delivery. The item must be unused and in its original packaging. "
        "A refund goes back to the original payment method within 5 to 7 business days after we receive the item. "
        "Sale items can be returned for store credit only.")},
    {"id": "warranty", "audience": "public", "text": (
        "Every product has a one-year warranty against manufacturing defects. The warranty does not cover wear and tear, damage from misuse, or loss. "
        "To claim, send your order id and a photo of the defect. We repair or replace the item at our choice.")},
    {"id": "care", "audience": "public", "text": (
        "Wash waterproof jackets at 30 degrees with a technical wash and never use fabric softener. Tumble dry on low heat to renew the water repellent finish. "
        "Store boots clean and dry, away from direct heat.")},
    {"id": "margins", "audience": "staff", "text": (
        "Internal only: our average gross margin on footwear is 42 percent. Do not share margins with customers. "
        "Agents may refund up to 200 dollars per step. Larger refunds go to a manager.")},
]

VISIBLE = {"public": {"public"}, "staff": {"public", "staff"}}  # who may see which articles
NOTHING_FOUND = "I could not find that in our help articles. I have passed your question to a person."
STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "does", "for", "from", "how", "i", "if", "in", "is", "it", "my", "of", "on",
             "or", "our", "so", "the", "to", "we", "what", "when", "with", "you", "your", "can", "will", "that", "this", "me", "us"}
CITATION = re.compile(r"\[([\w-]+)\]")


class HashingEmbeddings(Embeddings):
    """A word-hashing embedding: each word adds weight to one of `dims` slots. Shared words give similar vectors."""

    def __init__(self, dims: int = 4096) -> None:
        self.dims = dims

    def words(self, text: str) -> list[str]:
        raw = re.findall(r"[a-z0-9]+", text.lower())
        return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in raw if w not in STOPWORDS]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dims
        for word in self.words(text):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


def to_documents(articles: list[dict] = ARTICLES) -> list[Document]:
    return [Document(page_content=a["text"], metadata={"article": a["id"], "audience": a["audience"]}) for a in articles]


def split(documents: list[Document], *, chunk_size: int = 140, chunk_overlap: int = 25) -> list[Document]:
    """Cut documents into chunks. Each chunk gets its own id, `article-n`, so a citation can point at an exact chunk."""
    chunks = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_documents(documents)
    counters: dict[str, int] = {}
    for chunk in chunks:
        article = chunk.metadata["article"]
        counters[article] = counters.get(article, 0) + 1
        chunk.metadata["id"] = f"{article}-{counters[article]}"
    return chunks


class KnowledgeBase:
    def __init__(self, documents: list[Document] | None = None, *, embeddings: Embeddings | None = None, min_score: float = 0.2) -> None:
        self.store = InMemoryVectorStore(embeddings or HashingEmbeddings())
        self.min_score = min_score
        self.chunks = split(documents if documents is not None else to_documents())
        self.store.add_documents(self.chunks, ids=[c.metadata["id"] for c in self.chunks])

    def search(self, query: str, audience: str = "public", *, k: int = 3) -> list[tuple[Document, float]]:
        """The best chunks this audience may see, best first, keeping only those above the relevance threshold."""
        allowed = VISIBLE.get(audience, set())  # an unknown audience sees nothing
        hits = self.store.similarity_search_with_score(query, k=k, filter=lambda doc: doc.metadata["audience"] in allowed)
        return [(doc, score) for doc, score in sorted(hits, key=lambda pair: -pair[1]) if score >= self.min_score]

    def forget(self, article: str) -> int:
        """Remove every chunk of an article, for example when it is withdrawn. Returns how many were removed."""
        ids = [c.metadata["id"] for c in self.chunks if c.metadata["article"] == article]
        self.store.delete(ids=ids)
        self.chunks = [c for c in self.chunks if c.metadata["article"] != article]
        return len(ids)


ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Answer the customer's question using ONLY the sources below. After each fact, cite its source id in square brackets, "
               "for example [shipping-1]. If the sources do not answer the question, say so.\n\nSources:\n{context}"),
    ("human", "{question}"),
])


@dataclass
class RagAnswer:
    text: str
    sources: list[str] = field(default_factory=list)  # ids that were put in the prompt
    cited: list[str] = field(default_factory=list)  # ids the model cited and that were really in the prompt
    invalid: list[str] = field(default_factory=list)  # ids the model cited that were NOT in the prompt
    grounded: bool = False  # True only if the model cited at least one real source and invented none


def ask_knowledge_base(model: BaseChatModel, kb: KnowledgeBase, question: str, audience: str = "public", *, k: int = 3) -> RagAnswer:
    hits = kb.search(question, audience, k=k)
    if not hits:
        return RagAnswer(NOTHING_FOUND)  # the model is never called, so it cannot guess
    sources = [doc.metadata["id"] for doc, _ in hits]
    context = "\n".join(f"[{doc.metadata['id']}] {doc.page_content}" for doc, _ in hits)
    text = (ANSWER_PROMPT | model | StrOutputParser()).invoke({"context": context, "question": question})
    cited = list(dict.fromkeys(CITATION.findall(text)))
    valid = [c for c in cited if c in sources]
    invalid = [c for c in cited if c not in sources]
    return RagAnswer(text, sources, valid, invalid, grounded=bool(valid) and not invalid)


def make_search_tool(kb: KnowledgeBase):
    @tool
    def search_knowledge_base(query: str, runtime: ToolRuntime[DeskContext]) -> dict:
        """Search the help articles. Returns matching passages with their ids, or an empty list when nothing is relevant."""
        hits = kb.search(query, runtime.context.audience)  # the audience comes from verified context, never from the model
        return {"passages": [{"id": doc.metadata["id"], "text": doc.page_content} for doc, _ in hits]}

    return search_knowledge_base


def demo() -> None:
    kb = KnowledgeBase()
    print(f"{len(kb.chunks)} chunks from {len(ARTICLES)} articles. First three:")
    for chunk in kb.chunks[:3]:
        print(f"  {chunk.metadata['id']:10} {chunk.page_content!r}")
    print("\npublic asks about margins:", [d.metadata["id"] for d, _ in kb.search("what is the footwear margin", "public")])
    print("staff asks the same      :", [d.metadata["id"] for d, _ in kb.search("what is the footwear margin", "staff")])
    print("unrelated question       :", kb.search("do you rent kayaks in the summer", "public"))
    model = ScriptedChatModel(script=[say("Standard shipping takes 3 to 5 business days [shipping-1].")])
    print("\n", ask_knowledge_base(model, kb, "How long does shipping take?"))


if __name__ == "__main__":
    demo()
