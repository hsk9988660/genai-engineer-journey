"""Stage 12: LangChain runnables and the `|` pipe. Prompts, models and parsers as building blocks.

Everything in LangChain that does work is a RUNNABLE: a prompt, a chat model, an output parser, a plain function wrapped
in `RunnableLambda`, a whole agent, a LangGraph graph. Every runnable has the same methods:

    .invoke(x)    one input, one output          .batch([x, y])   many inputs, outputs in the same order
    .stream(x)    output in pieces               .ainvoke / .abatch / .astream   the async twins

The `|` operator joins runnables: `a | b` means "run a, give its output to b". The result is itself a runnable, so
chains nest and can be joined again. This is called LCEL (LangChain Expression Language).

Building blocks used here:

| block                          | what it does                                                                  |
|--------------------------------|-------------------------------------------------------------------------------|
| `ChatPromptTemplate`           | turns a dict of variables into a list of messages                             |
| `MessagesPlaceholder`          | inserts a whole list of earlier messages at that spot                         |
| a chat model                   | messages in, `AIMessage` out                                                  |
| `StrOutputParser`, `PydanticOutputParser` | `AIMessage` in, a string or a typed object out                     |
| `RunnableLambda(fn)`           | any Python function becomes a step. Your own code belongs here, not in a prompt |
| `RunnableParallel(a=..., b=...)` | runs branches on the same input, returns a dict                             |
| `RunnablePassthrough.assign(x=...)` | keeps the input dict and adds a key                                      |
| `.with_retry(...)`, `.with_fallbacks([...])` | resilience wrapped around any runnable                           |

The example: drafting a reply to a customer. Personal data is removed by a plain function BEFORE the prompt, so the
model never sees it. That is a rule in code, not a request in a prompt.

Run the offline demo:  python stage12_runnables.py
"""

import re
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough
from pydantic import BaseModel

from fakes import ScriptedChatModel, say

SIGNATURE = "Outdoor Gear Support"
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE = re.compile(r"(?<!\w)\+?\(?\d[\d ().-]{7,}\d")

REPLY_RULES = ("You write short, kind replies for an outdoor-gear shop. Use only facts from the customer's message. "
               "Never promise a refund. Never ask for payment details.")


def redact(text: str) -> str:
    """Replace emails and phone numbers. Regexes miss names and addresses, and treat a long run of digits such as a date as a phone
    number (a known false positive). See lesson 06 for the limits."""
    return PHONE.sub("[phone]", EMAIL.sub("[email]", text))


def clean_input(values: dict) -> dict:
    return {**values, "message": redact(values["message"])}


def add_signature(reply: str) -> str:
    return f"{reply.strip()}\n\n{SIGNATURE}"


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", REPLY_RULES),
        MessagesPlaceholder("history", optional=True),
        ("human", "Customer {name} writes:\n{message}"),
    ])


def build_reply_chain(model: BaseChatModel) -> Runnable:
    """dict in -> redact -> prompt -> model -> text -> signature -> str out."""
    return RunnableLambda(clean_input) | build_prompt() | model | StrOutputParser() | RunnableLambda(add_signature)


class Sentiment(BaseModel):
    label: Literal["positive", "neutral", "negative"]
    reason: str


def build_analysis_chain(sentiment_model: BaseChatModel, summary_model: BaseChatModel) -> Runnable:
    """Two independent questions about one message, asked at the same time. Returns {"sentiment": Sentiment, "summary": str}."""
    parser = PydanticOutputParser(pydantic_object=Sentiment)
    sentiment_prompt = ChatPromptTemplate.from_messages([
        ("system", "Classify the customer's mood.\n{format_instructions}"),
        ("human", "{message}"),
    ]).partial(format_instructions=parser.get_format_instructions())  # fill one variable now, leave the rest for later
    summary_prompt = ChatPromptTemplate.from_messages([("system", "Summarise in one sentence."), ("human", "{message}")])
    return RunnableLambda(clean_input) | RunnableParallel(
        sentiment=sentiment_prompt | sentiment_model | parser,
        summary=summary_prompt | summary_model | StrOutputParser(),
    )


def build_drafting_chain(model: BaseChatModel) -> Runnable:
    """Keep the input and add a `reply` key: {"name", "message", "reply"}."""
    return RunnablePassthrough.assign(reply=build_reply_chain(model))


def fallback_reply(values: dict) -> str:
    return add_signature(f"Hello {values['name']}, thank you for writing. A member of our team will reply personally soon.")


def build_reliable_reply_chain(primary: BaseChatModel, *, attempts: int = 2) -> Runnable:
    """Retry a connection problem a few times; if the model is still unavailable, send a safe canned reply instead."""
    reply = build_reply_chain(primary).with_retry(retry_if_exception_type=(ConnectionError,), stop_after_attempt=attempts, wait_exponential_jitter=False)
    return reply.with_fallbacks([RunnableLambda(fallback_reply)], exceptions_to_handle=(ConnectionError,))


def draft_many(chain: Runnable, items: list[dict], *, max_concurrency: int = 2) -> list:
    """Many customers at once. Results are in the same order as `items`. A failure is returned in place, not raised."""
    return chain.batch(items, config={"max_concurrency": max_concurrency}, return_exceptions=True)


def step_names(chain: Runnable) -> list[str]:
    """The pieces of a `a | b | c` chain, in order."""
    return [type(step).__name__ for step in chain.steps]


def demo() -> None:
    model = ScriptedChatModel(script=[say("Thanks for getting in touch. We are looking into your boots.")])
    chain = build_reply_chain(model)
    print("the chain is made of:", step_names(chain))
    reply = chain.invoke({"name": "Sam", "message": "My boots leak. Call me on +1 415 555 0100 or sam@example.com."})
    print("\nwhat the model was sent:")
    for message in model.seen[0]:
        print(f"  {message.type}: {message.content!r}")
    print("\nreply:\n" + reply)

    analysis = build_analysis_chain(
        ScriptedChatModel(script=[say('{"label": "negative", "reason": "the boots leak"}')]),
        ScriptedChatModel(script=[say("The customer's boots leak.")]),
    )
    print("\nanalysis:", analysis.invoke({"name": "Sam", "message": "My boots leak."}))


if __name__ == "__main__":
    demo()
