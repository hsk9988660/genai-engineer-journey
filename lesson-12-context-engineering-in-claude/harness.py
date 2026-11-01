"""Compare context strategies on a long synthetic conversation: do the planted facts survive, and what does each strategy cost?

Planted early in the conversation: a budget ($3000), a peanut allergy, and travel dates (April). Later the user CHANGES the dates to
October. After 40 turns, for each strategy we look at the context the next model call would receive and ask:
  retained       is each CURRENT fact (3000, peanuts, October) present in it?
  stale leaked   does the superseded value (April) still appear? (that is the "context clash")
  tokens         how big is that context, and how many tokens were sent in total across the whole conversation?

What this does and does not tell you. The summarizers below are TEST DOUBLES with known behavior (one drops digits, one invents a
number). They show what the MECHANISMS do: pinned facts survive a lossy summary, verification refuses a bad one, truncation loses
early facts, and full history leaks the stale value. They say NOTHING about how well a real model summarizes. Also, a fact being
present in the context is necessary but not sufficient: the model still has to use it. `python harness.py live` asks a real model
(needs ANTHROPIC_API_KEY) and has not been run.

Run:  python harness.py
"""

import re
import sys
from dataclasses import dataclass

from assembler import Budget
from compactor import Turn, numbers_in
from session import ContextManager
from tokens import estimate_tokens

SYSTEM = "You are a travel planning assistant. Use only the facts you are given."
TOPICS = ["street food", "museums", "train travel", "hiking trails", "local festivals", "photography spots", "ferry routes",
          "old towns", "hot springs", "night markets"]
NEEDLES = {"budget": "3000", "allergy": "peanuts", "dates": "October"}  # the CURRENT values at the end
STALE = ["April"]  # superseded by October


def make_conversation(n_turns: int = 40) -> list[tuple[str, str]]:
    """(role, text) pairs. User filler contains no digits; assistant filler contains numbers like a real reply would."""
    change_at = int(0.6 * n_turns)
    out = []
    for i in range(1, n_turns + 1):
        planted = {2: "My budget is $3000 for the whole trip.", 3: "I am allergic to peanuts, so please remember that.",
                   4: "I would like to travel in April.", change_at: "Actually, let's travel in October instead."}
        user = planted.get(i, f"Tell me something interesting about {TOPICS[i % len(TOPICS)]}.")
        reply = (f"Here is a thought about {TOPICS[i % len(TOPICS)]}: route {i} takes about {i + 2} hours, and many travellers "
                 f"enjoy stopping along the way to look around, taste something local, and take their time before moving on to "
                 f"the next place on the list, which keeps the whole journey relaxed and flexible for everyone involved.")
        out += [("user", user), ("assistant", reply)]
    return out


def extract_planted(text: str) -> list[tuple[str, str]]:
    """A rule-based stand-in for the live LLM fact extractor, which is exercised separately in the tests."""
    found = []
    if m := re.search(r"\$([\d,]+)", text):
        found.append(("budget", m.group(1).replace(",", "")))
    if m := re.search(r"allergic to (\w+)", text):
        found.append(("allergy", m.group(1)))
    if m := re.search(r"travel in (\w+)", text):
        found.append(("dates", m.group(1)))
    return found


# --- summarizer test doubles ------------------------------------------------------------------------------------------------
def lossy_summarizer(turns, feedback):
    """Keeps the first 8 words of each turn and DROPS every digit: a stand-in for a model that forgets numbers."""
    return "Earlier: " + "; ".join(re.sub(r"\d+", "", " ".join(t.text.split()[:8])).strip() for t in turns)


def feedback_aware_summarizer(turns, feedback):
    """Like the lossy one, but when told a value is stale it leaves out any turn that mentions it."""
    stale = [v for p in (feedback or []) if "no longer" in p for v in re.findall(r"'([^']+)'", p)]
    kept = [t for t in turns if not any(re.search(rf"\b{re.escape(v)}\b", t.text, re.I) for v in stale)]
    return lossy_summarizer(kept, None)


def hallucinating_summarizer(turns, feedback):
    """Invents a number that is in no turn."""
    return lossy_summarizer(turns, None) + " The user's budget is $4500."


# --- strategies -------------------------------------------------------------------------------------------------------------------
class FullHistory:
    name = "full_history"

    def __init__(self):
        self.turns = []

    def context(self, current: str) -> str:
        return "\n".join([SYSTEM] + [t.text for t in self.turns] + [current])

    def record(self, role: str, text: str):
        self.turns.append(Turn(len(self.turns) + 1, role, text))


class TruncateLastK(FullHistory):
    def __init__(self, k: int = 10):
        super().__init__()
        self.k, self.name = k, f"truncate_last_{k}"

    def context(self, current: str) -> str:
        return "\n".join([SYSTEM] + [t.text for t in self.turns[-self.k:]] + [current])


class NaiveSummary(FullHistory):
    """Summarize older turns with NO verification and NO fact store."""
    name = "naive_summary"

    def __init__(self, summarizer=lossy_summarizer, keep_recent: int = 6, over_tokens: int = 1500):
        super().__init__()
        self.summarizer, self.keep_recent, self.over, self.summary, self.through = summarizer, keep_recent, over_tokens, "", 0

    def context(self, current: str) -> str:
        live = [t for t in self.turns if t.id > self.through]
        if len(live) > self.keep_recent and sum(estimate_tokens(t.text) for t in live) > self.over:
            older = live[: -self.keep_recent]
            self.summary = self.summarizer(([Turn(0, "assistant", self.summary)] if self.summary else []) + older, None)
            self.through = older[-1].id
            live = live[-self.keep_recent:]
        return "\n".join([SYSTEM] + ([self.summary] if self.summary else []) + [t.text for t in live] + [current])


class Managed:
    """Fact store + verified compaction + the assembler."""

    def __init__(self, summarizer, label: str):
        self.name = label
        self.cm = ContextManager(SYSTEM, summarizer, budget=Budget(), keep_recent=6, compact_over_tokens=1500)

    def context(self, current: str) -> str:
        built = self.cm.build(current)
        self.last_record = built.record
        return built.system + "\n" + "\n".join(m["content"] for m in built.messages)

    def record(self, role: str, text: str):
        (self.cm.add_user if role == "user" else self.cm.add_assistant)(text)
        if role == "user":
            turn = self.cm.transcript.all()[-1]
            for key, value in extract_planted(text):
                self.cm.facts.set(key, value, turn.id, "user")


@dataclass
class Outcome:
    name: str
    retained: dict
    stale_leaked: bool
    final_tokens: int
    cumulative_tokens: int
    note: str = ""


def run_strategy(strategy, conversation) -> Outcome:
    """Feed the conversation in. Before each user message, measure the context that call would carry."""
    cumulative, last = 0, ""
    for role, text in conversation:
        if role == "user":
            last = strategy.context(text)  # this is what the model would see for this user message
            cumulative += estimate_tokens(last)
        strategy.record(role, text)
    final = strategy.context("What do you know about my trip so far?")
    flat = final.replace(",", "")
    retained = {k: bool(re.search(rf"\b{re.escape(v)}\b", flat, re.I)) for k, v in NEEDLES.items()}
    leaked = any(re.search(rf"\b{s}\b", flat, re.I) for s in STALE)
    note = ""
    if isinstance(strategy, Managed):
        log = strategy.cm.compaction_log
        note = f"compactions: {sum(l['status'] == 'compacted' for l in log)} ok, {sum(l['status'] == 'kept_raw' for l in log)} kept raw"
    return Outcome(strategy.name, retained, leaked, estimate_tokens(final), cumulative, note)


def evaluate(n_turns: int = 40) -> list[Outcome]:
    convo = make_conversation(n_turns)
    strategies = [FullHistory(), TruncateLastK(10), NaiveSummary(), Managed(feedback_aware_summarizer, "managed (lossy summarizer)"),
                  Managed(hallucinating_summarizer, "managed (hallucinating summarizer)")]
    return [run_strategy(s, convo) for s in strategies]


def cumulative_tokens(strategy_factory, n_turns: int) -> int:
    return run_strategy(strategy_factory(), make_conversation(n_turns)).cumulative_tokens


def print_table(outcomes: list[Outcome]) -> None:
    print(f"{'strategy':36s} {'budget':>6s} {'allergy':>7s} {'dates':>6s} {'stale April':>11s} {'final tok':>9s} {'total sent':>10s}")
    for o in outcomes:
        mark = lambda b: "kept" if b else "LOST"
        print(f"{o.name:36s} {mark(o.retained['budget']):>6s} {mark(o.retained['allergy']):>7s} {mark(o.retained['dates']):>6s} "
              f"{'LEAKED' if o.stale_leaked else 'clean':>11s} {o.final_tokens:>9d} {o.cumulative_tokens:>10d}  {o.note}")


def live_needle_eval(client, n_turns: int = 40, model: str = "claude-sonnet-5-5") -> None:
    """Ask a real model about the planted facts using each strategy's context. NOT RUN: needs an API key and costs money."""
    convo = make_conversation(n_turns)
    question = "What is my budget, what am I allergic to, and when am I travelling? Answer in one short sentence."
    for strategy in (FullHistory(), TruncateLastK(10), NaiveSummary(), Managed(feedback_aware_summarizer, "managed")):
        for role, text in convo:
            strategy.record(role, text)
        answer = client.messages.create(model=model, max_tokens=120, messages=[{"role": "user", "content": strategy.context(question)}])
        print(f"{strategy.name:28s} -> {answer.content[0].text}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "live":
        import anthropic
        live_needle_eval(anthropic.Anthropic())
    else:
        print_table(evaluate())
