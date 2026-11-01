"""Scripted stand-in for the Anthropic client.

Parallel agents call the client in an unpredictable order, so replies are NOT popped in one global order. Each call is
routed by its agent role (read from the "[ROLE: x]" tag in the system prompt), and readers also by task_id. Every
queue is thread-safe and consumed in order within its own key.
"""

import json
import re
import threading
import time
from types import SimpleNamespace as NS


def use(id_, tool_name, **inp):
    return NS(type="tool_use", id=id_, name=tool_name, input=inp)


def turn(*blocks, tokens=(100, 50)):
    return NS(content=list(blocks), stop_reason="tool_use", usage=NS(input_tokens=tokens[0], output_tokens=tokens[1]))


def submit(payload: dict, tokens=(100, 50)):
    return turn(use("sub", "submit_result", **payload), tokens=tokens)


def default_key(kw):
    role = re.search(r"\[ROLE: ([^\]]+)\]", kw["system"]).group(1)
    if role == "reader":
        return ("reader", json.loads(kw["messages"][0]["content"])["task_id"])
    return role


class ScriptedClient:
    def __init__(self, queues: dict, key=default_key, delays: dict | None = None):
        self.queues = {k: list(v) for k, v in queues.items()}
        self.key, self.delays = key, delays or {}
        self.calls, self._lock = [], threading.Lock()
        self.messages = NS(create=self._create)

    def _create(self, **kw):
        k = self.key(kw)
        with self._lock:
            self.calls.append(kw | {"messages": kw["messages"][:], "key": k})
            queue = self.queues.get(k)
            if not queue:
                raise AssertionError(f"no scripted reply left for {k!r}")
            reply = queue.pop(0)
        time.sleep(self.delays.get(k[0] if isinstance(k, tuple) else k, 0))
        return reply

    def calls_for(self, role):
        return [c for c in self.calls if (c["key"][0] if isinstance(c["key"], tuple) else c["key"]) == role]


def tool_names(call) -> list:
    return [t["name"] for t in call["tools"]]


def last_tool_results(call) -> list:
    last = call["messages"][-1]["content"]
    return last if isinstance(last, list) else []


# --- ready-made scripts for the demo corpus -----------------------------------------------------
def planner_reply(tasks=None):
    tasks = tasks or [
        {"task_id": 1, "question": "How do they compare on price?", "doc_ids": ["pricing_a", "pricing_b"], "acceptance": "both prices"},
        {"task_id": 2, "question": "How do they compare on security?", "doc_ids": ["security_a", "security_b"], "acceptance": "both certifications"},
    ]
    return submit({"tasks": tasks})


def reader_replies(task_id, doc_ids, quotes, answer="found it"):
    fetches = turn(*[use(f"f{i}", "fetch_document", doc_id=d) for i, d in enumerate(doc_ids)])
    evidence = [{"doc_id": d, "quote": q} for d, q in quotes]
    return [fetches, submit({"task_id": task_id, "status": "ok", "answer": answer, "evidence": evidence})]


PRICE_QUOTES = [("pricing_a", "$12 per user per month"), ("pricing_b", "$9 per user per month")]
SECURITY_QUOTES = [("security_a", "SOC 2 Type II certified"), ("security_b", "SOC 2 Type I certified")]

GOOD_REPORT = submit({"title": "Cloud comparison", "claims": [
    {"text": "Acme costs $12 and Beta $9 per user per month", "sources": [1]},
    {"text": "Acme is SOC 2 Type II and Beta is Type I", "sources": [2]}]})
CHECK_OK = submit({"ok": True, "problems": []})


def happy_queues():
    return {"planner": [planner_reply()],
            ("reader", 1): reader_replies(1, ["pricing_a", "pricing_b"], PRICE_QUOTES, "Acme $12, Beta $9"),
            ("reader", 2): reader_replies(2, ["security_a", "security_b"], SECURITY_QUOTES, "Acme Type II, Beta Type I"),
            "merger": [GOOD_REPORT], "checker": [CHECK_OK]}
