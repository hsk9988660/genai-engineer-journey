"""A context manager that ties the pieces together for one agent session.

The model never manages its own memory here. After each turn the HARNESS decides whether to compact, what to keep, what to offload,
and what the next call receives:

  transcript   append-only archive of every turn (never edited)
  facts        versioned, trusted facts (these survive compaction untouched)
  compaction   when the un-compacted history grows past a threshold, older turns are summarized, but the summary only replaces them
               if it passes verification; otherwise the raw turns stay
  tool results big ones are offloaded behind a handle, old ones are stubbed
  assembly     every model call is built by the assembler, which also returns an inspection record
"""

from assembler import Assembled, Budget, ContextAssembler, Item
from compactor import Compactor, Summary, Transcript, Turn
from facts import FactStore
from tokens import estimate_tokens
from tool_results import ToolItem, ToolResultStore


class ContextManager:
    def __init__(self, system: str, summarize_fn, tools=(), budget: Budget | None = None, keep_recent: int = 6,
                 compact_over_tokens: int = 2000, store: FactStore | None = None, count=estimate_tokens,
                 compaction_cooldown_turns: int = 8):
        self.system, self.tools, self.keep_recent, self.compact_over, self.count = system, list(tools), keep_recent, compact_over_tokens, count
        self.transcript = Transcript()
        self.facts = store or FactStore()
        self.compactor = Compactor(summarize_fn, self.facts, count)
        self.results = ToolResultStore(count=count)
        self.assembler = ContextAssembler(budget, count)
        self.summary: Summary | None = None
        self.compacted_through = 0  # id of the last turn covered by the summary
        self.tool_items: list[ToolItem] = []
        self.records: list = []
        self.compaction_log: list[dict] = []
        self.cooldown, self._retry_after = compaction_cooldown_turns, 0

    # --- recording -------------------------------------------------------------------------------------------
    def add_user(self, text: str) -> Turn:
        return self.transcript.append("user", text)

    def add_assistant(self, text: str) -> Turn:
        return self.transcript.append("assistant", text)

    def add_tool_result(self, tool: str, text: str) -> ToolItem:
        item = self.results.add(tool, text)
        self.tool_items.append(item)
        return item

    # --- compaction ----------------------------------------------------------------------------------------------
    def live_history(self) -> list[Turn]:
        return [t for t in self.transcript.all() if t.id > self.compacted_through]

    def maybe_compact(self) -> str:
        """Returns what happened: skipped | cooling_down | compacted | kept_raw.

        After a FAILED compaction the manager waits `cooldown` turns before trying again. Without this, a summarizer that keeps
        failing verification would be called again (and again, with a retry) on EVERY turn: a silent cost leak.
        """
        latest = self.transcript.all()[-1].id if self.transcript.all() else 0
        if latest < self._retry_after:
            return "cooling_down"
        live = self.live_history()
        if len(live) <= self.keep_recent or sum(self.count(t.text) for t in live) <= self.compact_over:
            return "skipped"
        older = live[: len(live) - self.keep_recent]
        # The previous summary is itself source material, so numbers it carries are not flagged as invented.
        sources = ([Turn(0, "assistant", self.summary.text)] if self.summary else []) + older
        result = self.compactor.compact(sources)
        self.compaction_log.append({"status": result.status, "turns": len(older), "problems": result.problems})
        if result.status != "compacted":
            self._retry_after = latest + self.cooldown
            return "kept_raw"
        self.summary = Summary(result.summary.text, (self.summary.source_ids if self.summary else ()) + tuple(t.id for t in older),
                               result.summary.tokens_before, result.summary.tokens_after)
        self.compacted_through = older[-1].id
        return "compacted"

    # --- building the next call ----------------------------------------------------------------------------------------
    def build(self, current: str, retrieved=(), volatile: str = "", call_id: str = "") -> Assembled:
        """`current` is the new user message. It is NOT added to the transcript here, so call add_user(current) afterwards."""
        self.maybe_compact()
        assembled = self.assembler.assemble(system=self.system, tools=self.tools, facts=self.facts, current=current, summary=self.summary,
                                            retrieved=list(retrieved), history=self.live_history(), tool_results=self.tool_items,
                                            volatile=volatile, call_id=call_id or f"call{len(self.records) + 1}")
        self.records.append(assembled.record)
        return assembled
