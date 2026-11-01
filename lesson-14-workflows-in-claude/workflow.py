"""A small workflow engine: steps, edges, parallel branches, pause for approval, checkpoint and resume. No framework.

The idea worth keeping from the framework is that control flow becomes DATA: a graph you can check before running, log while running, pause, and resume.
What this adds over the lesson's version, each enforced in code:
  * the graph is validated before it runs (unreachable steps, mixed edges, cycles, model steps that route)
  * a model step cannot choose the route. A conditional edge may only leave a CODE step, which reads checked data
  * every run is bounded: a step cap, a token cap, and a visit cap per step, so a loop cannot run away
  * state is checkpointed after every step and a run resumes from it, with side effects done once (see checkpoint.py)
  * an approval is bound to the exact thing being approved, is single-use, and a stale one is refused
  * parallel branches have timeouts, and the join policy says whether a missing branch is fatal
"""

import concurrent.futures as cf
import json
import threading
from dataclasses import dataclass, field
from typing import Callable

from checkpoint import CheckpointStore, EffectLedger, check_run_id
from middleware import Identity, ToolResult, invoke
from runtrace import Trace, digest


class WorkflowError(Exception):
    pass


class GraphError(WorkflowError):
    pass


class AmbiguousRoute(WorkflowError):
    pass


class NoRoute(WorkflowError):
    pass


class RouteError(WorkflowError):
    pass


class LoopLimit(WorkflowError):
    pass


class LimitExceeded(WorkflowError):
    pass


class StepFailed(WorkflowError):
    def __init__(self, node: str, cause: BaseException):
        super().__init__(f"step {node!r} failed: {type(cause).__name__}: {cause}")
        self.node, self.cause = node, cause


class BranchesFailed(WorkflowError):
    pass


class CheckpointMismatch(WorkflowError):
    pass


class RunExists(WorkflowError):
    pass


class ApprovalError(WorkflowError):
    pass


class StaleApproval(ApprovalError):
    pass


class UnknownApproval(ApprovalError):
    pass


class Paused(Exception):
    """Raised inside a step by `ctx.request_approval`. The engine catches it, saves the run, and returns a paused result."""

    def __init__(self, request_id: str, summary: dict, summary_hash: str):
        super().__init__(request_id)
        self.request_id, self.summary, self.summary_hash = request_id, summary, summary_hash


@dataclass(frozen=True)
class Limits:
    max_steps: int = 50
    max_tokens: int = 100_000


@dataclass(frozen=True)
class Approval:
    decision: str  # "approve" or "reject"
    summary_hash: str  # the hash of what the human was shown
    approver: str = ""


@dataclass
class Node:
    id: str
    kind: str  # "code" | "model" | "parallel"
    fn: Callable | None = None
    side_effect: bool = False
    max_visits: int = 3
    branches: dict | None = None
    join: Callable | None = None
    timeout_s: float = 5.0
    require: object = "all"


@dataclass
class Edge:
    src: str
    dst: str
    when: Callable | None = None
    name: str = ""
    otherwise: bool = False
    trust_model_output: bool = False


@dataclass
class RunResult:
    status: str  # "completed" | "paused"
    output: object
    path: list
    steps: int
    tokens: int
    pending: dict | None = None


class _Shared:
    """State shared by a run's steps and parallel branches."""

    def __init__(self, state, limits, ledger, responses, tools, middleware, trace, identity):
        self.state, self.limits, self.ledger, self.responses = state, limits, ledger, responses
        self.tools, self.middleware, self.trace, self.identity = tools, middleware, trace, identity
        self.lock = threading.Lock()
        self.tokens = state["tokens"]
        self.approvals_used: list[str] = []


class Ctx:
    def __init__(self, shared: _Shared, node: str, visit: int):
        self._s, self.node, self.visit = shared, node, visit
        self.identity, self.run_id = shared.identity, shared.state["run_id"]

    def branch(self, name: str) -> "Ctx":
        return Ctx(self._s, f"{self.node}.{name}", self.visit)

    @property
    def effect_key(self) -> str:
        return f"{self.run_id}:{self.node}:{self.visit}"

    def charge(self, tokens: int) -> None:
        with self._s.lock:
            self._s.tokens += int(tokens)
            if self._s.tokens > self._s.limits.max_tokens:
                raise LimitExceeded(f"token cap {self._s.limits.max_tokens} exceeded")

    def once(self, fn):
        """Do a side effect at most once per (run, step, visit), even if the step is re-run after a crash. `fn` receives the key."""
        if self._s.ledger is None:
            raise GraphError("this run has no EffectLedger, so side effects cannot be made safe to repeat")
        return self._s.ledger.once(self.effect_key, fn)

    def call_tool(self, name: str, **args) -> ToolResult:
        return invoke(self._s.tools, self._s.middleware, name, args, self.identity)

    def request_approval(self, summary: dict) -> str:
        """Pause the run until a human approves exactly `summary`. Returns "approve" or "reject". The step re-runs from its start on resume,
        so everything before this call must be free of side effects."""
        rid = f"{self.node}:{self.visit}"
        h = digest({"run": self.run_id, "request": rid, "summary": summary})  # bound to THIS run and step, so it cannot be replayed elsewhere
        resp = self._s.responses.get(rid)
        if resp is None:
            raise Paused(rid, summary, h)
        if rid in self._s.state["approvals_used"]:
            raise ApprovalError("this approval was already used")
        if resp.decision not in ("approve", "reject"):
            raise ApprovalError("decision must be 'approve' or 'reject'")
        if resp.summary_hash != h:
            raise StaleApproval("what was approved is not what is being asked now")
        with self._s.lock:
            self._s.approvals_used.append(rid)
        self._s.trace.emit("approval_used", request=rid, decision=resp.decision, approver=resp.approver)
        return resp.decision


class Workflow:
    def __init__(self, name: str = "workflow"):
        self.name, self.nodes, self.edges, self.start = name, {}, [], None

    # --- building -------------------------------------------------------------------------------------------------------------
    def _add(self, node: Node) -> "Workflow":
        if not node.id or node.id in self.nodes:
            raise GraphError(f"step id {node.id!r} is empty or already used")
        self.nodes[node.id] = node
        if self.start is None:
            self.start = node.id
        return self

    def step(self, id: str, fn: Callable, kind: str = "code", side_effect: bool = False, max_visits: int = 3) -> "Workflow":
        if kind not in ("code", "model"):
            raise GraphError("kind must be 'code' or 'model'")
        return self._add(Node(id, kind, fn, side_effect, max_visits))

    def parallel(self, id: str, branches: dict, join: Callable, timeout_s: float = 5.0, require="all", side_effect: bool = False) -> "Workflow":
        if not branches:
            raise GraphError("a parallel step needs at least one branch")
        return self._add(Node(id, "parallel", None, side_effect, 3, dict(branches), join, timeout_s, require))

    def edge(self, src: str, dst: str, when: Callable | None = None, name: str = "", trust_model_output: bool = False) -> "Workflow":
        self.edges.append(Edge(src, dst, when, name or (getattr(when, "__name__", "") if when else ""), False, trust_model_output))
        return self

    def otherwise(self, src: str, dst: str) -> "Workflow":
        self.edges.append(Edge(src, dst, None, "otherwise", True))
        return self

    def start_at(self, id: str) -> "Workflow":
        self.start = id
        return self

    def outgoing(self, id: str) -> list[Edge]:
        return [e for e in self.edges if e.src == id]

    def validate(self, allow_cycles: bool = False) -> None:
        if self.start is None or self.start not in self.nodes:
            raise GraphError("the workflow has no valid start step")
        for e in self.edges:
            if e.src not in self.nodes or e.dst not in self.nodes:
                raise GraphError(f"edge {e.src!r} -> {e.dst!r} refers to a step that does not exist")
        for nid, node in self.nodes.items():
            out = self.outgoing(nid)
            direct = [e for e in out if e.when is None and not e.otherwise]
            conditional = [e for e in out if e.when is not None]
            if len(direct) > 1 or (direct and (conditional or any(e.otherwise for e in out))):
                raise GraphError(f"step {nid!r} mixes direct edges with other edges, or has several direct edges: use a parallel step or conditions")
            if sum(1 for e in out if e.otherwise) > 1:
                raise GraphError(f"step {nid!r} has more than one 'otherwise' edge")
            if node.kind == "model":
                for e in conditional:
                    if not e.trust_model_output:
                        raise GraphError(f"step {nid!r} is a model step and cannot choose the route: put a code step that checks its output in between")
        seen, stack = set(), [self.start]
        while stack:
            cur = stack.pop()
            if cur not in seen:
                seen.add(cur)
                stack.extend(e.dst for e in self.outgoing(cur))
        missing = sorted(set(self.nodes) - seen)
        if missing:
            raise GraphError(f"steps that can never run: {missing}")
        if not allow_cycles and self._has_cycle():
            raise GraphError("the graph has a cycle; build with allow_cycles=True if a loop is intended (each step is still capped by max_visits)")

    def _has_cycle(self) -> bool:
        state: dict[str, int] = {}

        def visit(n) -> bool:
            state[n] = 1
            for e in self.outgoing(n):
                if state.get(e.dst) == 1 or (e.dst not in state and visit(e.dst)):
                    return True
            state[n] = 2
            return False
        return any(n not in state and visit(n) for n in self.nodes)

    def fingerprint(self) -> str:
        return digest({"start": self.start, "nodes": [(n.id, n.kind, n.side_effect, sorted(n.branches or {})) for n in self.nodes.values()],
                       "edges": [(e.src, e.dst, e.name, e.otherwise) for e in self.edges]})

    # --- running --------------------------------------------------------------------------------------------------------------
    def _route(self, node: Node, out) -> str | None:
        edges = self.outgoing(node.id)
        if not edges:
            return None
        if len(edges) == 1 and edges[0].when is None and not edges[0].otherwise:
            return edges[0].dst
        matched = []
        for e in edges:
            if e.when is None:
                continue
            try:
                ok = bool(e.when(out))
            except Exception as exc:
                raise RouteError(f"condition {e.name!r} on step {node.id!r} raised {type(exc).__name__}: {exc}") from exc
            if ok:
                matched.append(e)
        if len(matched) > 1:
            raise AmbiguousRoute(f"step {node.id!r}: conditions {[e.name for e in matched]} all matched")
        if matched:
            return matched[0].dst
        fallback = next((e for e in edges if e.otherwise), None)
        if fallback is None:
            raise NoRoute(f"step {node.id!r}: no condition matched and there is no 'otherwise' edge")
        return fallback.dst

    def _run_parallel(self, node: Node, payload, ctx: Ctx):
        ex = cf.ThreadPoolExecutor(max_workers=len(node.branches))
        futures = {name: ex.submit(fn, payload, ctx.branch(name)) for name, fn in node.branches.items()}
        cf.wait(list(futures.values()), timeout=node.timeout_s)
        ok, missing = {}, {}
        for name, f in futures.items():
            if not f.done():
                f.cancel()
                missing[name] = "timeout"
            elif f.exception() is not None:
                missing[name] = f"{type(f.exception()).__name__}: {f.exception()}"
            else:
                ok[name] = f.result()
        ex.shutdown(wait=False, cancel_futures=True)
        needed = len(node.branches) if node.require == "all" else 1 if node.require == "any" else int(node.require)
        if len(ok) < needed:
            raise BranchesFailed(f"step {node.id!r}: {len(ok)} of {len(node.branches)} branches finished, {needed} needed; missing: {missing}")
        return node.join(ok, missing)

    def run(self, payload, run_id: str, identity: Identity, *, checkpoints: CheckpointStore | None = None, ledger: EffectLedger | None = None,
            responses: dict | None = None, limits: Limits | None = None, trace: Trace | None = None, tools: dict | None = None,
            middleware=(), resume: bool = False, allow_cycles: bool = False) -> RunResult:
        self.validate(allow_cycles)
        check_run_id(run_id)
        limits, trace, responses = limits or Limits(), trace or Trace(), dict(responses or {})
        if any(n.side_effect for n in self.nodes.values()) and ledger is None:
            raise GraphError("a step is marked side_effect but no EffectLedger was given")
        state = checkpoints.load(run_id) if checkpoints else None
        if state is not None:
            if not resume:
                raise RunExists(f"run {run_id!r} already has a checkpoint; pass resume=True to continue it")
            if state["fingerprint"] != self.fingerprint():
                raise CheckpointMismatch("the workflow changed since this run was checkpointed")
            if state["status"] == "completed":
                return RunResult("completed", state["payload"], state["path"], state["steps"], state["tokens"])
            trace.emit("run_resumed", run=run_id, at=state["next"])
        else:
            json.dumps(payload)
            state = {"run_id": run_id, "fingerprint": self.fingerprint(), "status": "running", "next": self.start, "payload": payload, "visits": {},
                     "path": [], "tokens": 0, "steps": 0, "pending": None, "approvals_used": []}
            trace.emit("run_started", run=run_id, workflow=self.name)
        pending = state["pending"]
        if responses and (pending is None or set(responses) - {pending["id"]}):
            raise UnknownApproval(f"responses {sorted(responses)} do not match the pending request {pending and pending['id']!r}")
        shared = _Shared(state, limits, ledger, responses, tools or {}, list(middleware), trace, identity)

        def save(status: str) -> None:
            state["status"], state["tokens"] = status, shared.tokens
            if checkpoints:
                checkpoints.save(state)

        while state["next"] is not None:
            node = self.nodes[state["next"]]
            if state["steps"] >= limits.max_steps:
                save("failed")
                raise LimitExceeded(f"step cap {limits.max_steps} reached")
            visit = state["visits"].get(node.id, 0) + 1  # only successful steps are counted, so a retry keeps the same visit number and effect key
            if visit > node.max_visits:
                save("failed")
                raise LoopLimit(f"step {node.id!r} would run {visit} times; max_visits is {node.max_visits}")
            ctx = Ctx(shared, node.id, visit)
            trace.emit("step_started", node=node.id, visit=visit)
            try:
                out = self._run_parallel(node, state["payload"], ctx) if node.kind == "parallel" else node.fn(state["payload"], ctx)
                json.dumps(out)
                nxt = self._route(node, out)
            except Paused as p:
                state["pending"] = {"id": p.request_id, "node": node.id, "summary": p.summary, "summary_hash": p.summary_hash}
                trace.emit("paused", node=node.id, request=p.request_id, summary_hash=p.summary_hash)
                save("paused")
                return RunResult("paused", None, list(state["path"]), state["steps"], shared.tokens, state["pending"])
            except WorkflowError as exc:
                trace.emit("step_failed", node=node.id, error=type(exc).__name__)
                save("failed")
                raise
            except Exception as exc:
                trace.emit("step_failed", node=node.id, error=type(exc).__name__)
                save("failed")
                raise StepFailed(node.id, exc) from exc
            state["payload"], state["next"], state["pending"] = out, nxt, None
            state["visits"][node.id], state["steps"] = visit, state["steps"] + 1
            state["path"].append(node.id)
            state["approvals_used"] = state["approvals_used"] + shared.approvals_used
            shared.approvals_used = []
            trace.emit("step_completed", node=node.id, output_hash=digest(out)[:16], next=nxt)
            save("running" if nxt else "completed")
        trace.emit("run_completed", run=run_id, steps=state["steps"])
        return RunResult("completed", state["payload"], state["path"], state["steps"], shared.tokens)
