"""Measure YOUR local model's tool calling. Usage:

    python evaluate_local.py --base-url http://localhost:5272/v1 --model qwen2.5-7b-instruct --repeats 10

It builds a small sample project in a temporary folder, asks the model a handful of questions several times each, and prints how often it produced a correct call, with 95% intervals,
plus latency (median, p95, tokens per second when the server reports them). The base URL must be on this machine unless you pass --allow-remote.
I have NOT run this against a real model server: it is tested only against a scripted fake server.
"""

import argparse
import sys
import tempfile
from pathlib import Path

from agentloop import ModelError, OpenAICompatClient
from evalharness import ToolCase, run_tool_eval
from retrieval import BM25Index
from sandbox import Sandbox
from tools import build_tools

DEFAULT_CASES = [ToolCase("What files are in the project?", "list_files", {}), ToolCase("Show me the contents of auth.py.", "read_file", {"path": "auth.py"}),
                 ToolCase("How many functions and TODOs are in auth.py?", "analyze_code", {"path": "auth.py"}), ToolCase("Find all the TODO comments in the project.", "find_todos", {}),
                 ToolCase("According to the documentation, how are passwords handled?", "search_docs", {"query": "*"})]


def sample_project(root: Path) -> None:
    (root / "auth.py").write_text('"""Authentication helpers."""\n\ndef login(user, password):\n    # TODO: hash the password before comparing\n    return user == "admin"\n', encoding="utf8")
    (root / "utils.py").write_text("def clamp(v, lo, hi):\n    return max(lo, min(v, hi))\n", encoding="utf8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Measure a local model's tool calling.")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--allow-remote", action="store_true", help="allow a non-loopback address (your files' names and contents may leave this machine)")
    args = ap.parse_args(argv)
    try:
        client = OpenAICompatClient(args.base_url, timeout_s=args.timeout, require_local=not args.allow_remote)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        sample_project(root)
        index = BM25Index()
        index.add("auth", "The login() function checks credentials. It currently compares passwords in plain text, which should be hashed.")
        tools = build_tools(Sandbox(root), index)
        report = run_tool_eval(client, args.model, tools, DEFAULT_CASES, repeats=args.repeats)
    print(f"attempts: {report.attempts}")
    for outcome, n in report.counts.items():
        if n:
            print(f"  {outcome:<13} {n}")
    for label, (rate, lo, hi) in (("correct call", report.correct), ("valid call (any tool)", report.valid_call)):
        print(f"{label}: {rate:.0%}  (95% interval {lo:.0%} to {hi:.0%})")
    print("latency:", report.latency)
    return 0


if __name__ == "__main__":
    sys.exit(main())
