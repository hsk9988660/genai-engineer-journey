"""The lesson's four tools plus find_todos, wired to the sandbox and the local index."""

from agentloop import ToolDef
from retrieval import BM25Index, search_text
from sandbox import Sandbox, ToolOutput

_NO_ARGS = {"type": "object", "properties": {}}
_PATH = {"type": "object", "properties": {"path": {"type": "string", "maxLength": 300}}, "required": ["path"]}


def build_tools(sandbox: Sandbox, index: BM25Index | None = None) -> dict:
    tools = {
        "list_files": ToolDef("list_files", "List the files in the project folder.", _NO_ARGS, lambda: sandbox.list_files()),
        "read_file": ToolDef("read_file", "Read one text file inside the project folder, e.g. auth.py.", _PATH, lambda path: sandbox.read_text(path)),
        "analyze_code": ToolDef("analyze_code", "Report the line, function and TODO counts of one source file.", _PATH, lambda path: sandbox.analyze(path)),
        "find_todos": ToolDef("find_todos", "Find TODO and FIXME comments across the project, with file and line number.", _NO_ARGS, lambda: sandbox.find_todos()),
    }
    if index is not None:
        tools["search_docs"] = ToolDef("search_docs", "Search the project documentation. Says so if nothing relevant is found.",
                                       {"type": "object", "properties": {"query": {"type": "string", "maxLength": 200}}, "required": ["query"]},
                                       lambda query: ToolOutput(search_text(index, query)))
    return tools
