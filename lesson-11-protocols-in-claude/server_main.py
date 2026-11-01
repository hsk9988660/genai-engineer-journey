"""Run one of the demo servers over stdio:  python server_main.py flights

This is what an MCP host launches as a child process. stdout is the protocol channel: only JSON-RPC lines go there.
"""

import sys

from demo_servers import BUILDERS
from mcp_server import serve_stdio


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in BUILDERS:
        print(f"usage: python server_main.py {{{'|'.join(BUILDERS)}}}", file=sys.stderr)
        return 2
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    if argv[1] == "noisy":
        print("debug: starting up")  # the classic bug: a log line on the protocol channel
    serve_stdio(BUILDERS[argv[1]]())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
