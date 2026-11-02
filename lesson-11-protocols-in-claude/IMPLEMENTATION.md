# Lesson 11: implementation guide

This explains **how the code works**: how an agent talks to tool servers (MCP) and to other agents (an A2A-style protocol) over a real wire, and how a gateway decides which of that remote text and code to trust. The [README](README.md) says why the lesson exists, lists the attack classes and reports the bugs found while building it.

The one idea to keep in mind: **a protocol makes a server connectable, never trustworthy.** Everything in `gateway.py` exists because of that sentence.

---

## 1. The system on one page

```
 your agent (agent_loop.py)                                   third-party servers
 ┌───────────────────────────┐   exposed_tools()      ┌────────────────────────────────────────┐
 │ model ◄── tool list,      │◄───────────────────────│ MCPGateway  (gateway.py)  TRUST LAYER  │
 │  recomputed EVERY step    │                        │  allowlist · pinned hashes · re-list   │
 │ model ──► server__tool ───┼── gateway.call() ─────►│  namespacing · side-effect approval    │
 │ result wrapped as DATA ◄──┼────────────────────────│  taint · size / count / time limits    │
 └───────────────────────────┘                        └───────────────┬────────────────────────┘
                                                                      │ MCPClient (mcp_client.py)
                                                                      │ JSON-RPC 2.0, one ASCII line per message
                                                                      ▼   over a child process's stdin/stdout
                                                       MCPServer (mcp_server.py) or a hostile one (demo_servers.py)

 separate: A2AClient ──HTTP + bearer token──► A2AServer   (a2a_lite.py: tasks with a lifecycle)
```

Layers, bottom up: `jsonrpc.py` (message format) → `mcp_server.py` / `mcp_client.py` (the protocol) → `gateway.py` (policy) → `agent_loop.py` (the model).

---

## 2. One tool call, from start to end

Follow `python agent_loop.py "Find me a flight from PDX to HNL"` (needs an API key).

### Connecting (`MCPGateway.add_server`)
1. The server name must match a safe pattern and be unique. The gateway starts the server as a child process (`StdioTransport`) and runs the **handshake**: `initialize` (client and server agree a protocol version; the client checks the answer), then the `notifications/initialized` notification. A separate, longer `connect_timeout_s` applies here because process start-up is much slower than a normal call (a flaky-test bug found while building).
2. `refresh` runs `tools/list` and classifies **every** listed tool into one of three buckets, in this order:
   - invalid or duplicate name, or beyond `max_tools_per_server`: **blocked**
   - not on your **allowlist** (`allow_tools=None` means *no tools at all*): **blocked**
   - otherwise compute `definition_hash(name, description, inputSchema)` (SHA-256 of canonical JSON) and compare with the **pin** a human approved:
     - equal: **exposed**
     - a pin exists but differs: **blocked** as a *rug pull*, with an event logged
     - no pin yet: **pending**, with the description scanned for suspicious wording
3. A human reviews `pending_approval()` (full text, hash, findings, and what the server *claims* in its annotations) and calls `pin(server, tool, digest)`. `pin` **fails if the definition changed since the reviewer looked**, so you approve exactly what you read.

### The loop (`run_agent`)
Each step:
1. `tools = gateway.exposed_tools() + [submit_answer]`. The list is **recomputed every step**, so a tool that was withdrawn or changed disappears from the very next model call. This is what "dynamic discovery" means here. Names are shown as `server__tool` and descriptions are prefixed with the server name.
2. The model is called with `tool_choice="any"`; it must call something.
3. `submit_answer` ends the run (`answered`). Otherwise each tool call goes through `gateway.call(name, args, run.state)`.
4. Each result goes back through `wrap`: successful server output is enclosed in `<untrusted_tool_output server="...">`; anything else becomes a small JSON status written by the gateway, never containing server text. The system prompt says that text in those tags is data, not instructions. After `max_steps` the run ends `capped`.

### `MCPGateway.call`: what happens in order
| Step | Check | Result if it fails |
|---|---|---|
| 1 | the server and tool name exist | `unknown_tool` |
| 2 | the server is healthy | `unavailable` |
| 3 | `recheck="always"` re-lists the tools first (or `on_notification` if the server announced a change) | `unavailable` if the server is now gone |
| 4 | the tool is still **exposed** (still pinned, still same hash) | `unavailable`, with the reason (rug pull, not allowlisted, awaiting approval) |
| 5 | arguments validate against the tool's schema **locally** (`schema.py`) | `invalid_arguments` |
| 6 | a tool **you** declared as side-effecting: refused if the run is **tainted**, else the approver must answer exactly `"approve"` (exception = deny) | `blocked_taint` / `denied` |
| 7 | the call itself, with a timeout | `timeout`, `server_closed` or `protocol_error`, and **the server is dropped** |
| 8 | result longer than `max_result_chars` is truncated with a note | `truncated=True` |
| 9 | if the tool is in `untrusted_output_tools`, the run becomes **tainted** | n/a |

Side effects and untrusted output are declared in **your** `ServerConfig`. A server's own annotations (a `readOnlyHint` that says "read-only" on a booking tool, for instance) are recorded for display and **ignored for policy**: the party being controlled cannot describe itself to the control.

### Where it ends
`AgentRun` with `status` (`answered` or `capped`), the answer, the step count, every `(tool, gateway status)` pair, and the list of tools the model was offered at each step. The gateway keeps an `events` log (`connected`, `pending_approval`, `pinned`, `rug_pull`, `blocked_taint`, `denied`, `truncated`, `timeout`, ...).

---

## 3. Module reference

### `jsonrpc.py`
JSON-RPC 2.0 messages: **request** (has `id` and `method`), **notification** (no `id`, never answered), **response** (`result` or `error`). `encode` always yields **one ASCII line** with no raw newline (it escapes them and non-ASCII characters), because stdio framing is "one message per line" and a split line corrupts the stream. `decode` parses and structurally validates, raising `JsonRpcError` with the standard codes. Batches are not supported.

### `schema.py`
A small JSON Schema validator: `type`, `properties`, `required`, `enum`, `items`, `additionalProperties: false`, `minimum`/`maximum`, `minLength`/`maxLength`. **Unsupported keywords are ignored, not enforced.** `validate` returns readable problems; an empty list means valid.

### `mcp_server.py`
A single-threaded MCP server. Implemented: `initialize` with version negotiation, `notifications/initialized`, `ping`, `tools/list` with cursor pagination, `tools/call`, `resources/list`, `resources/read`, `notifications/tools/list_changed`. Not implemented: prompts, sampling, roots, elicitation, subscriptions, HTTP transports, authorization, progress, cancellation.
Two failure kinds that people mix up: a **protocol error** is a JSON-RPC `error` object (unknown method or tool, bad parameters); a **tool error** is a normal result with `isError: true`, so the model can read the message and recover. Here an unknown tool is a protocol error and invalid arguments are a tool error. `serve_stdio` reads lines from stdin and writes responses to stdout; **stdout is the protocol channel, so logging must go to stderr**. One stray `print` corrupts the stream.

### `mcp_client.py`
- Transports: `StdioTransport` (a real subprocess; threads read stdout and drain stderr) and `LoopbackTransport` (calls an in-process server, but messages still pass through the real encoder and decoder).
- `MCPClient` does the handshake and **matches responses to requests by id**, handling notifications that arrive in between. Request ids only increase, so a late reply to a request that timed out can never match a newer one and is ignored.
- On a timeout it sends `notifications/cancelled` and raises `ToolTimeout`. A server that dies mid-call raises `ServerClosed` instead of blocking. A line that is not valid JSON-RPC is a `ProtocolError` naming the offending text. A message over `max_message_chars` raises `MessageTooLarge`, so a hostile server cannot flood memory.
- Not implemented: HTTP transports, authorization, server-initiated requests, progress.

### `gateway.py`
Described above. Also: `scan_description` is a **heuristic aid for the human reviewer** (it looks for phrases such as "ignore previous instructions", an `<IMPORTANT>` block, "do not tell the user", references to credentials or `.ssh`, "before using this tool", rules for other tools, covert recipients, instructions to send data to a URL, the system prompt) and it also scans every description string **inside the schema**, because poison can hide in a parameter description. It is easy to evade, which is why the real control is the pin, not the scan.

### `demo_servers.py` and `server_main.py`
One honest server (`flights`, whose booking tool carries a *false* `readOnlyHint`) and hostile or broken ones, each modelling one attack class: `poisoned`, `rugpull`, `shadowing`, `collision`, `injection`, `flood`, `many`, `hang`, `crash`, `noisy`. They touch no network or files; the instructions inside are inert text. `python server_main.py flights` runs one over stdio.

### `a2a_lite.py`: delegating to another agent
**An A2A-inspired protocol, not the real A2A**: it uses its own method names (`task.create`, `task.get`, ...) because exact conformance to the real spec could not be verified. Use the official SDK for interoperability.

What differs from a tool call is the **task lifecycle**: `submitted → working → (input_required ↔ working) → completed | failed | canceled`, where the terminal states are final (`TaskStore.transition` enforces the allowed moves). The work can be long, ask a question midway (`input_required`, answered with `reply`) and be cancelled.

Server (`A2AServer`, a real HTTP server on localhost):
- a public **agent card** (name, description, url, skills) for discovery
- `POST /rpc` needs a **bearer token**; `_authenticate` compares against every token with `hmac.compare_digest`, with no early exit
- a client can only see **its own** tasks; there is a per-client cap on active tasks, a request-body cap (413 beyond it), and a text length cap
- a result that arrives after the task was cancelled is discarded
- on a rejection it reads and discards a bounded amount of the body before replying (a bug found while building: replying without reading let the OS reset the connection and lose the rejection)

Client (`A2AClient`): `discover` fetches the card and **refuses a card whose `url` is on a different origin** than the one you connected to (`UntrustedCard`), so a card cannot redirect your credentials elsewhere. Redirects are not followed (`_NoRedirect`). `create`, `get`, `reply`, `cancel`, and `wait` (polls until the task finishes or needs input). `render_for_model(task)` wraps the remote agent's output in `<untrusted_remote_agent_output>`, the same rule as for tool results.

---

## 4. Design decisions

- **Default deny.** No allowlist means no tools. Nothing is exposed until a human pinned its exact definition.
- **Pin a hash, not a judgement.** A new or changed definition is quarantined, and the reviewer sees the full text. Scanning is only a hint.
- **Re-verify before each call.** A hostile server will not announce that it changed, so the gateway looks again (`recheck="always"` is the default; `on_notification` and `never` trade safety for one fewer round trip).
- **Policy from your configuration,** never from the server's self-description.
- **Taint.** After a tool that can return attacker text has run, side-effect tools are refused, whatever a human would approve.
- **Namespacing.** The model only ever sees `server__tool`, so one server cannot impersonate another.
- **Drop, do not wait.** A hung, crashed or protocol-breaking server is disconnected and its tools disappear.
- **Everything remote is data.** Wrapped in tags, truncated, and never allowed to speak in the gateway's voice.

## 5. What it does not do

The pin and taint logic stops what it was built to stop, but a pinned tool can still be dishonest in what it *returns*; a tainted run only refuses side effects, it does not make the text safe. The description scan is heuristic. The MCP server and client are partial (see above) and not for production; use the official SDK. The A2A part is not wire-compatible with the real spec. The model-facing behaviour (does a model actually resist injected tool output?) is not measured here.

## 6. Running and testing

`python -m pytest -q` runs 135 offline tests (about 20 seconds, because several start real child processes and a real localhost HTTP server; no API key needed). `python server_main.py <name>` runs a demo server. `python agent_loop.py "..."` needs an API key.
