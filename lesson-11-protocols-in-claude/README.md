# Lesson 11: agent protocols (MCP, A2A), built from the wire up, with the security the lesson skipped

Lesson 11 of [ai-agents-for-beginners](https://github.com/microsoft/ai-agents-for-beginners) explains MCP, A2A and NLWeb, but its notebooks contain no MCP and no A2A: the MCP notebook defines two ordinary `@tool` functions and says they *simulate* MCP, and the A2A notebook is three prompts in a row inside one process. Its only security claim is "standardized authentication". This folder implements the real mechanics and the part that matters most: **a protocol makes a server connectable, never trustworthy.**

## Run

```bash
pip install anthropic pydantic pytest
python -m pytest -q                      # 134 offline tests (about 20 seconds: several launch real subprocesses and a real localhost HTTP server)
python server_main.py flights            # run a demo MCP server over stdio (it waits for JSON-RPC lines on stdin)
set ANTHROPIC_API_KEY=sk-ant-...         # macOS/Linux: export ANTHROPIC_API_KEY=...
python agent_loop.py "Find me a flight from PDX to HNL"   # a model using tools discovered at runtime (asks before booking)
```

The tests need no API key and no network beyond `127.0.0.1`. They use real child processes speaking real JSON-RPC over real pipes, so the protocol mechanics are genuinely exercised without any model.

## What is built

| File | What it is |
|---|---|
| `jsonrpc.py` | JSON-RPC 2.0 framing, validation and error codes. Encoded messages are always exactly one ASCII line. |
| `schema.py` | A small JSON Schema subset validator (type, required, enum, items, min/max, `additionalProperties: false`). Other keywords are ignored, not enforced. |
| `mcp_server.py` | A minimal MCP server over stdio: `initialize` with version negotiation, `notifications/initialized`, `ping`, `tools/list` (cursor pagination), `tools/call`, `resources/list`, `resources/read`, `list_changed` notifications. |
| `mcp_client.py` | A client: handshake, id matching, timeouts with `notifications/cancelled`, late-reply discard, server-crash detection, a message-size cap, and a loopback transport for fast tests. |
| `gateway.py` | **The trust layer**: allowlist, pinned tool definitions, re-verification, namespacing, side-effect approval, taint, size and tool-count limits. |
| `demo_servers.py` | One honest server and a gallery of hostile or broken ones, each modelling a reported attack or failure class. |
| `agent_loop.py` | Dynamic discovery wired into an agent loop: the tool list is recomputed from the gateway on every model call. |
| `a2a_lite.py` | An **A2A-inspired** task protocol over HTTP (agent card, task lifecycle, auth, limits). **Not wire-compatible with the real A2A spec.** |

## Two kinds of failure in MCP, which people mix up
- A **protocol error** is a JSON-RPC `error` object: unknown method, unknown tool, malformed parameters.
- A **tool error** is a normal result with `"isError": true`, so the model can read the message and recover.

Here an unknown tool is a protocol error, and invalid arguments are a tool error. Spec versions have treated input-validation errors differently, so check the version you target.

## Attack classes and where each is stopped

| Attack | Hostile server | Defense | Where |
|---|---|---|---|
| **Tool poisoning** (instructions in a description) | `poisoned` | Quarantined until a human pins the exact reviewed text. A heuristic scan tells the reviewer why it looks wrong. | `gateway.py` `pin`, `scan_description` |
| **Rug pull** (definition changes after approval) | `rugpull` | A hash of name, description and schema is pinned. The gateway re-lists before every call and blocks any change. | `gateway.py` `recheck="always"` |
| **Tool shadowing** (a description steers *other* tools) | `shadowing` | Flagged in review, plus namespacing. | `scan_description`, `SEP` |
| **Name collision / impersonation** | `collision` | The model only sees `server__tool`. | `exposed_tools` |
| **Injection via results** | `injection` | Tools you declare as untrusted-output taint the run. Side-effect tools are then refused, even if a human would approve. | `RunState`, `call` |
| **Untrusted hints** (a lying `readOnlyHint`) | `flights` | Side effects are declared in *your* config. Server annotations are recorded for display only. | `side_effect_tools` |
| **Context flooding** | `flood`, `many` | Result size cap, a message-size cap in the client, and a tool-count cap. | `max_result_chars`, `max_message_chars`, `max_tools_per_server` |
| **Hang / crash / protocol garbage** | `hang`, `crash`, `noisy` | Timeouts with cancel, EOF detection, a clear error naming the offending stdout text. The server is dropped. | `mcp_client.py`, `gateway._drop` |

## Findings from building it (bugs in my own code)
1. **The startup timeout was the call timeout.** The gateway applied one short timeout (0.5 s) to everything, including process start-up and the handshake. On this Windows machine roughly 1 connection in 5 then failed before any tool was called. I found it as a flaky test, repeated it 15 times to see the real error (`initialize timed out`), and added a separate `connect_timeout_s`.
2. **The A2A server could destroy its own rejections.** It answered `401`/`413` and closed the connection *without reading the request body*. The OS can then reset the connection and discard the reply, so the client saw a network error instead of the rejection. With small bodies that happened for about 4 of 300 requests (1.3%), which is how it showed up as an intermittent failure. The fix reads and discards a **bounded** amount of the body (1 MiB) before replying, so a hostile client still cannot make the server read forever.
3. **My first regression test for that was wrong about itself.** I called it "deterministic", then measured: against the broken server it failed only 1 run in 3. So the test is now explicitly **statistical** (12 large requests per path), and measured at **10 of 10** runs catching the original bug and 10 of 10 passing against the fix. A first version looped 300 small requests and took 16 seconds, so I replaced it.
4. **Notification timing is subtle.** A `list_changed` notification arrives *after* the reply that triggered it, so a client that only checks after one round trip can miss it. A test shows this, and it is one reason `recheck="always"` is the default: a hostile server simply won't announce its change.

## How I checked the tests can fail
I mutation-checked 21 security-critical behaviors in a scratch copy: breaking each one and confirming a test fails. This covered pin checking, re-listing before calls, the taint block, namespacing, approval, result truncation, local argument validation, default-deny, ignoring server annotations, the client's message-size cap and version check, protocol-vs-tool error handling, leaking internals, ASCII escaping, A2A authentication, task isolation, terminal-state immutability, the card-origin check, redirect handling and the body-size limit.

**19 were caught immediately. Two survived, and I looked at each instead of ignoring it:**
- **"Late replies not discarded"** was an **equivalent mutant**. Request ids only ever increase, so a stale reply carries an old id that can never match a newer request, and the plain id check already ignores it. The extra bookkeeping was dead code, so I deleted it. A stronger mutation (accepting a reply with *any* id) is caught.
- **"Redirects followed"** exposed a **real test gap**. My redirect test pointed at an unresolvable host, which fails whether or not the redirect is followed, so it proved nothing. It now uses a second real local server that records whether it was contacted and what credentials arrived. It checks that neither discovery nor an authenticated call ever reaches it. That mutation is now caught.

All 22 mutations are caught or removed as dead code.

## What is NOT known
- **No interoperability testing.** I did not test the client against any official MCP SDK, or the server against a real MCP host (an editor, a chat app). My server and client agree with *each other*; whether they agree with the real ecosystem is unverified. Both sides were written from my recollection of the spec.
- **Spec details may differ by version.** I wrote method names, error codes (including `-32002` for an unknown resource), the `notifications/cancelled` shape, and annotation names from memory. They have changed across spec versions. Check the version you target.
- **`a2a_lite.py` is not A2A.** I am not certain of the real spec's method names, fields or well-known path, so it uses its own (`task.create`, `task.get`, ...). It demonstrates the *ideas* (agent card, task lifecycle, input-required, cancellation, auth), not the standard. Use the official A2A SDK for interoperability.
- **NLWeb is not covered in code.** I did not look at the NLWeb project.
- **Nothing here ran against a live model.** `agent_loop.py` is exercised with a scripted model only.
- **Tested on Windows only.** Process handling and line endings are the likely places POSIX behavior differs.

## Limits of the design
- **The gateway cannot sandbox a server.** A local stdio server is a process running with *your* user's permissions from the moment you launch it. Pinning stops a server from changing what the *model* sees, and the gateway controls what reaches the model and what gets called. Neither stops malicious code in the server itself from acting on your machine. Run untrusted servers in a container or VM, with a restricted environment.
- **The description scanner is a heuristic.** A test shows a rephrased attack passes it. The real gate is a human reviewing a pinned hash, and review quality is a human problem.
- **`recheck="always"` costs a `tools/list` round trip per call.** `"on_notification"` is cheaper and misses silent changes (a test shows this). `"never"` trusts the first listing.
- **A pin approves a definition, not behavior.** A tool whose *description* is unchanged can still start doing something different. Pinning does nothing about that.
- **The demo server is single-threaded.** A hung tool blocks it, and the client's answer is to drop it.
- **Only stdio.** No HTTP transport, no OAuth, no server-initiated requests (sampling, elicitation), no progress notifications, no prompts.
- **A2A-inspired server:** localhost only, no TLS, static bearer tokens, no rate limiting beyond a per-client active-task cap, no persistence.
- Remote agent output and MCP results are treated as untrusted text wrapped as data. That reduces injection risk. It does not eliminate it.

## Exercises
1. Run `python server_main.py flights` and type JSON-RPC lines into it by hand (`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","clientInfo":{"name":"me"}}}`). Watch the handshake.
2. Point `MCPClient` at a real MCP server from the official ecosystem. What breaks? That is the interop gap this repo cannot tell you about.
3. Add a new hostile server of your own (for example one whose result tries to rename a tool). Write the test first, then the defense.
4. Make the gateway run each server in a restricted subprocess (a clean environment, a temporary working directory). What can still escape?
5. Replace `recheck="always"` with a cheaper strategy that still catches silent changes. What are you trading away?
