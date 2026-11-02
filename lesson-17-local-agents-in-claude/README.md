# Lesson 17: local agents, hardened for small models

Lesson 17 of [ai-agents-for-beginners](https://github.com/microsoft/ai-agents-for-beginners) runs an engineering assistant entirely on one machine: a Qwen model served by Foundry Local, four file and search tools, local RAG with Chroma, an optional local MCP server, and a hybrid cloud/local routing idea. This folder keeps the good parts (a local-first loop, sandboxed tools, local retrieval, a hybrid rule) and fixes what breaks when the model is small.

> **A local model changes where your data goes and what it costs. It does not change what an agent needs.** Tool calls still have to be validated, files still sandboxed, retrieved text still untrusted and quality still measured. A small model makes every one of those more important, because it gets tool calls wrong more often.

**Read this first: no real local model was run.** Claude is a cloud API, so this lesson cannot be rebuilt "on Claude" the way the others were. I have no local model installed here, so nothing below is a result about Qwen or any real model. What is built is the harness around *any* local model, tested against a scripted fake server, plus a script (`evaluate_local.py`) for you to measure your own model.

For how the code works (a run from start to end, module reference, extension), read [IMPLEMENTATION.md](IMPLEMENTATION.md).

## Run

```bash
pip install anthropic pydantic pytest
python -m pytest -q          # 251 offline tests, no model, no key, nothing leaves 127.0.0.1, a few seconds
python harness.py            # the seven scenarios below (offline; labelled test doubles)

# Measure YOUR local model (any server that speaks the OpenAI chat-completions shape):
python evaluate_local.py --base-url http://localhost:PORT/v1 --model YOUR_MODEL --repeats 10
```

## What is built

| File | What it is |
|---|---|
| `agentloop.py` | **The core.** A tool-calling loop for a local model that never raises for a model or tool problem, plus a standard-library client for OpenAI-compatible servers that **refuses non-loopback addresses by default**. |
| `toolcalls.py` | Reads a small model's calls without trusting them: JSON that may be broken, arguments checked against a schema, unknown tools, and `<tool_call>` text calls. |
| `sandbox.py` | File tools inside one folder: resolved root, no absolute or `..` paths, links that leave the folder refused, secret files denied, size caps, binary and non-UTF-8 handled, secrets **flagged**. |
| `retrieval.py` | BM25 local search with a coverage threshold, so it can say "nothing relevant". Saves to and loads from disk. |
| `hybrid.py` | Local-first routing with **taint tracking**: once a conversation touches something sensitive, it never reaches the cloud. Every answer says who produced it. |
| `resources.py` | Will it fit? Weights, KV cache and overhead as formulas. |
| `evalharness.py`, `evaluate_local.py` | Measure a model's tool calling: eight outcome kinds, 95% intervals, latency spread. |
| `latency.py` | Latency as a distribution: warm-up discarded, median, p95, spread, tokens per second. |
| `fakeserver.py`, `fakes.py` | A real HTTP server on `127.0.0.1` and scripted model doubles for the tests. |
| `harness.py` | The lesson's patterns versus the hardened ones on seven scenarios. |

## The rules the code enforces
1. **A bad tool call is feedback, not a crash.** Broken JSON, an invented argument, a missing one, an unknown tool name or an empty reply each become a short message the model can read and fix, within a repair budget. Nothing raises.
2. **Arguments are checked against the tool's schema**, extra properties always fail, and a boolean is not a number.
3. **A call written as text** (`<tool_call>...`) is understood, and answered as a user message because it has no call id.
4. **A sandbox refusal is an answer, not a failure.** The model is told "Not available" and can say so.
5. **File paths**: no absolute paths, drive letters, `~` or `..`; the real location (after following links) must be inside the root; the root is resolved once.
6. **Secret files are not readable**: `.env*`, private keys, certificates, credential files, and folders like `.git` and `.ssh`.
7. **Reads are bounded**: a size cap with a "truncated" notice, a placeholder for binary or non-UTF-8 files, and a safe cut through multi-byte characters.
8. **A result that looks like it contains a secret is flagged `sensitive`**, and that flag taints the run.
9. **Tool results are capped, labelled "data, not instructions" and repeated identical calls are stopped.** So are runs that exceed an iteration or time limit.
10. **Search says "nothing relevant"** when too few of the question's words appear in any chunk, and tells the model not to guess.
11. **The local client refuses an address that is not this machine**, so a mistyped URL cannot send files to the internet.
12. **Taint is one-way.** A question or tool result that looks sensitive keeps the whole conversation local. Tainted turns are not remembered for later cloud prompts, and the cloud never receives raw tool output.
13. **The answer always says who produced it** (`served_by`) and why, and whether the cloud failed and local took over.

## Measured results

`python harness.py`. The "local model" is a scripted test double (one is deliberately gullible, one has a fixed seeded accuracy).

| Scenario | The lesson's pattern | Hardened |
|---|---|---|
| A model sends broken JSON, an unknown tool name, then an invented argument before a good call | **crashes** on the first (`JSONDecodeError`) | answers: 3 invalid calls, 3 repairs, 1 valid call |
| The model writes its call as `<tool_call>` text | the raw tag is returned **as the final answer** | the call is run |
| Reading `.env`, `config.txt`, a 240 KB file, a binary file, `../outside.txt`, `auth.py` | **2 secrets returned unflagged**, 1 crash (binary), 240,000 characters of one file into the context | 0 unflagged secrets, 1 returned but **flagged** (`config.txt`), 2 denied, 0 crashes, the big file cut at about 20,000 |
| 3 questions the documents never mention | **3 of 3** get documents anyway | **0 of 3**; the 2 questions the documents do cover are still answered |
| A conversation reads a file with a password, then asks a "hard" question | route by request: the **password is sent to the cloud** | taint tracking: stays local, 0 cloud calls |
| A README tells the model to read `.env` | the model reads it: **the secret reaches the model** | the gullible model obeys and is refused; the secret never reaches it |
| Will a 7B model fit? (formulas) | | 8 GB: fp16 and int8 do not fit, int4 allows about 31,700 tokens of conversation. 16 GB: only fp16 fails. int4 at a full 32k context on 8 GB: 3.8 GB weights + 1.88 GB cache = 6.06 GB, which is over the 6.0 GB left after a 2 GB reserve |
| How sure is a measurement? (a model with a true 80% rate) | | 10 attempts: observed 90%, interval 60% to 98%. 200 attempts: observed 74%, interval 68% to 80% |

**How to read this honestly.**
- The first four rows are about *mechanisms against scripted misbehaviour*. They do not say how often Qwen, or any real model, emits broken JSON or follows an injected instruction.
- The last row is a reminder that even 200 attempts can land 6 points away from the truth, which is why the evaluation reports an interval and not a single percentage.
- "The lesson's pattern" is my re-statement of the notebook's behaviour in the harness (I cannot import a notebook), not its exact code.
- The memory numbers are formulas from a model shape I **recalled from memory**. I did not check it against any model card.

## Findings from building it
1. **A "secrets returned: 0" metric hid something.** My first file-probe summary counted the sandbox as returning no secrets, but it *does* return `config.txt` (with a password) to the model, only flagged. The metric now separates "unflagged" from "flagged".
2. **My first multi-byte repair was wrong.** When a size cap cuts a file in the middle of a character, I dropped four bytes and retried, which can still be mid-character. A test with 2-byte characters failed. It now tries dropping 1, 2 or 3 bytes.
3. **A strict `..` rule the first tests did not exercise.** `sub/../auth.py` stays inside the folder, but the sandbox refuses any `..` on purpose. Nothing proved that until mutation checking.
4. **Mutation checking found eleven weak spots**, including a ranking test that passed with the IDF term removed, a tool-failure test that used the one exception its mutant still caught, and token counts that were never summed in a test.
5. **Three tests were flaky in this environment.** The directory-link helper called `subprocess.run` with an inherited stdin handle that was sometimes invalid (`WinError 6`). Giving the child an explicit empty stdin fixed it, and the suite then passed 10 runs in a row.
6. **This machine cannot create symlinks** (a missing privilege). The tests use a real symlink where allowed and a Windows **directory junction** otherwise, and the sandbox correctly refuses a junction that leads outside. I could not create a *file* symlink, so two filters (a folder or a file whose real location is outside the root) are tested by patching the resolver rather than with a real link.

## How I checked the tests can fail
I mutation-checked 115 behaviors in a scratch copy (each run in parallel, every mutant checked to compile): argument validation, call parsing, every path rule, the deny rules, size and binary handling, the walk, the sensitivity flags, the coverage threshold, tokenization and ranking, the client's error mapping and loopback rule, every loop limit, feedback and repair accounting, tool-failure handling, taint rules, the cloud fallback, the resource formulas, latency summaries, the evaluation classifier and intervals, and the command-line guard. **104 were caught on the first run. The 11 survivors were test gaps, I added tests, and all 115 are now caught.** That means these 115 changes are detected, not that the suite is complete. Two of the 11 are caught only by tests that patch the resolver, as explained above.

## What is NOT known
- **No real local model was run.** I do not know how often Qwen or any other small model produces broken JSON, text-format calls, wrong tools or invented arguments, or how quantization changes that. `evaluate_local.py` exists so you can find out, and I have **not** run it against a real server.
- **The client was only tested against a scripted fake server.** Real servers differ: streaming defaults, how they return tool calls, error bodies, and what they do on a first request while a model loads.
- **The resource numbers are formulas.** They ignore mixed-precision quantization formats (a real "int4" file is somewhat larger than 0.5 bytes per parameter), activation memory, memory mapping, and GPU/NPU offload. The example model shape is from my recollection and is labelled to be verified.
- **My critique of the lesson's Chroma and Foundry Local claims is partly from recollection.** I believe Chroma downloads its default embedding model on first use and has had anonymous telemetry on by default, but I did not verify either. Check before relying on a "no network" claim.
- **Retrieval quality is unmeasured** beyond the tiny corpora in the tests. It is lexical only: a question phrased with different words than the document ("how do we keep logins safe?" against a document about hashing) will miss.
- **The cloud function was tested with scripted replies only,** not the real API.
- **Windows only.** Case sensitivity, path forms and link behaviour differ on Linux and macOS.

## Limits of the design
- **Sensitivity detection is a list of patterns.** A secret with no recognisable shape (a passphrase, an internal hostname) is not flagged and could reach the cloud.
- **Taint is blunt.** One flagged file keeps the whole conversation local, even for harmless later questions. That is the safe direction, and it costs usefulness.
- **The model's own answer is not scanned.** If a local model repeats a secret in its reply, that reply is not tainted.
- **The deny list is by name.** `secrets.yaml` or `prod.cfg` would be readable (and flagged only if they contain a secret shape). An allow-list of file types would be safer.
- **Prompt injection is bounded, not solved.** With read-only tools the damage is limited to what the model is shown and says, but a hostile file can still make it say something misleading.
- **No write tools, no MCP.** Anything that writes or runs commands needs lesson 16's approvals and idempotency, and must stay out of conversations that can reach the cloud.
- **No streaming**, so time to first token is not measured.
- **One agent at a time.** Not tested under concurrency.
- **A restart forgets a conversation's taint** unless you persist it.

## Exercises
1. Run `evaluate_local.py` against your own model with `--repeats 20`. How wide are the intervals, and which outcome kind (bad JSON, wrong tool, no call) is most common?
2. Re-run it at two quantization levels of the same model. Does tool calling change?
3. Replace the secret-file deny list with an allow-list of file extensions. What do you lose?
4. Put an embedding model behind `BM25Index.search`'s shape. Which "nothing relevant" threshold works, and how did you pick it?
5. Add a tool that writes a file. What would you need before you let it run in a conversation that can reach the cloud?
