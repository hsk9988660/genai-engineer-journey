# Lesson 02: Microsoft/Azure examples, rewritten for the Claude API

Lesson 02 of [ai-agents-for-beginners](https://github.com/microsoft/ai-agents-for-beginners) uses Azure-specific code. This folder redoes every example with the Claude API and no Azure account.

- `mini_framework.py` is a roughly 120-line framework: a `@tool` decorator and an `Agent` class. It shows what real frameworks do for you.
- `examples.py` holds the lesson's examples, using that framework.
- `test_mini_framework.py` has 11 offline tests (no API key).

## Setup

```bash
pip install -r ../lesson-01-agent-loop/requirements.txt
set ANTHROPIC_API_KEY=sk-ant-...      # macOS/Linux: export ANTHROPIC_API_KEY=...
python examples.py                     # lists the examples
python examples.py travel
python -m pytest -q
```

## Translation table

| Lesson 02 (Azure) | Here (Claude) | What it really is |
|---|---|---|
| `FoundryChatClient(endpoint, model, AzureCliCredential())` | `anthropic.Anthropic()` | A connection to a model host |
| `@tool(approval_mode="never_require")` | `@tool` | Builds a JSON schema from your function |
| `@tool` needing approval | `@tool(requires_approval=True)` | Human check before a side effect |
| `provider.as_agent(name, instructions, tools)` | `Agent(name, instructions, tools=[...])` | System prompt + tool list + loop |
| `await agent.run("...")` | `agent.run("...")` | The agent loop from lesson 01 |
| `stream=True` | `agent.run("...", on_text=print)` | Token-by-token output |
| Foundry thread + `create_message` + `create_and_process_run` | Reuse one `Agent` object | The history list that is resent every call |
| `AIProjectClient` hosted agent | (not needed) | You host the loop yourself |

## Example guide

| Run | README example | Teaches |
|---|---|---|
| `hello` | `my_agent` | Smallest possible agent |
| `travel` | `book_flight` | Tool calling and approval |
| `weather` | `get_weather` | A read-only tool |
| `chain` | `dataretrieval` then `dataanalysis` | Sequential prompt chain |
| `plan_and_execute` | `planner` then `executor` | Planning split from acting |
| `thread` | Foundry "Host" menu agent | Conversation memory |
| `streaming` | `stream=True` | Responsiveness |

## Differences from the original
1. `book_flight` requires human approval. The original disabled it for a tool that spends money.
2. The weather example is realistic, with no weather API key required.
3. The "multi-agent" example is called a prompt chain here, because that is what it is.
4. The analyst uses a `calculate_growth` tool instead of doing arithmetic itself. LLMs make arithmetic mistakes, so give them a calculator.

## Exercises
1. Run `thread` and print `host.messages` afterward. Find where "that" gets its meaning.
2. Make `get_weather`'s docstring vague (`"Weather stuff"`) and see how tool use changes.
3. In `chain`, make the retriever return JSON and have the analyst parse it. Is the result more reliable?
4. Add a `cancel_flight` tool with approval and test denying it.
