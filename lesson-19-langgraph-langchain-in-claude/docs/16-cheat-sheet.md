# Cheat sheet: the LangChain and LangGraph you need as an agent developer

Everything here appears in the code of this lesson. The import paths are the ones in `langchain` 1.4.9 and `langgraph` 1.2.20. The libraries change fast, so check them against your installed version.

## The functions and classes to know

| You want to... | Use | From | Stage |
|---|---|---|---|
| Get a chat model from a provider string | `init_chat_model("anthropic:claude-sonnet-5-5")` | `langchain.chat_models` | live |
| Build an agent (the loop) | `create_agent(model, tools, system_prompt=, middleware=, checkpointer=, store=, context_schema=, response_format=)` | `langchain.agents` | 1 |
| Define a tool | `@tool` on a function (docstring = description) | `langchain.tools` | 1 |
| Give a tool hidden, verified inputs | `runtime: ToolRuntime[Ctx]` (`.context`, `.store`, `.config`, `.tool_call_id`) | `langchain.tools` | 1 |
| Run, stream | `agent.invoke(input, config, context=)`, `.stream(..., stream_mode="updates")` | on any compiled graph | 1, 7 |
| Make a model return a typed object | `model.with_structured_output(Schema)` | on any chat model | 3 |
| Describe messages | `HumanMessage`, `AIMessage`, `ToolMessage`, `SystemMessage` | `langchain.messages` | 1 |
| Trim a long history | `trim_messages` | `langchain.messages` | 4 |
| Build your own graph | `StateGraph(State, context_schema=)`, `.add_node`, `.add_edge`, `.add_conditional_edges`, `.compile()` | `langgraph.graph` | 2, 7 |
| Mark start and end | `START`, `END` | `langgraph.graph` | 2 |
| Merge messages in state | `Annotated[list, add_messages]` | `langgraph.graph.message` | 2 |
| Run tool calls in a graph | `ToolNode(tools)` | `langgraph.prebuilt` | 2 |
| Save state per thread | `InMemorySaver()` and `config={"configurable": {"thread_id": ...}}` | `langgraph.checkpoint.memory` | 4 |
| Long-term memory | `InMemoryStore()` and `runtime.store.put/search/get` | `langgraph.store.memory` | 4 |
| Inspect saved state | `graph.get_state(config)`, `graph.get_state_history(config)` | on a compiled graph | 4 |
| Pause for a person | `HumanInTheLoopMiddleware(interrupt_on={...})` or `interrupt(value)` | `langchain.agents.middleware`, `langgraph.types` | 5 |
| Resume | `graph.invoke(Command(resume=...), same_config)` | `langgraph.types` | 5 |
| Add a concern around a tool | `@wrap_tool_call` | `langchain.agents.middleware` | 6 |
| Build the prompt per request | `@dynamic_prompt` | `langchain.agents.middleware` | 6 |
| Draw a graph | `graph.get_graph().draw_mermaid()` | on a compiled graph | 2 |

## More LangGraph (Part 2)

| You want to... | Use | Stage |
|---|---|---|
| Start a variable number of parallel workers | a conditional edge returning `[Send("node", private_input), ...]` | 8 |
| Let parallel nodes write one key | an `Annotated[list, operator.add]` reducer (without one: `InvalidUpdateError`) | 8 |
| Make a join wait for several branches | `add_edge(["a", "b"], "join")` (separate edges run it once per branch) | 8 |
| Change state and choose the next node together | return `Command(update=..., goto="node")`, annotated `Command[Literal[...]]` | 9 |
| Ask a question from inside a node | `answer = interrupt(question)` ... `invoke(Command(resume=answer), same_config)` | 9 |
| Know what runs on resume | the node runs again **from its first line**, so nothing irreversible before `interrupt()` | 9 |
| Limit what callers send and receive | `StateGraph(State, input_schema=..., output_schema=...)` | 9 |
| Keep state across restarts | `SqliteSaver(sqlite3.connect(path, check_same_thread=False))` (Postgres for many servers) | 10 |
| Look back, replay, or try a what-if | `get_state_history`, `invoke(None, old_config)`, `update_state(old_config, changes)` | 10 |
| Retry a node on chosen errors | `add_node(..., retry_policy=RetryPolicy(max_attempts=, retry_on=...))` | 10 |
| Stream progress | `stream(..., stream_mode=["updates", "messages", "custom"])`; `subgraphs=True`; `astream` | 11 |
| Emit your own progress events | a `StreamWriter` argument on a node, `writer({...})` | 8, 11 |

## More LangChain (Part 2)

| You want to... | Use | Stage |
|---|---|---|
| Build a prompt with variables and history | `ChatPromptTemplate.from_messages`, `MessagesPlaceholder`, `.partial` | 12 |
| Join steps | `a \| b \| c` (a `RunnableSequence`); your code goes in `RunnableLambda` | 12 |
| Run branches on one input | `RunnableParallel(x=..., y=...)`; keep the input with `RunnablePassthrough.assign` | 12 |
| Parse model output | `StrOutputParser`, `PydanticOutputParser` (errors raise `OutputParserException`) | 12 |
| Make any runnable resilient | `.with_retry(...)`, `.with_fallbacks([...], exceptions_to_handle=...)` | 12 |
| Process many inputs | `.batch(items, config={"max_concurrency": n}, return_exceptions=True)` | 12 |
| Shorten history safely | `trim_messages(..., start_on="human", include_system=True)`; `RemoveMessage` | 13 |
| Limit what a tool accepts | a pydantic `args_schema` with `Field(gt=, le=, pattern=)` | 13 |
| Flag a tool failure | `raise ToolException(...)` with `tool.handle_tool_error = True` | 13 |
| Give the app data the model need not read | `@tool(response_format="content_and_artifact")` | 13 |
| Let a tool write agent state | return `Command(update=...)` with `InjectedToolCallId` and a `state_schema` | 13 |
| Read state inside a tool | `Annotated[dict, InjectedState]` | 13 |
| Split and index documents | `Document`, `RecursiveCharacterTextSplitter`, `Embeddings`, `InMemoryVectorStore` | 14 |
| Search with access control | `similarity_search_with_score(q, k=, filter=lambda doc: ...)` then a score threshold | 14 |
| Count tokens and cost | a `BaseCallbackHandler` passed as `config={"callbacks": [...]}` | 15 |
| Skip repeated calls, limit call rate | `cache=InMemoryCache()`, `rate_limiter=InMemoryRateLimiter(...)` on the chat model | 15 |
| Retry, fall back, summarise | `ModelRetryMiddleware`, `ModelFallbackMiddleware`, `SummarizationMiddleware` | 15 |
| Pick a provider by string | `init_chat_model("provider:model")` | 15 |

## Middleware that ships with LangChain

| Name | Job |
|---|---|
| `HumanInTheLoopMiddleware` | Pause for approve, edit, reject (stage 5) |
| `ToolRetryMiddleware`, `ModelRetryMiddleware` | Retry temporary failures (stage 6) |
| `ToolErrorMiddleware` | Turn chosen exceptions into messages for the model (stage 6) |
| `ToolCallLimitMiddleware`, `ModelCallLimitMiddleware` | Budgets per run or per thread (stage 6) |
| `ModelFallbackMiddleware` | Switch to another model when one fails |
| `SummarizationMiddleware`, `ContextEditingMiddleware` | Keep long conversations inside the context window |
| `PIIMiddleware` | Detect and redact personal data |
| `LLMToolSelectorMiddleware` | Pick relevant tools when you have many |
| `TodoListMiddleware` | Give an agent a planning list |
| `ShellToolMiddleware`, `FilesystemFileSearchMiddleware` | Give an agent a shell or file search (high risk, sandbox them) |

The ones not used in this lesson are listed from the package's exports. Read their docstrings before you rely on them.

## `create_agent` or a hand-built graph?

| Use `create_agent` when... | Build a `StateGraph` when... |
|---|---|
| The task is "tools in a loop until done" | There are distinct stages with different rules |
| Middleware covers what you need | Code, not the model, must choose between paths |
| You want the least code | You need several agents, fixed hand-offs, or parallel branches |
| One model owns the whole conversation | Some steps need no model at all |

Real systems often do both, as stage 7 does: a graph for the workflow, an agent inside one node.

## Habits that come from the stages

1. Put rules about money and permissions in plain code, behind the tools (stage 0).
2. Identity comes from verified context, never from a model-supplied argument (stages 1 and 4).
3. Return business errors as data. Let bugs be loud, then decide on purpose how they reach the model (stages 1 and 6).
4. A schema checks shape. Check facts against ground truth in code (stage 3).
5. Cap loops. Know your framework's default limit; in the tested version it is very large (stages 1 and 6).
6. A human's approval is not the rule. Re-check in code when the action runs (stage 5).
7. Make side effects idempotent, and scope the key by thread (stages 5 and 7).
8. Choose middleware order on purpose, and test the order you chose (stage 6).
9. Let code, not the model, choose the route in a workflow (stage 7).
10. Test the whole thing offline with a scripted model, and run a mutation check on your tests.
11. Cap every fan-out, and give every parallel write a reducer (stage 8).
12. Never do anything irreversible before an `interrupt()`, because the node runs again on resume (stage 9).
13. Replay and fork re-run real nodes. Make side effects idempotent before you use time travel (stage 10).
14. Check what a stream event really is before you animate it (stage 11).
15. Keep personal data out of prompts with a plain function in front of the model (stage 12).
16. Cut message history only at valid places (stage 13).
17. Filter by permission before you rank, threshold the score, and verify every citation (stage 14).
18. Order your model middleware on purpose, and count cache hits instead of assuming them (stage 15).

## Where to go next

- **Persistence:** stage 10 uses the SQLite saver. For several servers use the Postgres saver (`langgraph-checkpoint-postgres`), which this lesson did not run.
- **Not covered here:** LangGraph's functional API (`@entrypoint`, `@task`), the LangGraph server and SDK, and `langchain.mcp`. Read their docs once you are comfortable with stages 1 to 15.
- **Tools from outside:** LangChain has a `langchain.mcp` module for connecting to MCP servers. Read lesson 11 first so you know what to distrust.
- **Tracing and evaluation:** LangSmith traces every step of a graph. Whatever you use, measure with confidence intervals (lesson 10).
- **More models:** the partner packages (`langchain-anthropic`, `langchain-openai`, `langchain-ollama`, ...) all follow the same chat model interface, so `init_chat_model("provider:model")` is all you change.
- **Serving:** wrap the compiled graph in an API with the admission control and idempotency from lesson 16.
