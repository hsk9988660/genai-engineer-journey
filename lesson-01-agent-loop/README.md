# Lesson 01: the agent loop, with no framework

An agent is an LLM in a loop, with tools, working toward a goal. `agent.py` is a working agent in about 100 lines. It uses the Claude API directly and prints every step.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
set ANTHROPIC_API_KEY=sk-ant-...  # macOS/Linux: export ANTHROPIC_API_KEY=...
python agent.py "I want a warm beach holiday on a low budget"
```

Run the tests (no API key needed):

```bash
python -m pytest -q
```

## What to notice
1. The model only *requests* tools. `run_agent` executes them.
2. Tool errors go back to the model as data, so it can recover.
3. `max_steps` prevents runaway loops and surprise bills.
4. Token usage grows each step because the whole history is resent.

## Exercises
1. Add a `get_weather(destination, month)` tool. Make its description vague, then precise, and compare what the model does.
2. Set `max_steps=1` and watch the agent get cut off.
3. Add a human approval prompt before any tool runs.
4. Print the cost per run from the token counts.
