"""Lesson 01: an AI agent is an LLM in a loop, with tools, working toward a goal.

A framework-free agent built on the Claude API. Every step is printed, so you can
watch the loop: model call -> tool request -> your code runs the tool -> result
goes back to the model -> repeat until the model gives a final answer.

Run:  python agent.py "I want a warm beach holiday. What do you recommend?"
"""

import json
import os
import sys
from dataclasses import dataclass, field

import anthropic

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = (
    "You are a helpful travel agent. Help users find their perfect vacation "
    "destination based on their preferences. Use the tools to see which "
    "destinations are available before recommending anything."
)


# --- Tools -------------------------------------------------------------------
# A tool is a plain function plus a JSON schema. The model only ever sees the
# schema (name + description + parameters), never your code, so the description
# is effectively a prompt. Vague descriptions cause wrong or missed tool calls.

DESTINATIONS = {
    "Barcelona": {"climate": "warm", "beach": True, "budget": "medium"},
    "Paris": {"climate": "mild", "beach": False, "budget": "high"},
    "Tokyo": {"climate": "mild", "beach": False, "budget": "high"},
    "Bali": {"climate": "tropical", "beach": True, "budget": "low"},
    "Cape Town": {"climate": "warm", "beach": True, "budget": "medium"},
    "Reykjavik": {"climate": "cold", "beach": False, "budget": "high"},
}


def get_destinations() -> list[str]:
    return sorted(DESTINATIONS)


def get_destination_details(name: str) -> dict:
    if name not in DESTINATIONS:
        # Return errors as data. The model can read them and recover.
        return {"error": f"Unknown destination '{name}'. Call get_destinations first."}
    return {"name": name, **DESTINATIONS[name]}


TOOLS = [
    {
        "name": "get_destinations",
        "description": "List the names of all vacation destinations we can book.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_destination_details",
        "description": (
            "Get climate, whether it has a beach, and budget level for ONE "
            "destination. The name must come from get_destinations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Destination name"}},
            "required": ["name"],
        },
    },
]

REGISTRY = {
    "get_destinations": get_destinations,
    "get_destination_details": get_destination_details,
}


# --- The agent loop ----------------------------------------------------------


@dataclass
class AgentResult:
    answer: str
    steps: int
    input_tokens: int = 0
    output_tokens: int = 0
    stopped_early: bool = False
    trace: list[str] = field(default_factory=list)


def run_agent(user_msg: str, client=None, max_steps: int = 6, verbose: bool = True) -> AgentResult:
    """Run the agent until it answers or max_steps is reached."""
    client = client or anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    messages = [{"role": "user", "content": user_msg}]
    result = AgentResult(answer="", steps=0)

    def log(line: str) -> None:
        result.trace.append(line)
        if verbose:
            print(line)

    for step in range(1, max_steps + 1):  # always cap the loop
        result.steps = step
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        result.input_tokens += resp.usage.input_tokens
        result.output_tokens += resp.usage.output_tokens
        log(f"[step {step}] stop_reason={resp.stop_reason}")

        # The whole assistant turn goes back into history, tool_use blocks included.
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            result.answer = "".join(b.text for b in resp.content if b.type == "text")
            return result

        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            log(f"  -> model requests {block.name}({json.dumps(block.input)})")
            fn = REGISTRY.get(block.name)
            if fn is None:
                out, is_error = {"error": f"No such tool: {block.name}"}, True
            else:
                try:
                    # YOUR code executes the tool. The model never runs anything.
                    out, is_error = fn(**block.input), False
                except Exception as exc:  # a bad tool call must not crash the loop
                    out, is_error = {"error": str(exc)}, True
            log(f"  <- result: {json.dumps(out)}")
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(out),
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": tool_results})

    result.stopped_early = True
    result.answer = "Stopped: step limit reached before the agent finished."
    return result


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or "I'm looking for a warm beach destination on a low budget."
    res = run_agent(prompt)
    print("\n=== Answer ===")
    print(res.answer)
    print(f"\nsteps={res.steps} input_tokens={res.input_tokens} output_tokens={res.output_tokens}")
