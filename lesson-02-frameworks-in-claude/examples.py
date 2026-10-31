"""Every example from lesson 02's README, rewritten for the Claude API.

Run one:   python examples.py travel
List all:  python examples.py
Needs ANTHROPIC_API_KEY in your environment.
"""

import sys

from mini_framework import Agent, tool


# 1. "Create an agent" (README: my_agent / "Hello, World!") -------------------------------
def hello():
    agent = Agent("my_agent", "You are a helpful assistant.")
    print(agent.run("Hello, World!"))


# 2. Tool calling (README: book_flight) ---------------------------------------------------
# The Microsoft example set approval_mode="never_require" on a tool that books and pays for
# travel. Here an action with side effects asks a human first: requires_approval=True.
@tool(requires_approval=True)
def book_flight(date: str, location: str) -> str:
    """Book a flight to a location on a date. Use ISO dates like 2025-01-01."""
    return f"Travel was booked to {location} on {date}"


def travel():
    agent = Agent(
        "travel_agent",
        "Help the user book travel. Use the book_flight tool when you have a destination and date.",
        tools=[book_flight],
        verbose=True,  # prints each tool request so you can see the loop
    )
    print(agent.run("I'd like to go to New York on January 1, 2025"))


# 3. A read-only tool (README: get_weather). No approval is needed because it changes nothing.
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {location} is sunny, 72F."


def weather():
    agent = Agent("weather_agent", "Help users check the weather.", tools=[get_weather], verbose=True)
    print(agent.run("Should I bring a jacket in Lisbon?"))


# 4. Two agents, one after the other (README: dataretrieval -> dataanalysis) ---------------
# The README calls this "multi-agent", but it is a prompt chain: agent A's text output is
# pasted into agent B's prompt. It is a good, cheap pattern. Real hand-offs come in lesson 08.
@tool
def retrieve_sales_data(quarter: str) -> dict:
    """Fetch total sales in dollars for a quarter, like 'Q3' or 'Q4'."""
    return {"Q3": {"sales": 120_000}, "Q4": {"sales": 150_000}}.get(quarter, {"error": "unknown quarter"})


@tool
def calculate_growth(previous: float, current: float) -> float:
    """Percentage growth from a previous value to a current value."""
    return round((current - previous) / previous * 100, 1)


def chain():
    retriever = Agent("dataretrieval", "Retrieve relevant data using your tools. Reply with the raw numbers only.",
                      tools=[retrieve_sales_data], verbose=True)
    analyst = Agent("dataanalysis", "Analyze the data you are given and give two sentences of insight. "
                    "Use calculate_growth for percentages; never do the arithmetic yourself.",
                    tools=[calculate_growth], verbose=True)
    data = retriever.run("Retrieve sales data for Q3 and Q4")
    print("\n--- retriever said:", data, "\n")
    print(analyst.run(f"Analyze this data: {data}"))


# 5. Planner then executor (README: planner / executor) -----------------------------------
@tool
def search_flights(destination: str) -> list:
    """Search available flights to a destination."""
    return [{"flight": "AF123", "price": 480}, {"flight": "LH456", "price": 520}]


@tool
def search_hotels(destination: str) -> list:
    """Search available hotels in a destination."""
    return [{"hotel": "Hotel Lumiere", "per_night": 140}, {"hotel": "Le Petit", "per_night": 95}]


def plan_and_execute():
    planner = Agent("planner", "Break the task into 3-5 short numbered steps. Output only the steps.")
    executor = Agent("executor", "Carry out the plan using your tools, then summarize the options found.",
                     tools=[search_flights, search_hotels], verbose=True)
    plan = planner.run("Plan a trip to Paris")
    print("--- plan:\n" + plan + "\n")
    print(executor.run(f"Execute this plan: {plan}"))


# 6. Threads and messages (README: the Foundry "Host" menu agent) -------------------------
# In the Azure Agent Service you create a thread, add messages, then start a run. With Claude
# the conversation is just a list. The Agent keeps it, so reusing one Agent object IS a thread.
@tool
def get_specials() -> str:
    """List today's menu specials."""
    return "Special Soup: Clam Chowder. Special Salad: Cobb Salad. Special Drink: Chai Tea."


@tool
def get_item_price(menu_item: str) -> str:
    """Get the price of one menu item."""
    return "$9.99"


def thread():
    host = Agent("Host", "Answer questions about the menu.", tools=[get_specials, get_item_price])
    for user_input in ["Hello", "What is the special soup?", "How much does that cost?", "Thank you"]:
        print(f"# User: {user_input!r}")
        print(f"# Agent: {host.run(user_input)}")  # 'that' resolves because the history is resent
    print(f"\nmessages in thread: {len(host.messages)}   input tokens used: {host.input_tokens}")


# 7. Streaming (lesson 01 notebook: stream=True) -----------------------------------------
def streaming():
    agent = Agent("guide", "You are a concise travel guide.")
    agent.run("Tell me about Tokyo as a travel destination", on_text=lambda t: print(t, end="", flush=True))
    print()


EXAMPLES = {f.__name__: f for f in (hello, travel, weather, chain, plan_and_execute, thread, streaming)}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if name not in EXAMPLES:
        print("Pick one:", ", ".join(EXAMPLES))
        sys.exit(1)
    EXAMPLES[name]()
