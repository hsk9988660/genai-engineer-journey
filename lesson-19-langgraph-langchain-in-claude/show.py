"""Print a conversation as readable lines, so you can watch an agent run."""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


def format_messages(messages: list[BaseMessage]) -> list[str]:
    lines = []
    for message in messages:
        if isinstance(message, SystemMessage):
            lines.append(f"system   : {_short(message.content)}")
        elif isinstance(message, HumanMessage):
            lines.append(f"customer : {_short(message.content)}")
        elif isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                lines.append(f"model    : asks for {tool_call['name']}({_args(tool_call['args'])})")
            if message.content:
                lines.append(f"model    : {_short(message.content)}")
        elif isinstance(message, ToolMessage):
            lines.append(f"tool     : {_short(message.content)}")
    return lines


def final_text(messages: list[BaseMessage]) -> str:
    """The text of the last message from the model."""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            return message.content if isinstance(message.content, str) else str(message.content)
    return ""


def _args(args: dict) -> str:
    return ", ".join(f"{key}={value!r}" for key, value in args.items())


def _short(content: object, limit: int = 110) -> str:
    text = content if isinstance(content, str) else str(content)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."
