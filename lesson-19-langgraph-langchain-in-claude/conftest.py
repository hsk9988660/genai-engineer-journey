import pytest

from desk import Desk


@pytest.fixture
def desk() -> Desk:
    return Desk()


def tool_texts(result: dict) -> list[str]:
    return [str(m.content) for m in result["messages"] if m.type == "tool"]


def kinds(result: dict) -> list[str]:
    return [m.type for m in result["messages"]]
