"""Error recovery in CODE: retry, fallback, and a circuit breaker.

The lesson 09 notebook told the model, in its prompt, to "try the primary system, and if it returns a 404 try the backup".
That is deterministic logic. Putting it in a prompt means it works only when the model reads an error correctly and picks
the right tool. Here the model calls ONE tool and the wrapper handles recovery, and records what happened so the answer
can be transparent about it.

  NotFoundError   permanent: the item does not exist. Retrying is pointless, and it does NOT count against the breaker
                  because the service itself is healthy.
  transient       timeouts and connection errors: retried, and they DO count against the breaker.
  CircuitBreaker  after `failure_threshold` consecutive transient failures the primary is skipped entirely until
                  `reset_after_s` passes, then one trial call decides whether to close the circuit again.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable


class NotFoundError(Exception):
    pass


class AllSourcesFailed(Exception):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, reset_after_s: float = 30.0, clock=time.monotonic):
        self.failure_threshold, self.reset_after_s, self.clock = failure_threshold, reset_after_s, clock
        self.state, self.failures, self.opened_at = "closed", 0, 0.0
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            if self.state == "closed":
                return True
            if self.state == "open" and self.clock() - self.opened_at >= self.reset_after_s:
                self.state = "half_open"  # let exactly one trial call through
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self.state, self.failures = "closed", 0

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self.state == "half_open" or self.failures >= self.failure_threshold:
                self.state, self.opened_at = "open", self.clock()


@dataclass
class Sourced:
    value: Any
    source: str  # "primary" or "backup"
    notes: list = field(default_factory=list)


def with_fallback(primary: Callable, backup: Callable, retries: int = 1, breaker: CircuitBreaker | None = None,
                  sleep: Callable[[float], None] = lambda s: None, transient=(TimeoutError, ConnectionError)) -> Callable:
    def call(*args, **kwargs) -> Sourced:
        notes, errors = [], []
        if breaker is not None and not breaker.allow():
            notes.append("primary system skipped: circuit open after repeated failures")
            errors.append("primary: circuit open")
        else:
            attempts = 0
            while True:
                try:
                    value = primary(*args, **kwargs)
                    if breaker:
                        breaker.record_success()
                    return Sourced(value, "primary", notes)
                except NotFoundError as exc:
                    errors.append(f"primary: {exc}")
                    notes.append("not found in the primary system; used the backup")
                    if breaker:
                        breaker.record_success()  # the service answered correctly; the item just is not there
                    break
                except transient as exc:
                    attempts += 1
                    errors.append(f"primary: {type(exc).__name__}: {exc}")
                    if breaker:
                        breaker.record_failure()
                    if attempts > retries:
                        notes.append("primary system unavailable; used the backup")
                        break
                    sleep(0.1 * attempts)
                except Exception as exc:  # an unexpected failure: do not retry blindly
                    errors.append(f"primary: {type(exc).__name__}: {exc}")
                    notes.append("primary system failed unexpectedly; used the backup")
                    if breaker:
                        breaker.record_failure()
                    break
        try:
            return Sourced(backup(*args, **kwargs), "backup", notes)
        except Exception as exc:
            errors.append(f"backup: {type(exc).__name__}: {exc}")
            raise AllSourcesFailed(errors) from exc

    return call


# --- the notebook's two flight systems, as stubs --------------------------------------------------
PRIMARY = {"Paris": "08:00, 12:30, 17:45 from $350", "Tokyo": "11:00, 23:30 from $890", "Barcelona": "07:15, 14:00, 19:30 from $280"}
BACKUP = {"Berlin": "09:00, 16:00 from $220", "Sydney": "22:00 from $1200", "New York City": "06:00, 10:30, 15:00, 20:00 from $450"}


def primary_lookup(destination: str) -> str:
    if destination in PRIMARY:
        return PRIMARY[destination]
    raise NotFoundError(f"no flights for {destination} in the primary system")


def backup_lookup(destination: str) -> str:
    if destination in BACKUP:
        return BACKUP[destination]
    raise NotFoundError(f"no flights for {destination} in the backup system")


def make_flight_tool(breaker: CircuitBreaker | None = None) -> Callable[[str], dict]:
    """The single tool the model sees. Recovery is invisible to it, but reported in `source` and `notes`."""
    lookup = with_fallback(primary_lookup, backup_lookup, breaker=breaker)

    def get_flight_times(destination: str) -> dict:
        try:
            r = lookup(destination)
            return {"destination": destination, "flights": r.value, "source": r.source, "notes": r.notes}
        except AllSourcesFailed:
            return {"error": f"No flights found for {destination} in any system."}

    return get_flight_times
