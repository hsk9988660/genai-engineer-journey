"""An idempotent booking service.

Agents retry. Networks time out, models repeat themselves. A tool with side effects must be safe to call
twice with the same request: the second call returns the first booking instead of creating another.

The lesson 04 notebook built its confirmation number with Python's hash(), which is randomized per process,
so the same passenger got a different "confirmation" on every run. Here the ID is derived from a stable SHA-256.
"""

import hashlib
import threading


class BookingService:
    def __init__(self):
        self._bookings: dict[str, dict] = {}
        self._lock = threading.Lock()  # tools may run in parallel threads

    @staticmethod
    def idempotency_key(origin: str, destination: str, passenger_name: str) -> str:
        raw = f"{origin.upper()}|{destination.upper()}|{passenger_name.strip().lower()}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def book_flight(self, origin: str, destination: str, passenger_name: str) -> dict:
        key = self.idempotency_key(origin, destination, passenger_name)
        with self._lock:
            existing = self._bookings.get(key)
            if existing:
                return existing | {"duplicate": True}
            booking = {"confirmation": "TRV-" + key[:8].upper(), "origin": origin.upper(),
                       "destination": destination.upper(), "passenger": passenger_name.strip(), "duplicate": False}
            self._bookings[key] = booking
            return booking

    def count(self) -> int:
        return len(self._bookings)
