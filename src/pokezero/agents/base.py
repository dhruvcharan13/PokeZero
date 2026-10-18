from __future__ import annotations

from typing import Protocol

from pokezero.env.base import Decision


class Agent(Protocol):
    name: str

    def act(self, decision: Decision) -> list[int]:
        """Return a legal list of option indices for ``decision``."""
        ...

    def reset(self) -> None:
        """Called at the start of every game."""
        ...
