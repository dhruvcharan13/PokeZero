"""Competition-agnostic game interface.

Everything outside ``pokezero.env.cabt`` talks to a game through these types only, so the
competition-specific engine can be swapped (or mocked in tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Decision:
    """One pending selection for one player: pick between ``min_count`` and ``max_count``
    distinct option indices.

    ``obs`` is exactly the observation the competition hands the acting player (public board
    plus that player's own hand). Nothing in this object reveals the opponent's hidden zones.
    """

    player: int
    select_type: int
    context: int
    min_count: int
    max_count: int
    options: tuple[dict[str, Any], ...]
    obs: dict[str, Any]

    @property
    def num_options(self) -> int:
        return len(self.options)


def validate_picks(decision: Decision, picks: list[int]) -> None:
    """Raise ``ValueError`` unless ``picks`` is a legal answer to ``decision``."""
    n = decision.num_options
    if not all(isinstance(p, int) and 0 <= p < n for p in picks):
        raise ValueError(f"pick out of range 0..{n - 1}: {picks}")
    if len(set(picks)) != len(picks):
        raise ValueError(f"duplicate picks: {picks}")
    if not decision.min_count <= len(picks) <= min(decision.max_count, n):
        raise ValueError(
            f"need {decision.min_count}..{min(decision.max_count, n)} picks, got {len(picks)}"
        )


class GameEnv(Protocol):
    """A single two-player game."""

    def reset(self, seed: int | None = None) -> Decision: ...

    def step(self, picks: list[int]) -> Decision | None:
        """Apply the acting player's picks; return the next decision or ``None`` if over."""
        ...

    @property
    def done(self) -> bool: ...

    def reward(self, player: int) -> float:
        """+1 win, -1 loss, 0 draw. Only valid once ``done``."""
        ...

    def close(self) -> None: ...


@dataclass
class TrajectoryStep:
    player: int
    picks: list[int]
    num_options: int
    select_type: int


@dataclass
class Trajectory:
    """Replayable record of one game: decks, root seed (Python RNG only), and every pick."""

    decks: list[list[int]]
    seed: int | None
    steps: list[TrajectoryStep] = field(default_factory=list)
    result: int | None = None  # 0/1 = winning player, 2 = draw

    def to_json(self) -> dict[str, Any]:
        return {
            "decks": self.decks,
            "seed": self.seed,
            "result": self.result,
            "steps": [[s.player, s.picks, s.num_options, s.select_type] for s in self.steps],
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Trajectory:
        steps = [TrajectoryStep(p, list(k), n, t) for p, k, n, t in d["steps"]]
        return cls(decks=d["decks"], seed=d["seed"], steps=steps, result=d["result"])
