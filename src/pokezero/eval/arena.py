"""Head-to-head games, round-robin tables and Elo."""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from pokezero.agents.base import Agent
from pokezero.env.base import Decision, Trajectory
from pokezero.env.cabt import CabtEnv
from pokezero.eval.elo import fit_elo

MAX_DECISIONS = 3000  # safety valve; a drawn game is scored 0.5


@dataclass
class MatchResult:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    decisions: int = 0
    seconds: float = 0.0
    scores: list[float] = field(default_factory=list)  # from agent A's view

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        return (self.wins + 0.5 * self.draws) / max(1, self.games)


def play_game(
    agents: Sequence[Agent], env: CabtEnv | None = None, seed: int | None = None
) -> tuple[list[float], Trajectory]:
    """Play one game; ``agents[i]`` controls seat ``i``. Returns (rewards by seat, trajectory)."""
    env = env or CabtEnv()
    for a in agents:
        a.reset()
    d: Decision | None = env.reset(seed)
    n = 0
    while d is not None and n < MAX_DECISIONS:
        d = env.step(agents[d.player].act(d))
        n += 1
    traj = env.trajectory
    assert traj is not None
    rewards = [env.reward(0), env.reward(1)] if env.done else [0.0, 0.0]
    env.close()
    return rewards, traj


def play_match(
    a: Agent, b: Agent, games: int, seed: int = 0, on_game: Callable[[int], None] | None = None
) -> MatchResult:
    """``games`` games, alternating seats (A first in even games)."""
    res = MatchResult()
    env = CabtEnv()
    t0 = time.perf_counter()
    for g in range(games):
        a_seat = g % 2
        seats = [a, b] if a_seat == 0 else [b, a]
        rewards, traj = play_game(seats, env, seed + g)
        r = rewards[a_seat]
        res.decisions += len(traj.steps)
        res.scores.append(0.5 + 0.5 * r)
        if r > 0:
            res.wins += 1
        elif r < 0:
            res.losses += 1
        else:
            res.draws += 1
        if on_game:
            on_game(g)
    res.seconds = time.perf_counter() - t0
    return res


def round_robin(agents: Sequence[Agent], games_per_pair: int, seed: int = 0) -> dict:
    names = [a.name for a in agents]
    if len(set(names)) != len(names):
        raise ValueError("agent names must be unique")
    pair: dict[tuple[str, str], MatchResult] = {}
    for i, j in itertools.combinations(range(len(agents)), 2):
        pair[(names[i], names[j])] = play_match(agents[i], agents[j], games_per_pair, seed)
    elo = fit_elo(names, {k: v.scores for k, v in pair.items()})
    return {"pairs": pair, "elo": elo, "names": names}


def format_report(rr: dict) -> str:
    lines = ["pair                          W   L   D   win%"]
    for (a, b), m in rr["pairs"].items():
        lines.append(
            f"{a + ' vs ' + b:<28} {m.wins:>3} {m.losses:>3} {m.draws:>3}  {100 * m.win_rate:5.1f}"
        )
    lines.append("")
    lines.append("Elo (anchor = first agent @ 1000)")
    for n in sorted(rr["names"], key=lambda n: -rr["elo"][n]):
        lines.append(f"  {n:<24} {rr['elo'][n]:7.1f}")
    return "\n".join(lines)
