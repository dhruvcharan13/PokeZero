"""Self-play game generation.

``generate`` is a pure function of (weights, config, seed): it is the unit of work a future Ray
worker would run. ``LocalExecutor`` runs it in-process today.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import torch

from pokezero.agents.az import Example, PolicyAgent, SearchAgent
from pokezero.env.base import Trajectory
from pokezero.env.cabt import CabtEnv, default_deck
from pokezero.eval.arena import play_game
from pokezero.model.net import NetConfig, PokeNet
from pokezero.search.puct import SearchConfig


@dataclass
class SelfPlayResult:
    examples: list[Example] = field(default_factory=list)
    trajectories: list[Trajectory] = field(default_factory=list)
    games: int = 0
    decisions: int = 0
    sims: int = 0
    engine_calls: int = 0
    seconds: float = 0.0
    wins: list[int] = field(default_factory=lambda: [0, 0, 0])  # seat0, seat1, draw


def state_dict_bytes(net: PokeNet) -> bytes:
    buf = io.BytesIO()
    torch.save(net.state_dict(), buf)
    return buf.getvalue()


def net_from_bytes(net_cfg: NetConfig, data: bytes) -> PokeNet:
    net = PokeNet(net_cfg)
    net.load_state_dict(torch.load(io.BytesIO(data), map_location="cpu", weights_only=True))
    net.eval()
    return net


def generate(
    net: PokeNet,
    search_cfg: SearchConfig,
    games: int,
    seed: int,
    temperature: float = 1.0,
    temperature_moves: int = 12,
    mode: str = "policy",
) -> SelfPlayResult:
    """``policy``: plain on-policy self-play, public information only (the fair default).
    ``oracle_search``: PUCT over the engine clone, which leaks the opponent's hand (research)."""
    import time

    if mode not in ("policy", "oracle_search"):
        raise ValueError(f"unknown selfplay mode {mode!r}")

    out = SelfPlayResult()
    deck = default_deck()
    env = CabtEnv(deck, deck)
    t0 = time.perf_counter()
    for g in range(games):
        agents: list = [
            PolicyAgent(net, f"sp{i}", temperature, seed * 1000 + g * 2 + i, record=True)
            if mode == "policy"
            else SearchAgent(
                net,
                search_cfg,
                deck,
                name=f"sp{i}",
                temperature=temperature,
                temperature_moves=temperature_moves,
                record=True,
                seed=seed * 1000 + g * 2 + i,
            )
            for i in range(2)
        ]
        rewards, traj = play_game(agents, env, seed=seed * 1000 + g)
        for a in agents:
            for ex in a.examples:
                ex.z = rewards[ex.player]
            out.examples += a.examples
            out.sims += getattr(a, "sims", 0)
            out.engine_calls += getattr(a, "engine_calls", 0)
            a.close()
        out.trajectories.append(traj)
        out.decisions += len(traj.steps)
        out.wins[0 if rewards[0] > 0 else 1 if rewards[1] > 0 else 2] += 1
        out.games += 1
    out.seconds = time.perf_counter() - t0
    return out


class LocalExecutor:
    """Runs self-play in-process. A Ray executor would map ``generate`` over workers instead."""

    def run(
        self,
        net: PokeNet,
        search_cfg: SearchConfig,
        games: int,
        seed: int,
        temperature: float,
        temperature_moves: int,
        mode: str = "policy",
    ) -> SelfPlayResult:
        return generate(net, search_cfg, games, seed, temperature, temperature_moves, mode)
