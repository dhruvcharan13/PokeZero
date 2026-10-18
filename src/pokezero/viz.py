"""Render a game with Kaggle's own replay viewer.

Games are run through the official ``kaggle_environments`` runner (not ``CabtEnv``) because the
viewer's data comes from the runner's per-game ``visualize`` payload. The viewer is a spectator
tool: it shows BOTH players' hands and prize cards. Agents still receive only their own
observation from the runner, so nothing here changes what an agent can see.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pokezero.agents.base import Agent
from pokezero.env.cabt import default_deck, to_decision


def kaggle_callable(agent: Agent) -> Callable[[Any], list[int]]:
    """Adapt a pokezero ``Agent`` to the ``agent(obs) -> list[int]`` contract of the runner."""
    deck = default_deck()

    def fn(obs: Any) -> list[int]:
        if obs["select"] is None:
            return list(deck)  # deck-submission phase
        return agent.act(to_decision(json.loads(json.dumps(obs))))

    return fn


def render_game_html(a: Agent, b: Agent) -> tuple[str, list[float]]:
    """Play one best-of-1 game (``a`` seat 0, ``b`` seat 1) and return (viewer HTML, rewards)."""
    from kaggle_environments import make

    a.reset()
    b.reset()
    env = make("cabt", configuration={"bo": 1}, debug=True)
    env.run([kaggle_callable(a), kaggle_callable(b)])
    rewards = [float(s.reward) for s in env.steps[-1]]
    return env.render(mode="html"), rewards
