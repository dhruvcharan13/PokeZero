"""Rule-based baseline with a guaranteed-legal fallback.

Option type meanings are inferred from observed play (not documented by the SDK):
13 = attack (``attackId``), 14 = end turn, 8 = play a hand card onto a Pokemon (attach /
evolve), 7 = other hand action, 3/6/15 = card or energy picks inside effect prompts.
Priority on the main prompt (select type 0): play cards > strongest attack > end turn.
"""

from __future__ import annotations

import random

from pokezero.agents.random_agent import RandomAgent
from pokezero.env.base import Decision, validate_picks
from pokezero.env.cards import attack_table

MAIN_SELECT = 0
ATTACK, END_TURN = 13, 14


class HeuristicAgent:
    name = "heuristic"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self._fallback = RandomAgent(seed)
        self._attacks = attack_table()

    def reset(self) -> None:
        pass

    def act(self, decision: Decision) -> list[int]:
        try:
            picks = self._choose(decision)
            validate_picks(decision, picks)
            return picks
        except Exception:
            return self._fallback.act(decision)

    def _choose(self, d: Decision) -> list[int]:
        n = d.num_options
        if d.select_type != MAIN_SELECT:
            k = min(d.max_count, n)
            return list(range(max(d.min_count, k) if d.min_count else min(1, k)))[:k]
        # Develop the board first (attacking ends the turn); attack once the action budget is
        # spent so repeatable effects cannot stall the game.
        late = d.obs["current"]["turnActionCount"] > 25
        scored: list[tuple[float, int]] = []
        for i, op in enumerate(d.options):
            t = op["type"]
            if t == ATTACK:
                dmg = self._attacks.get(op["attackId"], {}).get("damage", 0)
                score = 500.0 + dmg if late else 50.0 + dmg / 10.0
            elif t == END_TURN:
                score = -1000.0
            else:
                score = 100.0 - t + self.rng.random() if not late else 10.0 - t
            scored.append((score, i))
        scored.sort(reverse=True)
        return [scored[0][1]]
