from __future__ import annotations

import random

from pokezero.env.base import Decision


class RandomAgent:
    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def reset(self) -> None:
        pass

    def act(self, decision: Decision) -> list[int]:
        hi = min(decision.max_count, decision.num_options)
        k = self.rng.randint(decision.min_count, hi) if hi >= decision.min_count else 0
        return self.rng.sample(range(decision.num_options), k)
