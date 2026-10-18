"""Network-driven agents.

``PolicyAgent``  -- network priors only; no search, no hidden-information access. This is the
                    submission-safe agent.
``SearchAgent``  -- PUCT over the engine forward model. RESEARCH ONLY: the engine leaks the
                    opponent's hand into the tree (see ``pokezero.search.puct``).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from pokezero.env.base import Decision
from pokezero.env.search_engine import EngineSearchError
from pokezero.model.encoding import Encoded, encode, selection_complete
from pokezero.model.net import PokeNet
from pokezero.search.puct import PUCTSearch, SearchConfig


@dataclass
class Example:
    """One (sub-)decision: what the player saw, the MCTS target, and later the game result."""

    enc: Encoded
    pi: np.ndarray  # (n+1,) target distribution over option rows + STOP
    player: int
    z: float = 0.0


def _sample(p: np.ndarray, temperature: float, rng: random.Random) -> int:
    if temperature <= 1e-3:
        return int(np.argmax(p))
    q = np.power(p, 1.0 / temperature)
    q = q / q.sum()
    return int(rng.choices(range(len(q)), weights=q.tolist())[0])


class PolicyAgent:
    def __init__(
        self, net: PokeNet, name: str = "policy", temperature: float = 0.0, seed: int | None = None
    ):
        self.net, self.name, self.temperature = net, name, temperature
        self.rng = random.Random(seed)

    def reset(self) -> None:
        pass

    def act(self, decision: Decision) -> list[int]:
        chosen: tuple[int, ...] = ()
        n = decision.num_options
        if n == 0:
            return []
        while True:
            if selection_complete(decision, chosen):
                return list(chosen)
            enc = encode(decision, chosen)
            if enc.mask.sum() == 1:
                a = int(np.flatnonzero(enc.mask)[0])
            else:
                pr, _ = self.net.predict([enc])
                a = _sample(pr[0], self.temperature, self.rng)
            if a == n:
                return list(chosen)
            chosen += (a,)


class SearchAgent:
    """AlphaZero self-play / evaluation agent. Records ``Example`` rows when ``record`` is True."""

    def __init__(
        self,
        net: PokeNet,
        cfg: SearchConfig,
        deck: list[int],
        name: str = "puct",
        temperature: float = 1.0,
        temperature_moves: int = 12,
        record: bool = False,
        seed: int | None = None,
    ):
        self.net, self.cfg, self.deck, self.name = net, cfg, deck, name
        self.temperature, self.temperature_moves = temperature, temperature_moves
        self.record = record
        self.rng = random.Random(seed)
        self.examples: list[Example] = []
        self.search: PUCTSearch | None = None
        self.moves = 0
        self.sims = 0
        self.engine_calls = 0
        self.search_failures = 0

    def reset(self) -> None:
        if self.search is not None:
            self.search.close()
        self.search = PUCTSearch(self.net, self.cfg, self.deck, rng=self.rng)
        self.moves = 0

    def close(self) -> None:
        if self.search is not None:
            self.search.close()
            self.search = None

    def act(self, decision: Decision) -> list[int]:
        assert self.search is not None, "call reset() first"
        n = decision.num_options
        if n == 0:
            return []
        chosen: tuple[int, ...] = ()
        while not selection_complete(decision, chosen):
            enc = encode(decision, chosen)
            legal = np.flatnonzero(enc.mask)
            if len(legal) == 1:  # forced: nothing to learn or search
                a = int(legal[0])
            else:
                try:
                    res = self.search.run(decision, chosen)
                except EngineSearchError:
                    # Setup-phase states cannot be searched: play the network prior instead.
                    self.search_failures += 1
                    pr, _ = self.net.predict([enc])
                    a = _sample(pr[0], self.temperature, self.rng)
                    self.search.advance(decision, chosen, a)
                    if a == n:
                        break
                    chosen += (a,)
                    continue
                self.sims += res.sims
                self.engine_calls += res.engine_calls
                pi = res.visits / max(1.0, res.visits.sum())
                temp = self.temperature if self.moves < self.temperature_moves else 0.0
                a = _sample(pi, temp, self.rng)
                if self.record:
                    self.examples.append(Example(enc, pi, decision.player))
                self.moves += 1
            self.search.advance(decision, chosen, a)
            if a == n:
                break
            chosen += (a,)
        return list(chosen)
