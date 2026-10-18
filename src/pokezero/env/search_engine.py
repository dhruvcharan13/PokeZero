"""Forward model for tree search, backed by the engine's native ``Search*`` API.

CONFIRMED (docs/sdk-findings.md): ``SearchBegin`` rebuilds state from the observation's
``search_begin_input`` and IGNORES the opponent hidden-zone arrays we pass. The opponent's real
hand is reproduced in the search tree. Search through this class therefore sees information a
competition agent should not have. It is labelled ``ORACLE_LEAKS_HIDDEN_INFO`` and must stay
behind the explicit ``search.allow_engine_oracle`` flag; submission agents must not enable it.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

from pokezero.env.base import Decision, validate_picks
from pokezero.env.cabt import to_decision
from pokezero.env.engine import c_ints, load_lib

ORACLE_LEAKS_HIDDEN_INFO = True


class EngineSearchError(RuntimeError):
    """``SearchBegin``/``SearchStep`` refused the request (e.g. during the turn-0 setup phase,
    where the engine reports error 1 because zones are not fully dealt yet)."""


@dataclass(frozen=True)
class SearchState:
    """A node in the engine's search graph. ``decision`` is None at terminal states."""

    search_id: int
    decision: Decision | None
    obs: dict[str, Any]

    @property
    def terminal(self) -> bool:
        return self.decision is None

    def reward(self, player: int) -> float:
        res = self.obs["current"]["result"]
        if res not in (0, 1):
            return 0.0
        return 1.0 if res == player else -1.0


def _unseen_cards(deck: list[int], seen: list[int]) -> list[int]:
    pool = list(deck)
    for c in seen:
        if c in pool:
            pool.remove(c)
    return pool


def visible_card_ids(player_state: dict[str, Any]) -> list[int]:
    """Card ids a player's observation reveals for that player's side of the table."""
    ids = [c["id"] for c in player_state["discard"] if c]
    ids += [c["id"] for c in (player_state["hand"] or []) if c]
    for mon in player_state["active"] + player_state["bench"]:
        if not mon:
            continue
        ids.append(mon["id"])
        ids += [e["id"] for e in mon["energyCards"]]
        ids += [t["id"] for t in mon["tools"]]
        ids += [p["id"] for p in mon["preEvolution"]]
    return ids


class EngineSearch:
    """One search session. Call ``end()`` when done: it frees every search id issued so far."""

    def __init__(self) -> None:
        self._lib = load_lib()
        self._agent = self._lib.AgentStart()
        self.engine_calls = 0

    def begin(
        self,
        decision: Decision,
        my_deck: list[int],
        assumed_opponent_deck: list[int],
        rng: random.Random,
    ) -> SearchState:
        """Open a search at the acting player's real observation.

        Hidden zones are *determinized* from public information only (own deck minus own visible
        cards, assumed opponent deck minus opponent visible cards). The engine currently ignores
        the opponent half of this; see module docstring.
        """
        cur = decision.obs["current"]
        me, opp = cur["yourIndex"], 1 - cur["yourIndex"]
        mine, theirs = cur["players"][me], cur["players"][opp]
        my_rest = _unseen_cards(my_deck, visible_card_ids(mine))
        rng.shuffle(my_rest)
        n_deck, n_prize = mine["deckCount"], len(mine["prize"])
        md, mp = my_rest[:n_deck], my_rest[n_deck : n_deck + n_prize]
        th_rest = _unseen_cards(assumed_opponent_deck, visible_card_ids(theirs))
        rng.shuffle(th_rest)
        e_deck, e_prize = theirs["deckCount"], len(theirs["prize"])
        ed = th_rest[:e_deck]
        ep = th_rest[e_deck : e_deck + e_prize]
        eh = th_rest[e_deck + e_prize : e_deck + e_prize + theirs["handCount"]]
        payload = decision.obs["search_begin_input"].encode("ascii")
        raw = self._lib.SearchBegin(
            self._agent,
            payload,
            len(payload),
            c_ints(md),
            c_ints(mp),
            c_ints(ed),
            c_ints(ep),
            c_ints(eh),
            c_ints([]),
            0,
        )
        self.engine_calls += 1
        return self._parse(json.loads(raw))

    def step(self, state: SearchState, picks: list[int]) -> SearchState:
        """Apply picks from ``state``. Chance events are re-sampled on every call."""
        assert state.decision is not None
        validate_picks(state.decision, picks)
        raw = self._lib.SearchStep(self._agent, state.search_id, c_ints(picks), len(picks))
        self.engine_calls += 1
        return self._parse(json.loads(raw))

    def end(self) -> None:
        self._lib.SearchEnd(self._agent)

    @staticmethod
    def _parse(r: dict[str, Any]) -> SearchState:
        if r.get("error"):
            raise EngineSearchError(f"engine search error {r['error']}")
        st = r["state"]
        obs = st["observation"]
        decision = None if obs["current"]["result"] >= 0 else to_decision(obs)
        return SearchState(st["searchId"], decision, obs)
