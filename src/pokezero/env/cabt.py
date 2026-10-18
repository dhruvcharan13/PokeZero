"""Adapter for the competition engine (``kaggle_environments.envs.cabt``).

Confirmed engine behaviour this adapter relies on (see docs/sdk-findings.md):

* One ``Select`` call answers one pending selection with a list of option indices.
* The engine has no seed argument. ``seed`` therefore only seeds *our* Python RNG; games are
  made reproducible by recording and replaying picks (``Trajectory``), not by re-simulation.
* ``search_begin_input`` in each observation is an opaque payload that ``SearchBegin`` turns
  into a cloneable search state.
"""

from __future__ import annotations

import copy
import ctypes
import json
import random
from typing import Any

from pokezero.env.base import Decision, Trajectory, TrajectoryStep, validate_picks
from pokezero.env.engine import c_ints, load_lib


class _StartData(ctypes.Structure):
    _fields_ = [
        ("battlePtr", ctypes.c_void_p),
        ("errorPlayer", ctypes.c_int),
        ("errorType", ctypes.c_int),
    ]


class _SerialData(ctypes.Structure):
    _fields_ = [
        ("json", ctypes.c_char_p),
        ("data", ctypes.POINTER(ctypes.c_ubyte)),
        ("count", ctypes.c_int),
        ("selectPlayer", ctypes.c_int),
    ]


def default_deck() -> list[int]:
    """The 60-card deck the official starter agents submit."""
    from kaggle_environments.envs.cabt import cabt

    return list(cabt.deck)


def to_decision(obs: dict[str, Any]) -> Decision:
    sel = obs["select"]
    return Decision(
        player=obs["current"]["yourIndex"],
        select_type=sel["type"],
        context=sel["context"],
        min_count=sel["minCount"],
        max_count=sel["maxCount"],
        options=tuple(sel["option"]),
        obs=obs,
    )


class CabtEnv:
    """One local game. ``reset`` -> ``Decision``; ``step(picks)`` -> next ``Decision`` or None."""

    def __init__(self, deck0: list[int] | None = None, deck1: list[int] | None = None):
        d = default_deck()
        self.decks = [list(deck0 or d), list(deck1 or d)]
        self._lib = load_lib()
        self._lib.BattleStart.restype = _StartData
        self._lib.BattleStart.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self._lib.BattleFinish.argtypes = [ctypes.c_void_p]
        self._lib.GetBattleData.restype = _SerialData
        self._lib.GetBattleData.argtypes = [ctypes.c_void_p]
        self._lib.Select.restype = ctypes.c_int
        self._lib.Select.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self._ptr: int | None = None
        self._obs: dict[str, Any] | None = None
        self._decision: Decision | None = None
        self.trajectory: Trajectory | None = None

    # -- lifecycle -----------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> Decision:
        self.close()
        if seed is not None:
            random.seed(seed)  # engine itself cannot be seeded
        cards = self.decks[0] + self.decks[1]
        start = self._lib.BattleStart((ctypes.c_int * len(cards))(*cards))
        if not start.battlePtr:
            raise RuntimeError(f"deck rejected by engine: player={start.errorPlayer}")
        self._ptr = start.battlePtr
        self.trajectory = Trajectory(decks=copy.deepcopy(self.decks), seed=seed)
        self._read()
        assert self._decision is not None
        return self._decision

    def close(self) -> None:
        if self._ptr:
            self._lib.BattleFinish(self._ptr)
        self._ptr = None

    def __del__(self) -> None:  # best effort: engine memory is manually managed
        try:
            self.close()
        except Exception:
            pass

    # -- stepping ------------------------------------------------------------------------
    def _read(self) -> None:
        sd = self._lib.GetBattleData(self._ptr)
        obs = json.loads(sd.json.decode())
        obs["search_begin_input"] = ctypes.string_at(sd.data, sd.count).decode("ascii")
        self._obs = obs
        res = obs["current"]["result"]
        if res >= 0:
            self._decision = None
            if self.trajectory is not None:
                self.trajectory.result = res if res in (0, 1) else 2
        else:
            self._decision = to_decision(obs)

    def step(self, picks: list[int]) -> Decision | None:
        if self._decision is None or self._ptr is None:
            raise RuntimeError("game is over; call reset()")
        validate_picks(self._decision, picks)
        d = self._decision
        err = self._lib.Select(self._ptr, c_ints(picks), len(picks))
        if err != 0:
            raise RuntimeError(f"engine rejected picks {picks} (error {err})")
        assert self.trajectory is not None
        self.trajectory.steps.append(
            TrajectoryStep(d.player, list(picks), d.num_options, d.select_type)
        )
        self._read()
        return self._decision

    @property
    def done(self) -> bool:
        return self._decision is None

    @property
    def decision(self) -> Decision | None:
        return self._decision

    def reward(self, player: int) -> float:
        assert self._obs is not None and self.done
        res = self._obs["current"]["result"]
        if res not in (0, 1):
            return 0.0
        return 1.0 if res == player else -1.0
