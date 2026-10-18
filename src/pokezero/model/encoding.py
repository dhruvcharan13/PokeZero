"""Observation / action encoding.

``encode`` turns one pending selection into numpy arrays. It reads **only** what the acting
player's observation contains: its own hand, public board, discard piles and the *counts* of
hidden zones. Opponent hand and prize contents are never touched (they are ``None`` / null in
real observations; ``hide_hand`` additionally blanks the viewer's own hand when we evaluate
opponent nodes inside a search tree, so the opponent policy is public-information-only).

Multi-pick prompts (``max_count > 1``) are decomposed into sequential single picks. The action
space of one step is ``num_options + 1``: the options, plus a final STOP row that is legal once
``min_count`` picks exist and fewer than ``max_count`` have been made.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pokezero.env.base import Decision

N_SLOTS = 9  # active + up to 8 bench per side
MAX_HAND, MAX_DISCARD = 30, 60
N_ENERGY_TYPES = 13
SLOT_FEAT = N_ENERGY_TYPES + 6
GLOBAL_FEAT = 8 + 2 * 11
N_OPT_TYPES, N_AREAS, N_CTX = 17, 10, 64
OPT_FEAT = N_OPT_TYPES + N_AREAS + N_AREAS + 7
STOP_TYPE = 16


@dataclass
class Encoded:
    glob: np.ndarray
    slot_ids: np.ndarray  # (2*N_SLOTS,) int64, 0 = empty
    slot_feat: np.ndarray  # (2*N_SLOTS, SLOT_FEAT)
    hand_ids: np.ndarray  # (MAX_HAND,)
    disc_me: np.ndarray  # (MAX_DISCARD,)
    disc_opp: np.ndarray  # (MAX_DISCARD,)
    stadium_id: int
    select_type: int
    context: int
    opt_feat: np.ndarray  # (n+1, OPT_FEAT)
    opt_card: np.ndarray  # (n+1,)
    opt_attack: np.ndarray  # (n+1,)
    mask: np.ndarray  # (n+1,) bool legal
    n_remaining: float


def _pad(ids: list[int], size: int) -> np.ndarray:
    out = np.zeros(size, dtype=np.int64)
    ids = ids[:size]
    out[: len(ids)] = ids
    return out


def _slot(mon: dict[str, Any]) -> tuple[int, np.ndarray]:
    f = np.zeros(SLOT_FEAT, dtype=np.float32)
    for e in mon["energies"]:
        f[min(int(e), N_ENERGY_TYPES - 1)] += 1.0
    k = N_ENERGY_TYPES
    f[k] = mon["hp"] / 350.0
    f[k + 1] = mon["maxHp"] / 350.0
    f[k + 2] = float(mon["appearThisTurn"])
    f[k + 3] = len(mon["tools"]) / 2.0
    f[k + 4] = len(mon["preEvolution"]) / 2.0
    f[k + 5] = len(mon["energies"]) / 6.0
    return mon["id"], f


def _player_scalars(p: dict[str, Any]) -> list[float]:
    return [
        p["deckCount"] / 60.0,
        p["handCount"] / 20.0,
        len(p["prize"]) / 6.0,
        len(p["discard"]) / 60.0,
        p["benchMax"] / 8.0,
        float(p["poisoned"]),
        float(p["burned"]),
        float(p["asleep"]),
        float(p["paralyzed"]),
        float(p["confused"]),
        p["win"] / 3.0,
    ]


def _resolve_card(
    op: dict[str, Any], me: int, players: list[dict[str, Any]], hand: list[int]
) -> int:
    """Best-effort card behind an option (area codes inferred empirically, see sdk-findings)."""
    if "cardId" in op:
        return int(op["cardId"])
    area, idx = op.get("area"), op.get("index", 0)
    pi = op.get("playerIndex", me)
    if area == 2 and pi == me and 0 <= idx < len(hand):
        return hand[idx]
    if area in (4, 5) and 0 <= pi < 2:
        p = players[pi]
        zone = p["active"] if area == 4 else p["bench"]
        if 0 <= idx < len(zone):
            return int(zone[idx]["id"])
    return 0


def encode(decision: Decision, chosen: tuple[int, ...] = (), hide_hand: bool = False) -> Encoded:
    obs = decision.obs
    cur = obs["current"]
    me = decision.player
    players = cur["players"]
    mine, theirs = players[me], players[1 - me]
    hand_cards = [] if hide_hand else [c["id"] for c in (mine["hand"] or []) if c]

    glob = np.array(
        [
            cur["turn"] / 50.0,
            cur["turnActionCount"] / 50.0,
            float(cur["supporterPlayed"]),
            float(cur["stadiumPlayed"]),
            float(cur["energyAttached"]),
            float(cur["retreated"]),
            float(cur["firstPlayer"] == me),
            float(bool(cur["stadium"])),
            *_player_scalars(mine),
            *_player_scalars(theirs),
        ],
        dtype=np.float32,
    )
    slot_ids = np.zeros(2 * N_SLOTS, dtype=np.int64)
    slot_feat = np.zeros((2 * N_SLOTS, SLOT_FEAT), dtype=np.float32)
    for side, p in enumerate((mine, theirs)):
        # Face-down setup Pokemon show up as null entries; they carry no information.
        for j, mon in enumerate((p["active"] + p["bench"])[:N_SLOTS]):
            if mon is None or mon.get("energies") is None:
                continue
            cid, f = _slot(mon)
            slot_ids[side * N_SLOTS + j] = cid
            slot_feat[side * N_SLOTS + j] = f

    n = decision.num_options
    opt_feat = np.zeros((n + 1, OPT_FEAT), dtype=np.float32)
    opt_card = np.zeros(n + 1, dtype=np.int64)
    opt_attack = np.zeros(n + 1, dtype=np.int64)
    mask = np.zeros(n + 1, dtype=bool)
    full_hand = [c["id"] for c in (mine["hand"] or []) if c]  # option -> card resolution only
    for i, op in enumerate(decision.options):
        row = opt_feat[i]
        row[min(op["type"], N_OPT_TYPES - 1)] = 1.0
        o = N_OPT_TYPES
        if "area" in op:
            row[o + min(op["area"], N_AREAS - 1)] = 1.0
        o += N_AREAS
        if "inPlayArea" in op:
            row[o + min(op["inPlayArea"], N_AREAS - 1)] = 1.0
        o += N_AREAS
        row[o] = op.get("index", 0) / 10.0
        row[o + 1] = op.get("inPlayIndex", 0) / 10.0
        row[o + 2] = float(op.get("playerIndex", me) == me)
        row[o + 3] = op.get("count", 0) / 5.0
        row[o + 4] = op.get("energyIndex", 0) / 10.0
        row[o + 5] = float(i in chosen)
        opt_card[i] = _resolve_card(op, me, players, full_hand)
        opt_attack[i] = op.get("attackId", 0)
        mask[i] = i not in chosen
    cap = min(decision.max_count, n)
    stop_ok = len(chosen) >= decision.min_count and len(chosen) < cap
    opt_feat[n, STOP_TYPE] = 1.0
    mask[n] = stop_ok
    if len(chosen) >= cap:
        mask[:] = False  # caller should have finished the selection already
    opt_feat[:, OPT_FEAT - 1] = (cap - len(chosen)) / 5.0

    st = cur["stadium"]
    return Encoded(
        glob=glob,
        slot_ids=slot_ids,
        slot_feat=slot_feat,
        hand_ids=_pad(hand_cards, MAX_HAND),
        disc_me=_pad([c["id"] for c in mine["discard"] if c], MAX_DISCARD),
        disc_opp=_pad([c["id"] for c in theirs["discard"] if c], MAX_DISCARD),
        stadium_id=int(st[0]["id"]) if st else 0,
        select_type=min(decision.select_type, N_CTX - 1),
        context=min(decision.context, N_CTX - 1),
        opt_feat=opt_feat,
        opt_card=opt_card,
        opt_attack=opt_attack,
        mask=mask,
        n_remaining=float(cap - len(chosen)),
    )


def selection_complete(decision: Decision, chosen: tuple[int, ...]) -> bool:
    return len(chosen) >= min(decision.max_count, decision.num_options)
