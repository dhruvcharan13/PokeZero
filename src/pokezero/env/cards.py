"""Static card / attack tables exported by the native engine (``AllCard`` / ``AllAttack``)."""

from __future__ import annotations

import ctypes
import json
from functools import lru_cache
from typing import Any

import numpy as np

from pokezero.env.engine import load_lib

CARD_FEATURE_DIM = 12


@lru_cache(maxsize=1)
def card_table() -> dict[int, dict[str, Any]]:
    lib = load_lib()
    lib.AllCard.restype = ctypes.c_char_p
    lib.AllCard.argtypes = []
    return {c["cardId"]: c for c in json.loads(lib.AllCard().decode())}


@lru_cache(maxsize=1)
def attack_table() -> dict[int, dict[str, Any]]:
    lib = load_lib()
    lib.AllAttack.restype = ctypes.c_char_p
    lib.AllAttack.argtypes = []
    return {a["attackId"]: a for a in json.loads(lib.AllAttack().decode())}


def vocab_size() -> int:
    """Embedding table size: ids are used directly; index 0 is padding (no card has id 0)."""
    return max(card_table()) + 1


def attack_vocab_size() -> int:
    return max(attack_table()) + 1


@lru_cache(maxsize=1)
def card_feature_matrix() -> np.ndarray:
    """Dense numeric features per card id (row 0 = padding). Shape (vocab, CARD_FEATURE_DIM)."""
    m = np.zeros((vocab_size(), CARD_FEATURE_DIM), dtype=np.float32)
    atk = attack_table()
    for cid, c in card_table().items():
        best = 0
        for a in c.get("attacks") or []:
            aid = a if isinstance(a, int) else a.get("attackId")
            if aid in atk:
                best = max(best, atk[aid]["damage"])
        m[cid] = [
            c["cardType"] / 8.0,
            c["hp"] / 350.0,
            c["retreatCost"] / 4.0,
            float(c["basic"]),
            float(c["stage1"]),
            float(c["stage2"]),
            float(c["ex"] or c["megaEx"]),
            float(c["tera"]),
            float(c["aceSpec"]),
            c["energyType"] / 12.0,
            c["pokemonType"] / 12.0,
            best / 350.0,
        ]
    return m
