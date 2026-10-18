"""Thin ctypes layer over the engine shipped inside ``kaggle_environments`` (``libcg``).

``kaggle_environments.envs.cabt.cg.game`` keeps the active battle in a class attribute, so only
one battle can be live per process. We re-bind the same C entry points here and keep the
battle pointer on the instance instead. The search entry points (``SearchBegin`` etc.) are
exported by the library but have no ctypes signatures upstream; they are declared here.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache

_INTP = ctypes.POINTER(ctypes.c_int)


@lru_cache(maxsize=1)
def load_lib() -> ctypes.CDLL:
    from kaggle_environments.envs.cabt.cg.sim import lib  # imports + GameInitialize()

    lib.AgentStart.restype = ctypes.c_void_p
    lib.AgentStart.argtypes = []
    lib.SearchBegin.restype = ctypes.c_char_p
    lib.SearchBegin.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int,
        _INTP,
        _INTP,
        _INTP,
        _INTP,
        _INTP,
        _INTP,
        ctypes.c_int,
    ]
    lib.SearchStep.restype = ctypes.c_char_p
    lib.SearchStep.argtypes = [ctypes.c_void_p, ctypes.c_int64, _INTP, ctypes.c_int]
    lib.SearchEnd.restype = None
    lib.SearchEnd.argtypes = [ctypes.c_void_p]
    lib.SearchRelease.restype = None
    lib.SearchRelease.argtypes = [ctypes.c_void_p, ctypes.c_int64]
    return lib  # type: ignore[no-any-return]


def c_ints(values: list[int]) -> ctypes.Array[ctypes.c_int]:
    return (ctypes.c_int * max(1, len(values)))(*values)
