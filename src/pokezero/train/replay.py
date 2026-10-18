"""Bounded replay buffer with a dependency-free (no pickle) ``.npz`` serialization."""

from __future__ import annotations

import io
from collections import deque
from collections.abc import Sequence

import numpy as np
import torch

from pokezero.agents.az import Example
from pokezero.model.encoding import Encoded
from pokezero.model.net import Batch, collate

_FIXED = ["glob", "slot_ids", "slot_feat", "hand_ids", "disc_me", "disc_opp"]
_RAGGED = ["opt_feat", "opt_card", "opt_attack", "mask"]
_SCALAR = ["stadium_id", "select_type", "context", "n_remaining"]


class ReplayBuffer:
    """FIFO window over the most recent ``max_examples`` examples."""

    def __init__(self, max_examples: int):
        self.max_examples = max_examples
        self._data: deque[Example] = deque(maxlen=max_examples)

    def __len__(self) -> int:
        return len(self._data)

    def add(self, examples: Sequence[Example]) -> None:
        self._data.extend(examples)

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[Example]:
        idx = rng.integers(0, len(self._data), size=batch_size)
        return [self._data[int(i)] for i in idx]

    def examples(self) -> list[Example]:
        return list(self._data)

    # -- serialization ---------------------------------------------------------------------
    def to_bytes(self) -> bytes:
        return examples_to_bytes(list(self._data))

    @classmethod
    def from_bytes(cls, data: bytes, max_examples: int) -> ReplayBuffer:
        buf = cls(max_examples)
        buf.add(examples_from_bytes(data))
        return buf


def examples_to_bytes(examples: Sequence[Example]) -> bytes:
    out: dict[str, np.ndarray] = {}
    for f in _FIXED:
        out[f] = np.stack([getattr(e.enc, f) for e in examples]) if examples else np.zeros(0)
    for f in _SCALAR:
        out[f] = np.array([getattr(e.enc, f) for e in examples], dtype=np.float64)
    lens = np.array([e.enc.mask.shape[0] for e in examples], dtype=np.int64)
    out["lens"] = lens
    for f in _RAGGED:
        out[f] = np.concatenate([getattr(e.enc, f) for e in examples]) if examples else np.zeros(0)
    out["pi"] = np.concatenate([e.pi for e in examples]) if examples else np.zeros(0, np.float32)
    out["z"] = np.array([e.z for e in examples], dtype=np.float32)
    out["player"] = np.array([e.player for e in examples], dtype=np.int64)
    buf = io.BytesIO()
    np.savez_compressed(buf, **out)  # type: ignore[arg-type]
    return buf.getvalue()


def examples_from_bytes(data: bytes) -> list[Example]:
    z = np.load(io.BytesIO(data), allow_pickle=False)
    lens = z["lens"]
    offs = np.concatenate([[0], np.cumsum(lens)])
    res: list[Example] = []
    for i in range(len(lens)):
        s, e = int(offs[i]), int(offs[i + 1])
        enc = Encoded(
            glob=z["glob"][i],
            slot_ids=z["slot_ids"][i],
            slot_feat=z["slot_feat"][i],
            hand_ids=z["hand_ids"][i],
            disc_me=z["disc_me"][i],
            disc_opp=z["disc_opp"][i],
            stadium_id=int(z["stadium_id"][i]),
            select_type=int(z["select_type"][i]),
            context=int(z["context"][i]),
            opt_feat=z["opt_feat"][s:e],
            opt_card=z["opt_card"][s:e],
            opt_attack=z["opt_attack"][s:e],
            mask=z["mask"][s:e].astype(bool),
            n_remaining=float(z["n_remaining"][i]),
        )
        res.append(Example(enc, z["pi"][s:e], int(z["player"][i]), float(z["z"][i])))
    return res


def collate_examples(examples: Sequence[Example]) -> tuple[Batch, torch.Tensor, torch.Tensor]:
    batch = collate([e.enc for e in examples])
    a_max = batch.mask.shape[1]
    pi = np.zeros((len(examples), a_max), dtype=np.float32)
    for i, e in enumerate(examples):
        pi[i, : len(e.pi)] = e.pi
    z = torch.tensor([e.z for e in examples], dtype=torch.float32)
    return batch, torch.from_numpy(pi), z
