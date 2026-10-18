import numpy as np

from pokezero.agents.az import Example
from pokezero.env.cabt import CabtEnv
from pokezero.model.encoding import encode
from pokezero.train.replay import (
    ReplayBuffer,
    collate_examples,
    examples_from_bytes,
    examples_to_bytes,
)


def _examples(n=12):
    env = CabtEnv()
    d = env.reset()
    out = []
    while d is not None and len(out) < n:
        e = encode(d)
        pi = e.mask / e.mask.sum()
        out.append(Example(e, pi.astype(np.float32), d.player, float(d.player * 2 - 1)))
        d = env.step(list(range(min(d.max_count, d.num_options))))
    env.close()
    return out


def test_roundtrip_is_exact():
    ex = _examples()
    back = examples_from_bytes(examples_to_bytes(ex))
    assert len(back) == len(ex)
    for a, b in zip(ex, back, strict=True):
        for f in ("glob", "slot_ids", "slot_feat", "hand_ids", "opt_feat", "opt_card", "mask"):
            assert np.array_equal(getattr(a.enc, f), getattr(b.enc, f)), f
        assert np.allclose(a.pi, b.pi) and a.z == b.z and a.player == b.player
        assert a.enc.select_type == b.enc.select_type


def test_empty_roundtrip():
    assert examples_from_bytes(examples_to_bytes([])) == []


def test_buffer_is_bounded_and_keeps_recent():
    buf = ReplayBuffer(5)
    ex = _examples(12)
    for i, e in enumerate(ex):
        e.z = float(i)
        buf.add([e])
    assert len(buf) == 5
    assert [e.z for e in buf.examples()] == [7.0, 8.0, 9.0, 10.0, 11.0]
    buf2 = ReplayBuffer.from_bytes(buf.to_bytes(), 5)
    assert [e.z for e in buf2.examples()] == [7.0, 8.0, 9.0, 10.0, 11.0]


def test_collate_pads_targets():
    ex = _examples(6)
    batch, pi, z = collate_examples(ex)
    assert pi.shape == batch.mask.shape and z.shape[0] == 6
    assert np.allclose(pi.sum(1).numpy(), 1.0)
