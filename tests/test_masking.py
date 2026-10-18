import copy

import numpy as np
import torch

from pokezero.env.base import Decision
from pokezero.env.cabt import CabtEnv
from pokezero.model.encoding import encode, selection_complete
from pokezero.model.net import PokeNet, collate


def _rich_decision():
    env = CabtEnv()
    d = env.reset()
    for _ in range(30):
        if d.select_type == 0 and d.num_options > 3:
            break
        d = env.step(list(range(min(d.max_count, d.num_options))))
    return d


def test_mask_marks_options_and_stop():
    d = _rich_decision()
    e = encode(d)
    assert e.mask.shape == (d.num_options + 1,)
    assert e.mask[: d.num_options].all()
    assert not e.mask[-1]  # single-pick prompt with min_count=1: STOP illegal


def test_chosen_options_masked_and_stop_rules():
    d = Decision(0, 1, 0, 0, 2, tuple({"type": 3} for _ in range(4)), _rich_decision().obs)
    e0 = encode(d)
    assert e0.mask.all()  # min_count=0 -> STOP legal at the start
    e1 = encode(d, (2,))
    assert not e1.mask[2] and e1.mask[[0, 1, 3, 4]].all()
    assert selection_complete(d, (1, 2))
    assert not encode(d, (1, 2)).mask.any()


def test_stop_illegal_until_min_count_reached():
    d = Decision(0, 1, 0, 2, 3, tuple({"type": 3} for _ in range(4)), _rich_decision().obs)
    assert not encode(d, ()).mask[-1] and not encode(d, (0,)).mask[-1]
    assert encode(d, (0, 1)).mask[-1]


def test_illegal_actions_get_zero_probability():
    torch.manual_seed(0)
    net = PokeNet().eval()
    base = _rich_decision()
    d = Decision(base.player, base.select_type, base.context, 1, 3, base.options, base.obs)
    encs = [encode(d), encode(d, (0,))]
    logits, v = net(collate(encs))
    p = torch.softmax(logits, -1)
    for i, e in enumerate(encs):
        m = torch.from_numpy(e.mask)
        assert torch.all(p[i][: len(m)][~m] == 0)
        assert abs(p[i].sum().item() - 1.0) < 1e-5
    assert torch.all(v.abs() <= 1.0)


def test_encoder_never_reads_opponent_hidden_zones():
    d = _rich_decision()
    obs = copy.deepcopy(d.obs)
    opp = obs["current"]["players"][1 - d.player]
    opp["hand"] = [{"id": 1262, "serial": 1, "playerIndex": 1}] * 7  # would be a leak
    opp["prize"] = [{"id": 723}] * len(opp["prize"])
    leaked = Decision(d.player, d.select_type, d.context, d.min_count, d.max_count, d.options, obs)
    a, b = encode(d), encode(leaked)
    for f in ("glob", "slot_ids", "slot_feat", "hand_ids", "disc_opp", "opt_feat", "opt_card"):
        assert np.array_equal(getattr(a, f), getattr(b, f)), f


def test_hide_hand_blanks_viewer_hand():
    d = _rich_decision()
    assert encode(d).hand_ids.any()
    assert not encode(d, hide_hand=True).hand_ids.any()
