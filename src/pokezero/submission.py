"""Build a Kaggle submission archive containing the policy-only agent (no search, no oracle).

The archive holds ``main.py`` (exposes ``agent(obs)``), the ``pokezero`` package and the weights.
UNVERIFIED: whether the competition runtime provides PyTorch. ``main.py`` falls back to the
heuristic agent if importing torch fails, so a submission always returns legal actions.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

MAIN_PY = """\
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from pokezero.env.base import validate_picks  # noqa: E402
from pokezero.env.cabt import default_deck, to_decision  # noqa: E402
from pokezero.agents.heuristic import HeuristicAgent  # noqa: E402

_DECK = default_deck()
_FALLBACK = HeuristicAgent()
_POLICY = None
try:
    import torch  # noqa: E402
    from pokezero.agents.az import PolicyAgent  # noqa: E402
    from pokezero.model.net import NetConfig, PokeNet  # noqa: E402

    _ck = torch.load(os.path.join(_HERE, "model.pt"), map_location="cpu", weights_only=True)
    _net = PokeNet(NetConfig(**_ck["net_cfg"]))
    _net.load_state_dict(_ck["state_dict"])
    _POLICY = PolicyAgent(_net.eval())
except Exception:  # missing torch / bad weights: stay legal with the heuristic
    _POLICY = None


def agent(obs):
    if obs["select"] is None:
        return list(_DECK)  # deck submission phase
    decision = to_decision(obs)
    try:
        picks = _POLICY.act(decision) if _POLICY is not None else _FALLBACK.act(decision)
        validate_picks(decision, picks)
        return picks
    except Exception:
        return _FALLBACK.act(decision)
"""


def build_submission(checkpoint: str, out: str = "submission.tar.gz") -> str:
    src_root = Path(__file__).resolve().parent
    with tarfile.open(out, "w:gz") as tar:
        data = MAIN_PY.encode()
        info = tarfile.TarInfo("main.py")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
        tar.add(checkpoint, arcname="model.pt")
        for p in sorted(src_root.rglob("*.py")):
            tar.add(p, arcname=f"pokezero/{p.relative_to(src_root)}")
    return out
