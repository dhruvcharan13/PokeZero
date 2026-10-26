"""Build a Kaggle submission archive containing the policy-only agent (no search, no oracle).

Playground format (competition page): a ``.tar.gz`` with ``main.py`` at the top level and a
``deck.csv`` (60 card ids, one per line); at most 197.7 MiB; files are unpacked to
``/kaggle_simulations/agent/``. The archive holds those two files plus the ``pokezero`` package and
the weights. The grader image is Kaggle's Python image plus ``kaggle_environments``; whether it has
PyTorch is unverified, so ``main.py`` falls back to the heuristic agent if the network cannot load.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

from pokezero.env.cabt import default_deck

MAX_BYTES = int(197.7 * 2**20)

MAIN_PY = """\
import os
import sys

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # main.py may be exec'd without __file__
    _HERE = "/kaggle_simulations/agent"
sys.path.insert(0, _HERE)

from pokezero.env.base import validate_picks  # noqa: E402
from pokezero.env.cabt import default_deck, to_decision  # noqa: E402
from pokezero.agents.heuristic import HeuristicAgent  # noqa: E402


def _load_deck():
    try:
        with open(os.path.join(_HERE, "deck.csv")) as f:
            deck = [int(x) for x in f.read().split() if x.strip()]
        if len(deck) == 60:
            return deck
    except Exception:
        pass
    return default_deck()


_DECK = _load_deck()
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


def build_submission(checkpoint: str | None, out: str = "submission.tar.gz") -> str:
    src_root = Path(__file__).resolve().parent
    with tarfile.open(out, "w:gz") as tar:
        for name, content in (
            ("main.py", MAIN_PY.encode()),
            ("deck.csv", ("\n".join(map(str, default_deck())) + "\n").encode()),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
        if checkpoint is not None:  # heuristic-only baseline ships no weights
            tar.add(checkpoint, arcname="model.pt")
        for p in sorted(src_root.rglob("*.py")):
            tar.add(p, arcname=f"pokezero/{p.relative_to(src_root)}")
    size = Path(out).stat().st_size
    if size > MAX_BYTES:
        raise ValueError(f"{out} is {size / 2**20:.1f} MiB; the limit is 197.7 MiB")
    return out
