import importlib
import sys
import tarfile

import torch

from pokezero.env.cabt import CabtEnv
from pokezero.model.net import NetConfig, PokeNet
from pokezero.submission import build_submission


def test_submission_archive_plays_legal_game(tmp_path):
    cfg = NetConfig(card_dim=16, slot_dim=32, hidden=64, state_dim=32, action_dim=16)
    ck = tmp_path / "m.pt"
    torch.save({"state_dict": PokeNet(cfg).state_dict(), "net_cfg": cfg.__dict__}, ck)
    out = build_submission(str(ck), str(tmp_path / "s.tar.gz"))
    with tarfile.open(out) as t:
        names = t.getnames()
        t.extractall(tmp_path / "x", filter="data")
    assert "main.py" in names and "model.pt" in names and "deck.csv" in names
    deck = (tmp_path / "x" / "deck.csv").read_text().split()
    assert len(deck) == 60 and all(x.isdigit() for x in deck)
    main_src = (tmp_path / "x" / "main.py").read_text()
    assert "puct" not in main_src and "SearchAgent" not in main_src  # no oracle in submissions
    sys.path.insert(0, str(tmp_path / "x"))
    try:
        sys.modules.pop("main", None)
        main = importlib.import_module("main")
        assert len(main.agent({"select": None})) == 60
        env = CabtEnv()
        d = env.reset()
        while d is not None:
            d = env.step(main.agent(d.obs))
        assert env.done
    finally:
        sys.path.remove(str(tmp_path / "x"))
        sys.modules.pop("main", None)
