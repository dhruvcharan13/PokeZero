import json

import pytest

from pokezero.config import load_config
from pokezero.storage import LocalStore
from pokezero.train.loop import run_training
from pokezero.train.replay import examples_from_bytes


@pytest.mark.slow
def test_one_full_iteration(tmp_path):
    cfg = load_config(
        None,
        [
            "iterations=1",
            "net.card_dim=16",
            "net.slot_dim=32",
            "net.hidden=64",
            "net.state_dim=32",
            "net.action_dim=16",
            "selfplay.games_per_iter=2",
            "train.steps_per_iter=5",
            "train.batch_size=16",
            "train.min_examples=16",
            "eval.games=2",
            "eval.baseline_games=2",
            "eval.promote_threshold=0.0",
        ],
    )
    store = LocalStore(tmp_path)
    out = run_training(cfg, store)
    assert out["trained"] and out["promoted"]
    for key in (
        "champion.pt",
        "checkpoints/iter_0000.pt",
        "checkpoints/iter_0001.pt",
        "replay/iter_0001.npz",
        "trajectories/iter_0001.jsonl",
        "metrics.jsonl",
        "config.json",
    ):
        assert store.exists(key), key
    ex = examples_from_bytes(store.get_bytes("replay/iter_0001.npz"))
    assert ex and all(abs(e.pi.sum() - 1) < 1e-4 and e.z in (-1.0, 0.0, 1.0) for e in ex)
    assert all((e.pi[~e.enc.mask] == 0).all() for e in ex)  # visit targets only on legal actions
    assert json.loads(store.get_bytes("metrics.jsonl").splitlines()[-1])["iteration"] == 1
    # resume continues from the saved champion
    out2 = run_training(cfg, store)
    assert out2["iteration"] == 2


def test_search_refuses_without_oracle_flag():
    from pokezero.search.puct import OracleDisabledError, PUCTSearch, SearchConfig

    with pytest.raises(OracleDisabledError):
        PUCTSearch(None, SearchConfig(), [1] * 60)  # type: ignore[arg-type]


def test_policy_selfplay_never_touches_engine_search(monkeypatch):
    from pokezero.env import search_engine
    from pokezero.model.net import NetConfig, PokeNet
    from pokezero.search.puct import SearchConfig
    from pokezero.train.selfplay import generate

    def boom(*a, **k):
        raise AssertionError("engine search used in fair self-play")

    monkeypatch.setattr(search_engine.EngineSearch, "__init__", boom)
    net = PokeNet(NetConfig(card_dim=16, slot_dim=32, hidden=64, state_dim=32, action_dim=16)).eval()
    res = generate(net, SearchConfig(), games=1, seed=0)
    assert res.examples and res.sims == 0 and res.engine_calls == 0
    assert all(e.pi.max() == 1.0 and e.z in (-1.0, 0.0, 1.0) for e in res.examples)
