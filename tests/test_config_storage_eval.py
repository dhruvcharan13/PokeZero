import pytest

from pokezero.agents import HeuristicAgent, RandomAgent
from pokezero.config import apply_overrides, from_dict, load_config
from pokezero.eval.arena import play_match, round_robin
from pokezero.eval.elo import fit_elo
from pokezero.storage import LocalStore


def test_overrides_and_unknown_keys(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("iterations: 3\nselfplay: {games_per_iter: 2}\n")
    cfg = load_config(p, ["train.lr=0.5", "selfplay.search.simulations=7", "wandb.enabled=false"])
    assert cfg.iterations == 3 and cfg.train.lr == 0.5
    assert cfg.selfplay.games_per_iter == 2 and cfg.selfplay.search.simulations == 7
    assert cfg.wandb.enabled is False  # W&B off unless asked for
    with pytest.raises(KeyError):
        from_dict(type(cfg), {"itertions": 1})
    with pytest.raises(ValueError):
        apply_overrides({}, ["novalue"])


def test_local_store_roundtrip_and_traversal(tmp_path):
    s = LocalStore(tmp_path / "root")
    s.put_bytes("a/b.bin", b"x")
    assert s.get_bytes("a/b.bin") == b"x" and s.exists("a/b.bin")
    assert s.list("a") == ["a/b.bin"]
    with pytest.raises(ValueError):
        s.put_bytes("../escape", b"no")


def test_elo_orders_players():
    res = {
        ("a", "b"): [1.0] * 8 + [0.0] * 2,
        ("b", "c"): [1.0] * 8 + [0.0] * 2,
        ("a", "c"): [1.0] * 10,
    }
    elo = fit_elo(["a", "b", "c"], res)
    assert elo["a"] > elo["b"] > elo["c"]


def test_heuristic_beats_random():
    m = play_match(HeuristicAgent(0), RandomAgent(0), 30, seed=0)
    assert m.win_rate > 0.7


def test_round_robin_report_has_all_agents():
    rr = round_robin([RandomAgent(0), HeuristicAgent(0)], 4)
    assert set(rr["elo"]) == {"random", "heuristic"}
