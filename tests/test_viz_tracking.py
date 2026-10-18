import pytest

from pokezero.agents import HeuristicAgent, RandomAgent
from pokezero.config import WandbConfig
from pokezero.storage import LocalStore
from pokezero.tracking import Tracker
from pokezero.viz import render_game_html


def test_replay_html_renders_through_kaggle_viewer():
    html, rewards = render_game_html(HeuristicAgent(0), RandomAgent(1))
    assert html.lstrip().lower().startswith("<!doctype html") or "<html" in html[:500].lower()
    assert len(html) > 100_000 and rewards[0] == -rewards[1]


def test_tracker_is_inert_without_wandb(tmp_path):
    t = Tracker(LocalStore(tmp_path), WandbConfig())
    t.log(1, {"a": 1.0})
    t.log_train_step({"loss": 0.1})
    t.log_checkpoint("x", b"1")
    t.log_replay(1, "<html></html>")
    assert not t.wandb_enabled
    assert b'"a": 1.0' in (tmp_path / "metrics.jsonl").read_bytes()


def test_tracker_offline_wandb(tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    t = Tracker(LocalStore(tmp_path / "s"), WandbConfig(enabled=True, mode="offline"), {"x": 1})
    t.log_train_step({"loss": 0.5})
    t.log(1, {"h2h_win_rate": 0.6, "promoted": True})
    t.log_checkpoint("m", b"weights", {"iteration": 1})
    t.finish()
    assert t.wandb_enabled
