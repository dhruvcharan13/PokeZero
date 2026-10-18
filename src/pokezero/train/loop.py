"""The AlphaZero outer loop: self-play -> replay -> train -> evaluate -> (maybe) promote."""

from __future__ import annotations

import copy
import json
import random
import time
from typing import Any

import numpy as np
import torch

from pokezero.agents import HeuristicAgent, RandomAgent
from pokezero.agents.az import PolicyAgent, SearchAgent
from pokezero.config import Config, to_dict
from pokezero.env.cabt import default_deck
from pokezero.eval.arena import play_match
from pokezero.model.net import PokeNet
from pokezero.search.puct import SearchConfig
from pokezero.storage import ArtifactStore, load_checkpoint, save_checkpoint
from pokezero.tracking import Tracker
from pokezero.train.learner import train_steps
from pokezero.train.replay import ReplayBuffer, examples_to_bytes
from pokezero.train.selfplay import LocalExecutor


def _checkpoint(
    net: PokeNet, cfg: Config, iteration: int, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "state_dict": net.state_dict(),
        "net_cfg": to_dict(cfg)["net"],
        "iteration": iteration,
        **(extra or {}),
    }


def candidate_vs_champion(
    cand: PokeNet, champ: PokeNet, cfg: Config, seed: int
) -> dict[str, float]:
    ev = cfg.eval
    if ev.use_search:
        sc = SearchConfig(
            simulations=ev.search_simulations, dirichlet_eps=0.0, allow_engine_oracle=True
        )
        deck = default_deck()
        a: Any = SearchAgent(cand, sc, deck, name="candidate", temperature=0.0, seed=seed)
        b: Any = SearchAgent(champ, sc, deck, name="champion", temperature=0.0, seed=seed + 1)
    else:
        a = PolicyAgent(cand, "candidate", seed=seed)
        b = PolicyAgent(champ, "champion", seed=seed + 1)
    # Greedy policies are deterministic, so seat-swapped games only differ through engine
    # randomness; that is what makes the game count meaningful.
    m = play_match(a, b, ev.games, seed)
    for ag in (a, b):
        if hasattr(ag, "close"):
            ag.close()
    return {"win_rate": m.win_rate, "wins": m.wins, "losses": m.losses, "draws": m.draws}


def run_training(cfg: Config, store: ArtifactStore) -> dict[str, Any]:
    random.seed(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    tracker = Tracker(store, cfg.wandb, to_dict(cfg))
    store.put_bytes("config.json", json.dumps(to_dict(cfg), indent=2).encode())

    if store.exists("champion.pt"):  # resume
        ck = load_checkpoint(store, "champion.pt")
        start_iter = int(ck["iteration"]) + 1
        champion = PokeNet(cfg.net)
        champion.load_state_dict(ck["state_dict"])
        champion.eval()
    else:
        start_iter = 1
        champion = PokeNet(cfg.net).eval()
        save_checkpoint(store, "champion.pt", _checkpoint(champion, cfg, 0))
        save_checkpoint(store, "checkpoints/iter_0000.pt", _checkpoint(champion, cfg, 0))
    learner = copy.deepcopy(champion)
    opt = torch.optim.AdamW(
        learner.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    buf = ReplayBuffer(cfg.replay.max_examples)
    executor = LocalExecutor()
    last: dict[str, Any] = {}
    sp = cfg.selfplay

    for it in range(start_iter, start_iter + cfg.iterations):
        t0 = time.perf_counter()
        # 1. self-play with the current champion
        res = executor.run(
            champion,
            sp.search,
            sp.games_per_iter,
            cfg.seed * 7919 + it,
            sp.temperature,
            sp.temperature_moves,
        )
        buf.add(res.examples)
        store.put_bytes(f"replay/iter_{it:04d}.npz", examples_to_bytes(res.examples))
        store.put_bytes(
            f"trajectories/iter_{it:04d}.jsonl",
            "\n".join(json.dumps(t.to_json()) for t in res.trajectories).encode(),
        )
        # 2. train candidate
        losses: dict[str, float] = {"policy_loss": float("nan"), "value_loss": float("nan")}
        trained = len(buf) >= cfg.train.min_examples
        if trained:
            losses = train_steps(learner, opt, buf, cfg.train, np_rng)
        # 3. evaluate candidate vs champion, and both vs baselines
        h2h = candidate_vs_champion(learner, champion, cfg, cfg.seed + it)
        promoted = trained and h2h["win_rate"] >= cfg.eval.promote_threshold
        base = {}
        for name, opp in (("random", RandomAgent(it)), ("heuristic", HeuristicAgent(it))):
            m = play_match(
                PolicyAgent(learner, "candidate"), opp, cfg.eval.baseline_games, cfg.seed + it
            )
            base[f"candidate_vs_{name}"] = m.win_rate
        # 4. checkpoint (+ promote)
        save_checkpoint(
            store,
            f"checkpoints/iter_{it:04d}.pt",
            _checkpoint(learner, cfg, it, {"promoted": promoted}),
        )
        if promoted:
            champion = copy.deepcopy(learner).eval()
            save_checkpoint(store, "champion.pt", _checkpoint(champion, cfg, it))
        last = {
            "iteration": it,
            "promoted": promoted,
            "replay_size": len(buf),
            "selfplay_games": res.games,
            "selfplay_examples": len(res.examples),
            "selfplay_decisions": res.decisions,
            "selfplay_sims_per_s": res.sims / max(res.seconds, 1e-9),
            "selfplay_seconds": res.seconds,
            "selfplay_seat0_wins": res.wins[0],
            "h2h_win_rate": h2h["win_rate"],
            **losses,
            **base,
            "iter_seconds": time.perf_counter() - t0,
            "trained": trained,
        }
        tracker.log(it, last)
        print(json.dumps(last))
    tracker.finish()
    return last
