"""Command line: ``pokezero {train,eval,bench,package}``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from pokezero.config import load_config
from pokezero.storage import LocalStore


def _load_net(path: str):  # type: ignore[no-untyped-def]
    from pokezero.model.net import NetConfig, PokeNet

    ck = torch.load(path, map_location="cpu", weights_only=True)
    net = PokeNet(NetConfig(**ck["net_cfg"]))
    net.load_state_dict(ck["state_dict"])
    return net.eval()


def cmd_train(args: argparse.Namespace) -> None:
    from pokezero.train.loop import run_training

    cfg = load_config(args.config, args.overrides)
    store = LocalStore(Path(cfg.artifact_root) / cfg.run_name)
    run_training(cfg, store)


def cmd_eval(args: argparse.Namespace) -> None:
    from pokezero.agents import Agent, HeuristicAgent, RandomAgent
    from pokezero.agents.az import PolicyAgent
    from pokezero.eval.arena import format_report, round_robin

    agents: list[Agent] = [RandomAgent(0), HeuristicAgent(0)]
    for p in args.checkpoint or []:
        agents.append(PolicyAgent(_load_net(p), name=Path(p).stem))
    rr = round_robin(agents, args.games)
    print(format_report(rr))


def cmd_bench(args: argparse.Namespace) -> None:
    from pokezero.agents import HeuristicAgent, RandomAgent
    from pokezero.agents.az import PolicyAgent, SearchAgent
    from pokezero.env.cabt import default_deck
    from pokezero.eval.arena import play_match
    from pokezero.model.net import PokeNet
    from pokezero.search.puct import SearchConfig

    out: dict[str, float] = {}
    m = play_match(RandomAgent(0), RandomAgent(1), args.games)
    out["random_vs_random_games_per_s"] = m.games / m.seconds
    out["random_vs_random_decisions_per_s"] = m.decisions / m.seconds
    m = play_match(HeuristicAgent(0), RandomAgent(1), args.games)
    out["heuristic_vs_random_win_rate"] = m.win_rate
    net = _load_net(args.checkpoint) if args.checkpoint else PokeNet().eval()
    m = play_match(PolicyAgent(net), RandomAgent(1), max(2, args.games // 4))
    out["policy_vs_random_win_rate"] = m.win_rate
    out["policy_games_per_s"] = m.games / m.seconds
    sc = SearchConfig(simulations=args.sims, allow_engine_oracle=True)
    agent = SearchAgent(net, sc, default_deck(), seed=0)
    t = time.perf_counter()
    m = play_match(agent, RandomAgent(1), 2)
    dt = time.perf_counter() - t
    out["search_sims_per_s"] = agent.sims / dt
    out["search_engine_calls_per_s"] = agent.engine_calls / dt
    agent.close()
    print(json.dumps(out, indent=2))


def _make_agent(spec: str, seed: int = 0):  # type: ignore[no-untyped-def]
    """``random`` | ``heuristic`` | path to a checkpoint (policy-only agent)."""
    from pokezero.agents import HeuristicAgent, RandomAgent
    from pokezero.agents.az import PolicyAgent

    if spec == "random":
        return RandomAgent(seed)
    if spec == "heuristic":
        return HeuristicAgent(seed)
    return PolicyAgent(_load_net(spec), name=Path(spec).stem)


def cmd_replay(args: argparse.Namespace) -> None:
    from pokezero.viz import render_game_html

    html, rewards = render_game_html(_make_agent(args.a, 0), _make_agent(args.b, 1))
    Path(args.out).write_text(html)
    print(f"wrote {args.out} (seat0 reward {rewards[0]:+.0f}); open it in a browser")


def cmd_package(args: argparse.Namespace) -> None:
    from pokezero.submission import build_submission

    print(build_submission(args.checkpoint, args.out))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="pokezero")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="run the AlphaZero loop")
    t.add_argument("--config", default=None)
    t.add_argument("overrides", nargs="*", help="a.b=value config overrides")
    t.set_defaults(fn=cmd_train)
    e = sub.add_parser("eval", help="round-robin win rate + Elo")
    e.add_argument("--games", type=int, default=20)
    e.add_argument("--checkpoint", action="append")
    e.set_defaults(fn=cmd_eval)
    b = sub.add_parser("bench", help="games/s, search throughput, baseline win rates")
    b.add_argument("--games", type=int, default=40)
    b.add_argument("--sims", type=int, default=32)
    b.add_argument("--checkpoint", default=None)
    b.set_defaults(fn=cmd_bench)
    r = sub.add_parser("replay", help="render one game in Kaggle's replay viewer (HTML)")
    r.add_argument("--a", default="heuristic", help="seat 0: random | heuristic | checkpoint path")
    r.add_argument("--b", default="random", help="seat 1: random | heuristic | checkpoint path")
    r.add_argument("--out", default="replay.html")
    r.set_defaults(fn=cmd_replay)
    k = sub.add_parser("package", help="build a Kaggle submission archive (policy-only agent)")
    k.add_argument("--checkpoint", default=None, help="omit for the heuristic-only baseline")
    k.add_argument("--out", default="submission.tar.gz")
    k.set_defaults(fn=cmd_package)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main(sys.argv[1:])
