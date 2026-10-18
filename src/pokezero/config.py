"""Typed YAML configuration: dataclasses + ``key.sub=value`` command-line overrides."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

from pokezero.model.net import NetConfig
from pokezero.search.puct import SearchConfig
from pokezero.train.learner import TrainConfig

T = TypeVar("T")


@dataclass
class SelfPlayConfig:
    games_per_iter: int = 4
    temperature: float = 1.0
    temperature_moves: int = 12
    search: SearchConfig = field(
        default_factory=lambda: SearchConfig(simulations=16, allow_engine_oracle=True)
    )


@dataclass
class ReplayConfig:
    max_examples: int = 20000  # bounded recent window


@dataclass
class EvalConfig:
    games: int = 20  # candidate vs champion
    promote_threshold: float = 0.55  # win rate (draw = 0.5) needed to replace the champion
    use_search: bool = False  # False = greedy policy only (what a submission can run)
    search_simulations: int = 16
    baseline_games: int = 10


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "pokezero"
    entity: str = ""
    run_name: str = ""
    mode: str = "online"


@dataclass
class Config:
    run_name: str = "debug"
    artifact_root: str = "runs"
    seed: int = 0
    iterations: int = 1
    net: NetConfig = field(default_factory=NetConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Build a dataclass from nested dicts; unknown keys are an error (typos fail loudly)."""
    fields = {f.name: f for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - set(fields)
    if unknown:
        raise KeyError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    defaults = cls()  # type: ignore[call-arg]
    hints = {f.name: type(getattr(defaults, f.name)) for f in fields.values()}
    for k, v in data.items():
        t = hints[k]
        if dataclasses.is_dataclass(t) and isinstance(v, dict):
            base = dataclasses.asdict(getattr(defaults, k))
            base.update(v)
            kwargs[k] = from_dict(t, base)
        else:
            kwargs[k] = (
                type(getattr(defaults, k))(v)
                if v is not None and not isinstance(v, bool) and t in (int, float, str)
                else v
            )
    return cls(**kwargs)  # type: ignore[call-arg]


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for o in overrides:
        key, _, raw = o.partition("=")
        if not _:
            raise ValueError(f"override must look like a.b=value, got {o!r}")
        cur = data
        parts = key.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = yaml.safe_load(raw)
    return data


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    data: dict[str, Any] = {}
    if path:
        data = yaml.safe_load(Path(path).read_text()) or {}
    data = apply_overrides(data, overrides or [])
    return from_dict(Config, data)


def to_dict(cfg: Config) -> dict[str, Any]:
    return dataclasses.asdict(cfg)
