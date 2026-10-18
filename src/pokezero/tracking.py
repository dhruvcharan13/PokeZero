"""Metric logging: JSONL always, Weights & Biases only when explicitly enabled.

W&B is an optional extra (``uv sync --extra wandb``) and reads its API key from the standard
``WANDB_API_KEY`` environment variable; nothing is stored in the repo or config. When enabled it
receives three streams: per-training-step losses (``train/*`` against ``train_step``),
per-iteration metrics (``iter/*`` against ``iteration``), and optionally promoted checkpoints
(as artifacts) and a candidate-vs-champion replay (as HTML).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from pokezero.storage import ArtifactStore


class Tracker:
    def __init__(
        self, store: ArtifactStore, wandb_cfg: Any = None, run_config: dict[str, Any] | None = None
    ):
        self.store = store
        self._lines: list[str] = []
        self._wandb: Any = None
        self._train_step = 0
        self.log_checkpoints = bool(getattr(wandb_cfg, "log_checkpoints", False))
        self.log_replays = bool(getattr(wandb_cfg, "log_replays", False))
        if wandb_cfg is not None and getattr(wandb_cfg, "enabled", False):
            try:
                import wandb
            except ImportError as e:
                raise RuntimeError(
                    "wandb.enabled=true but wandb is not installed (uv sync --extra wandb)"
                ) from e
            self._wandb = wandb.init(
                project=wandb_cfg.project,
                entity=wandb_cfg.entity or None,
                name=wandb_cfg.run_name or None,
                config=run_config,
                mode=wandb_cfg.mode,
            )
            wandb.define_metric("train_step")
            wandb.define_metric("train/*", step_metric="train_step")
            wandb.define_metric("iteration")
            wandb.define_metric("iter/*", step_metric="iteration")

    @property
    def wandb_enabled(self) -> bool:
        return self._wandb is not None

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        """Per-iteration metrics (always written to ``metrics.jsonl``)."""
        self._lines.append(json.dumps({"step": step, **metrics}))
        self.store.put_bytes("metrics.jsonl", ("\n".join(self._lines) + "\n").encode())
        if self._wandb is not None:
            numeric = {
                f"iter/{k}": float(v) for k, v in metrics.items() if isinstance(v, (int, float))
            }
            self._wandb.log({"iteration": step, **numeric})

    def log_train_step(self, losses: dict[str, float]) -> None:
        """One optimizer step's losses (W&B only; too chatty for ``metrics.jsonl``)."""
        self._train_step += 1
        if self._wandb is not None:
            self._wandb.log(
                {"train_step": self._train_step, **{f"train/{k}": v for k, v in losses.items()}}
            )

    def log_checkpoint(
        self, name: str, data: bytes, metadata: dict[str, Any] | None = None
    ) -> None:
        if self._wandb is None or not self.log_checkpoints:
            return
        import wandb

        art = wandb.Artifact(name, type="model", metadata=metadata or {})
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model.pt"
            p.write_bytes(data)
            art.add_file(str(p))
            self._wandb.log_artifact(art)

    def log_replay(self, step: int, html: str) -> None:
        if self._wandb is None or not self.log_replays:
            return
        import wandb

        self._wandb.log({"iteration": step, "iter/replay": wandb.Html(html, inject=False)})

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
