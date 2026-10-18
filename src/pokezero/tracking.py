"""Metric logging: JSONL always, Weights & Biases only when explicitly enabled.

W&B is an optional extra (``uv sync --extra wandb``) and reads its API key from the standard
``WANDB_API_KEY`` environment variable; nothing is stored in the repo or config.
"""

from __future__ import annotations

import json
from typing import Any

from pokezero.storage import ArtifactStore


class Tracker:
    def __init__(
        self, store: ArtifactStore, wandb_cfg: Any = None, run_config: dict[str, Any] | None = None
    ):
        self.store = store
        self._lines: list[str] = []
        self._wandb = None
        if wandb_cfg is not None and getattr(wandb_cfg, "enabled", False):
            try:
                import wandb
            except ImportError as e:  # pragma: no cover
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

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        self._lines.append(json.dumps({"step": step, **metrics}))
        self.store.put_bytes("metrics.jsonl", ("\n".join(self._lines) + "\n").encode())
        if self._wandb is not None:  # pragma: no cover
            self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self._wandb is not None:  # pragma: no cover
            self._wandb.finish()
