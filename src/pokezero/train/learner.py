from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from pokezero.model.net import PokeNet
from pokezero.train.replay import ReplayBuffer, collate_examples


@dataclass
class TrainConfig:
    batch_size: int = 128
    steps_per_iter: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4  # L2 regularization (decoupled, AdamW)
    value_coef: float = 1.0
    grad_clip: float = 1.0
    min_examples: int = 256
    device: str = "cpu"  # "cpu" | "cuda" | "mps" | "auto" (self-play itself always runs on CPU)


def resolve_device(name: str = "auto") -> torch.device:
    """``auto`` picks CUDA, then Apple MPS, then CPU."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        return torch.device("mps" if mps is not None and mps.is_available() else "cpu")
    return torch.device(name)


def loss_fn(
    net: PokeNet, batch, pi: torch.Tensor, z: torch.Tensor, value_coef: float
) -> tuple[torch.Tensor, dict[str, float]]:
    logits, v = net(batch)
    logp = F.log_softmax(logits.masked_fill(~batch.mask, -1e9), dim=-1)
    policy_loss = -(pi * logp.masked_fill(~batch.mask, 0.0)).sum(-1).mean()
    value_loss = F.mse_loss(v, z)
    total = policy_loss + value_coef * value_loss
    return total, {
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
    }


def train_steps(
    net: PokeNet,
    opt: torch.optim.Optimizer,
    buf: ReplayBuffer,
    cfg: TrainConfig,
    rng: np.random.Generator,
    on_step: Callable[[dict[str, float]], None] | None = None,
) -> dict[str, float]:
    dev = resolve_device(cfg.device)
    net.to(dev).train()
    agg = {"policy_loss": 0.0, "value_loss": 0.0}
    for _ in range(cfg.steps_per_iter):
        batch, pi, z = collate_examples(buf.sample(cfg.batch_size, rng))
        batch, pi, z = batch.to(dev), pi.to(dev), z.to(dev)
        loss, parts = loss_fn(net, batch, pi, z, cfg.value_coef)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        opt.step()
        if on_step is not None:
            on_step({**parts, "loss": float(loss.detach())})
        for k, v in parts.items():
            agg[k] += v / cfg.steps_per_iter
    net.to("cpu").eval()  # agents and self-play run on CPU
    return agg
