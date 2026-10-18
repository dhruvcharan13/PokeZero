"""Policy/value network.

state:  card-ID embeddings + numeric card features -> per-slot encoder (both boards),
        set-pooled hand / discard embeddings, global scalars -> MLP trunk -> h
action: option type / area one-hots + embedded referenced card + embedded attack -> a_i
policy: logit_i = <W([h, ctx]), a_i> / sqrt(d) + b(a_i), masked to legal options (+ STOP)
value:  tanh(MLP(h)) = expected result for the acting player in [-1, 1]
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from pokezero.env.cards import (
    CARD_FEATURE_DIM,
    attack_table,
    attack_vocab_size,
    card_feature_matrix,
    vocab_size,
)
from pokezero.model.encoding import (
    GLOBAL_FEAT,
    MAX_DISCARD,
    MAX_HAND,
    N_CTX,
    N_SLOTS,
    OPT_FEAT,
    SLOT_FEAT,
    Encoded,
)


@dataclass
class NetConfig:
    card_dim: int = 64
    slot_dim: int = 128
    hidden: int = 512
    state_dim: int = 256
    action_dim: int = 128


@dataclass
class Batch:
    glob: torch.Tensor
    slot_ids: torch.Tensor
    slot_feat: torch.Tensor
    hand_ids: torch.Tensor
    disc_me: torch.Tensor
    disc_opp: torch.Tensor
    stadium_id: torch.Tensor
    select_type: torch.Tensor
    context: torch.Tensor
    opt_feat: torch.Tensor
    opt_card: torch.Tensor
    opt_attack: torch.Tensor
    mask: torch.Tensor  # (B, A) bool

    def to(self, device: torch.device | str) -> Batch:
        return Batch(**{k: v.to(device) for k, v in self.__dict__.items()})


def collate(items: Sequence[Encoded]) -> Batch:
    a_max = max(e.mask.shape[0] for e in items)
    b = len(items)
    opt_feat = np.zeros((b, a_max, OPT_FEAT), dtype=np.float32)
    opt_card = np.zeros((b, a_max), dtype=np.int64)
    opt_attack = np.zeros((b, a_max), dtype=np.int64)
    mask = np.zeros((b, a_max), dtype=bool)
    for i, e in enumerate(items):
        n = e.mask.shape[0]
        opt_feat[i, :n], opt_card[i, :n] = e.opt_feat, e.opt_card
        opt_attack[i, :n], mask[i, :n] = e.opt_attack, e.mask

    def stack(f: str, dt: type = np.int64) -> torch.Tensor:
        return torch.from_numpy(np.stack([getattr(e, f) for e in items]).astype(dt))

    def scalar(f: str) -> torch.Tensor:
        return torch.tensor([getattr(e, f) for e in items], dtype=torch.long)

    return Batch(
        glob=stack("glob", np.float32),
        slot_ids=stack("slot_ids"),
        slot_feat=stack("slot_feat", np.float32),
        hand_ids=stack("hand_ids"),
        disc_me=stack("disc_me"),
        disc_opp=stack("disc_opp"),
        stadium_id=scalar("stadium_id"),
        select_type=scalar("select_type"),
        context=scalar("context"),
        opt_feat=torch.from_numpy(opt_feat),
        opt_card=torch.from_numpy(opt_card),
        opt_attack=torch.from_numpy(opt_attack),
        mask=torch.from_numpy(mask),
    )


def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


class PokeNet(nn.Module):
    def __init__(self, cfg: NetConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or NetConfig()
        self.card_emb = nn.Embedding(vocab_size(), cfg.card_dim, padding_idx=0)
        self.register_buffer("card_feats", torch.from_numpy(card_feature_matrix()))
        self.card_feat_proj = nn.Linear(CARD_FEATURE_DIM, cfg.card_dim)
        self.attack_emb = nn.Embedding(attack_vocab_size(), cfg.card_dim, padding_idx=0)
        dmg = np.zeros((attack_vocab_size(), 1), dtype=np.float32)
        for aid, a in attack_table().items():
            dmg[aid, 0] = a["damage"] / 350.0
        self.register_buffer("attack_dmg", torch.from_numpy(dmg))
        self.slot_enc = _mlp(cfg.card_dim + SLOT_FEAT, cfg.slot_dim, cfg.slot_dim)
        self.slot_side = nn.Parameter(torch.zeros(2 * N_SLOTS, cfg.slot_dim))
        self.zone_proj = nn.Linear(cfg.card_dim, cfg.card_dim)
        trunk_in = 2 * N_SLOTS * cfg.slot_dim + 3 * 2 * cfg.card_dim + cfg.card_dim + GLOBAL_FEAT
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.state_dim),
            nn.ReLU(),
        )
        self.type_emb = nn.Embedding(N_CTX, cfg.state_dim)
        self.ctx_emb = nn.Embedding(N_CTX, cfg.state_dim)
        self.action_enc = _mlp(OPT_FEAT + 2 * cfg.card_dim + 1, cfg.hidden // 2, cfg.action_dim)
        self.query = nn.Linear(cfg.state_dim, cfg.action_dim)
        self.action_bias = nn.Linear(cfg.action_dim, 1)
        self.value_head = nn.Sequential(
            nn.Linear(cfg.state_dim, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh()
        )

    # -- helpers -------------------------------------------------------------------------
    def embed_cards(self, ids: torch.Tensor) -> torch.Tensor:
        feats = self.card_feat_proj(self.card_feats[ids])  # type: ignore[index]
        return (self.card_emb(ids) + feats) * (ids > 0).unsqueeze(-1)

    def _pool(self, ids: torch.Tensor) -> torch.Tensor:
        e = torch.relu(self.zone_proj(self.embed_cards(ids)))
        m = (ids > 0).unsqueeze(-1).float()
        e = e * m
        return torch.cat([e.sum(1) / 10.0, e.max(1).values], dim=-1)

    def encode_state(self, b: Batch) -> torch.Tensor:
        slots = torch.cat([self.embed_cards(b.slot_ids), b.slot_feat], dim=-1)
        slots = self.slot_enc(slots) + self.slot_side
        slots = slots * (b.slot_ids > 0).unsqueeze(-1)
        x = torch.cat(
            [
                slots.flatten(1),
                self._pool(b.hand_ids),
                self._pool(b.disc_me),
                self._pool(b.disc_opp),
                self.embed_cards(b.stadium_id),
                b.glob,
            ],
            dim=-1,
        )
        h = self.trunk(x)
        return h + self.type_emb(b.select_type) + self.ctx_emb(b.context)

    def forward(self, b: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (masked logits (B, A), value (B,))."""
        h = self.encode_state(b)
        a_in = torch.cat(
            [
                b.opt_feat,
                self.embed_cards(b.opt_card),
                self.attack_emb(b.opt_attack),
                self.attack_dmg[b.opt_attack],  # type: ignore[index]
            ],
            dim=-1,
        )
        a = self.action_enc(a_in)
        q = self.query(h).unsqueeze(1)
        logits = (q * a).sum(-1) / a.shape[-1] ** 0.5 + self.action_bias(a).squeeze(-1)
        logits = logits.masked_fill(~b.mask, float("-inf"))
        return logits, self.value_head(h).squeeze(-1)

    @torch.no_grad()
    def predict(self, items: Sequence[Encoded]) -> tuple[list[np.ndarray], np.ndarray]:
        """Priors (softmax over legal actions, per item) and values for a list of encodings."""
        was = self.training
        self.eval()
        logits, v = self(collate(items))
        probs = torch.softmax(logits, dim=-1)
        self.train(was)
        pr = [probs[i, : e.mask.shape[0]].numpy() for i, e in enumerate(items)]
        return pr, v.numpy()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["Batch", "NetConfig", "PokeNet", "collate", "MAX_HAND", "MAX_DISCARD"]
