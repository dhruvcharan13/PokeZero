"""PUCT tree search over the engine's forward model.

HIDDEN-INFORMATION APPROXIMATION (initial, deliberately simple -- read before trusting results)
------------------------------------------------------------------------------------------
1. Observations given to the network are strictly the acting player's real observation. At
   opponent nodes the network sees the *public* board only (``hide_hand=True``).
2. The forward model is the engine's native search. It reproduces the opponent's REAL hand
   regardless of what we pass in (confirmed, see docs/sdk-findings.md). So transitions inside
   the tree are conditioned on hidden information. This is an *oracle leak*; it is why the
   search refuses to run unless ``allow_engine_oracle`` is set, and why agents built on it are
   labelled research-only. Replace ``EngineSearch`` with a leak-free forward model / belief
   sampler (information-set MCTS) to remove it; nothing else here needs to change.
3. Hidden zones are still determinized from public information before ``SearchBegin`` (so the
   code path is correct the day the engine honours them), under the mirror-deck assumption.

STOCHASTICITY
-------------
Decision nodes are separated by chance: every engine step re-samples shuffles, coin flips and
opponent hidden choices. An edge therefore owns a *set* of outcome children keyed by a hash of
the resulting state. While an edge has fewer than ``max_chance_children`` distinct outcomes we
call the engine and (possibly) discover a new one; afterwards we resample an existing outcome in
proportion to its visit count. Edges whose first ``chance_probe`` samples all coincide are
treated as deterministic and stop calling the engine.

VALUES are stored per acting player (zero-sum), so a leaf value is converted to each ancestor's
viewpoint during backup.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field

import numpy as np

from pokezero.env.base import Decision
from pokezero.env.search_engine import EngineSearch, SearchState
from pokezero.model.encoding import Encoded, encode, selection_complete
from pokezero.model.net import PokeNet


@dataclass
class SearchConfig:
    simulations: int = 32
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    max_chance_children: int = 4
    chance_probe: int = 3
    max_depth: int = 40
    reuse_tree: bool = True
    allow_engine_oracle: bool = False


class OracleDisabledError(RuntimeError):
    pass


@dataclass
class Edge:
    n: int = 0
    w: float = 0.0  # sum of values from the owning node's player's viewpoint
    prior: float = 0.0
    samples: int = 0
    outcomes: dict[str, Node] = field(default_factory=dict)
    visit_by_outcome: dict[str, int] = field(default_factory=dict)

    @property
    def q(self) -> float:
        return self.w / self.n if self.n else 0.0


@dataclass
class Node:
    state: SearchState | None  # None only for terminal nodes
    chosen: tuple[int, ...]
    player: int  # acting player (-1 for terminal)
    terminal_value: tuple[float, float] | None = None
    enc: Encoded | None = None
    edges: dict[int, Edge] = field(default_factory=dict)
    value: float = 0.0  # network value for ``player`` at expansion
    n: int = 0

    @property
    def expanded(self) -> bool:
        return bool(self.edges) or self.terminal_value is not None


def public_key(decision: Decision, chosen: tuple[int, ...]) -> str:
    """Hash of everything the acting player may see (opponent hand/prize are excluded)."""
    cur = decision.obs["current"]
    me = decision.player
    parts = []
    for i, p in enumerate(cur["players"]):
        q = {k: v for k, v in p.items() if k not in ("prize", "win")}
        if i != me:
            q["hand"] = None
        parts.append(q)
    blob = json.dumps(
        [
            parts,
            cur["turn"],
            cur["stadium"],
            me,
            decision.select_type,
            decision.context,
            [o for o in decision.options],
            list(chosen),
        ],
        sort_keys=True,
    )
    return hashlib.md5(blob.encode()).hexdigest()


def outcome_key(state: SearchState) -> str:
    if state.decision is None:
        return "T" + str(state.obs["current"]["result"])
    return hashlib.md5(json.dumps(state.obs["current"], sort_keys=True).encode()).hexdigest()


@dataclass
class SearchResult:
    visits: np.ndarray  # (n+1,) root visit counts per action row
    root_value: float
    root_q: np.ndarray
    sims: int
    engine_calls: int
    reused: bool


class PUCTSearch:
    def __init__(
        self,
        net: PokeNet,
        cfg: SearchConfig,
        my_deck: list[int],
        opponent_deck: list[int] | None = None,
        rng: random.Random | None = None,
        engine: EngineSearch | None = None,
    ):
        if not cfg.allow_engine_oracle:
            raise OracleDisabledError(
                "engine-backed search leaks the opponent's hidden hand; "
                "set search.allow_engine_oracle=true to use it for research"
            )
        self.net, self.cfg = net, cfg
        self.my_deck = my_deck
        self.opp_deck = opponent_deck or my_deck  # mirror-deck assumption
        self.rng = rng or random.Random()
        self.engine = engine or EngineSearch()  # injectable for tests / alternative forward models
        self._root: Node | None = None
        self._pending: Node | None = None
        self._root_player = -1
        self._np_rng = np.random.default_rng(self.rng.randrange(2**31))

    def close(self) -> None:
        self.engine.end()
        self._root = None

    # -- public API ----------------------------------------------------------------------
    def run(self, decision: Decision, chosen: tuple[int, ...] = ()) -> SearchResult:
        calls0 = self.engine.engine_calls
        root, reused = self._get_root(decision, chosen)
        self._root_player = decision.player
        if not root.expanded:
            self._expand(root)
        self._add_noise(root)
        for _ in range(self.cfg.simulations):
            self._simulate(root)
        n = decision.num_options + 1
        visits = np.zeros(n, dtype=np.float32)
        q = np.zeros(n, dtype=np.float32)
        for a, e in root.edges.items():
            visits[a], q[a] = e.n, e.q
        self._root = root
        return SearchResult(
            visits, root.value, q, self.cfg.simulations, self.engine.engine_calls - calls0, reused
        )

    def advance(self, decision: Decision, chosen: tuple[int, ...], action: int) -> None:
        """Remember the child reached by ``action`` for within-selection tree reuse."""
        root = self._root
        self._root = None
        if root is None or not self.cfg.reuse_tree:
            return
        if selection_complete(decision, chosen + (action,)) or action == decision.num_options:
            self._pending = root  # keep for cross-decision matching
        else:
            e = root.edges.get(action)
            if e is not None and "partial" in e.outcomes:
                self._root = e.outcomes["partial"]

    # -- tree policy ---------------------------------------------------------------------
    def _get_root(self, decision: Decision, chosen: tuple[int, ...]) -> tuple[Node, bool]:
        if self._root is not None and self._root.chosen == chosen and self._root.state is not None:
            if public_key(self._root.state.decision, chosen) == public_key(decision, chosen):  # type: ignore[arg-type]
                return self._root, True
        pending = self._pending
        self._pending = None
        if pending is not None and self.cfg.reuse_tree:
            hit = self._find(pending, public_key(decision, chosen), decision.player, chosen)
            if hit is not None:
                return hit, True
        self.engine.end()  # drop stale ids before opening a fresh session
        st = self.engine.begin(decision, self.my_deck, self.opp_deck, self.rng)
        return self._make_node(st, chosen), False

    def _find(self, root: Node, key: str, player: int, chosen: tuple[int, ...]) -> Node | None:
        best, stack, seen = None, [root], 0
        while stack and seen < 5000:
            node = stack.pop()
            seen += 1
            if node.state is not None and node.player == player and node.chosen == chosen:
                if node.n > 0 and public_key(node.state.decision, chosen) == key:  # type: ignore[arg-type]
                    if best is None or node.n > best.n:
                        best = node
            for e in node.edges.values():
                stack.extend(e.outcomes.values())
        return best

    def _make_node(self, st: SearchState, chosen: tuple[int, ...]) -> Node:
        if st.decision is None:
            r0, r1 = st.reward(0), st.reward(1)
            return Node(None, chosen, -1, terminal_value=(r0, r1))
        return Node(st, chosen, st.decision.player)

    def _expand(self, node: Node) -> None:
        assert node.state is not None and node.state.decision is not None
        d = node.state.decision
        node.enc = encode(d, node.chosen, hide_hand=(d.player != self._root_player))
        priors, values = self.net.predict([node.enc])
        p, node.value = priors[0], float(values[0])
        for a in np.flatnonzero(node.enc.mask):
            node.edges[int(a)] = Edge(prior=float(p[a]))

    def _add_noise(self, root: Node) -> None:
        acts = list(root.edges)
        if len(acts) < 2 or self.cfg.dirichlet_eps <= 0:
            return
        noise = self._np_rng.dirichlet([self.cfg.dirichlet_alpha] * len(acts))
        for a, nz in zip(acts, noise, strict=True):
            e = root.edges[a]
            e.prior = (1 - self.cfg.dirichlet_eps) * e.prior + self.cfg.dirichlet_eps * float(nz)

    def _select(self, node: Node) -> int:
        sqrt_n = math.sqrt(max(1, node.n))
        best, best_a = -1e18, -1
        for a, e in node.edges.items():
            u = e.q + self.cfg.c_puct * e.prior * sqrt_n / (1 + e.n)
            if u > best:
                best, best_a = u, a
        return best_a

    def _child(self, node: Node, a: int) -> Node:
        """Follow action ``a`` from ``node``, handling partial picks and chance outcomes."""
        e = node.edges[a]
        d = node.state.decision  # type: ignore[union-attr]
        assert d is not None
        chosen = node.chosen
        if a != d.num_options:
            chosen = chosen + (a,)
        if a != d.num_options and not selection_complete(d, chosen):
            child = e.outcomes.get("partial")
            if child is None:
                child = Node(node.state, chosen, node.player)
                e.outcomes["partial"] = child
            return child
        # selection completes: engine transition (stochastic)
        cfg = self.cfg
        deterministic = len(e.outcomes) == 1 and e.samples >= cfg.chance_probe
        full = len(e.outcomes) >= cfg.max_chance_children
        if (deterministic or full) and e.outcomes:
            keys = list(e.outcomes)
            w = [e.visit_by_outcome.get(k, 0) + 1 for k in keys]
            k = self.rng.choices(keys, weights=w)[0]
        else:
            st = self.engine.step(node.state, list(chosen))  # type: ignore[arg-type]
            e.samples += 1
            k = outcome_key(st)
            if k not in e.outcomes:
                e.outcomes[k] = self._make_node(st, ())
        e.visit_by_outcome[k] = e.visit_by_outcome.get(k, 0) + 1
        return e.outcomes[k]

    def _simulate(self, root: Node) -> None:
        path: list[tuple[Node, int]] = []
        node = root
        for _ in range(self.cfg.max_depth):
            if node.terminal_value is not None or not node.expanded:
                break
            a = self._select(node)
            path.append((node, a))
            node = self._child(node, a)
        if node.terminal_value is not None:
            vec = node.terminal_value
        else:
            if not node.expanded:
                self._expand(node)
            v = node.value
            vec = (v, -v) if node.player == 0 else (-v, v)
        for parent, a in path:
            e = parent.edges[a]
            e.n += 1
            e.w += vec[parent.player]
            parent.n += 1
