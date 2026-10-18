"""PUCT selection / backup / chance handling against a tiny scripted forward model."""

import copy
import random

import numpy as np
import pytest

from pokezero.env.base import Decision
from pokezero.env.cabt import CabtEnv
from pokezero.env.search_engine import SearchState
from pokezero.search.puct import OracleDisabledError, PUCTSearch, SearchConfig


class FakeNet:
    """Uniform priors, value 0 everywhere (so only terminal results drive the search)."""

    def __init__(self, priors=None):
        self.priors = priors

    def predict(self, items):
        out = []
        for e in items:
            p = e.mask.astype(np.float32)
            if self.priors is not None and len(self.priors) == len(p):
                p = p * np.asarray(self.priors, dtype=np.float32)
            out.append(p / p.sum())
        return out, np.zeros(len(items), dtype=np.float32)


@pytest.fixture(scope="module")
def template_obs():
    env = CabtEnv()
    d = env.reset()
    obs = copy.deepcopy(d.obs)
    env.close()
    return obs


class FakeEngine:
    """root (player 0) --0--> terminal, player 0 loses
                          --1--> node (player 1) --any--> terminal, player 0 wins
    With ``chance=True`` action 1 first lands in one of three distinct (turn-tagged) nodes."""

    def __init__(self, template, chance=False, max_count=1):
        self.t, self.chance, self.max_count = template, chance, max_count
        self.engine_calls = 0
        self._id = 0

    def _decision(self, player, tag=0):
        obs = copy.deepcopy(self.t)
        obs["current"]["yourIndex"] = player
        obs["current"]["turn"] = tag
        opts = tuple({"type": 7, "index": i} for i in range(2))
        obs["select"]["option"] = list(opts)
        return Decision(player, 0, 0, 1, self.max_count, opts, obs)

    def _state(self, decision, result=-1):
        self._id += 1
        obs = copy.deepcopy(decision.obs if decision else self.t)
        obs["current"]["result"] = result
        return SearchState(self._id, decision if result < 0 else None, obs)

    def begin(self, decision, my_deck, opp_deck, rng):
        return self._state(self._decision(0))

    def step(self, state, picks):
        self.engine_calls += 1
        d = state.decision
        if d.player == 0:
            if picks == [0]:
                return self._state(None, result=1)  # player 1 wins
            tag = random.randrange(3) if self.chance else 0
            return self._state(self._decision(1, tag))
        return self._state(None, result=0)  # player 0 wins

    def end(self):
        pass


def _search(engine, **kw):
    cfg = SearchConfig(
        simulations=kw.pop("sims", 60), dirichlet_eps=0.0, allow_engine_oracle=True, **kw
    )
    return PUCTSearch(FakeNet(), cfg, [1] * 60, rng=random.Random(0), engine=engine)


def test_oracle_flag_required():
    with pytest.raises(OracleDisabledError):
        PUCTSearch(FakeNet(), SearchConfig(), [1] * 60)


def test_search_finds_winning_action_and_backs_up_with_sign(template_obs):
    eng = FakeEngine(template_obs)
    s = _search(eng)
    d = eng.begin(None, None, None, None).decision
    res = s.run(d)
    assert res.visits[1] > 4 * res.visits[0]
    assert res.visits[:2].sum() == 60
    assert res.root_q[1] > 0.9 and res.root_q[0] < -0.9  # root player's viewpoint
    root = s._root
    child = next(iter(root.edges[1].outcomes.values()))
    assert child.player == 1
    # player 1 always loses below action 1, so its edge values are negative from its own view
    assert all(e.q < 0 for e in child.edges.values() if e.n)


def test_selection_follows_priors_with_no_value_signal(template_obs):
    eng = FakeEngine(template_obs)
    s = PUCTSearch(
        FakeNet([0.9, 0.1, 0.0]),
        SearchConfig(simulations=1, dirichlet_eps=0.0, allow_engine_oracle=True),
        [1] * 60,
        rng=random.Random(0),
        engine=eng,
    )
    res = s.run(eng.begin(None, None, None, None).decision)
    assert res.visits.argmax() == 0  # first simulation takes the highest-prior action


def test_chance_children_are_capped_and_resampled(template_obs):
    eng = FakeEngine(template_obs, chance=True)
    s = _search(eng, sims=80, max_chance_children=2)
    res = s.run(eng.begin(None, None, None, None).decision)
    edge = s._root.edges[1]
    assert 1 < len(edge.outcomes) <= 2
    assert sum(edge.visit_by_outcome.values()) == res.visits[1]
    assert res.engine_calls < 80 + 5  # capped edges stop calling the engine


def test_deterministic_edges_stop_calling_engine(template_obs):
    eng = FakeEngine(template_obs, chance=False)
    s = _search(eng, sims=60, chance_probe=3)
    s.run(eng.begin(None, None, None, None).decision)
    # edge 1 -> single outcome; after 3 probes no more engine calls for that edge
    assert eng.engine_calls < 25


def test_multi_pick_creates_partial_nodes_without_engine_calls(template_obs):
    eng = FakeEngine(template_obs, max_count=2)
    d = eng._decision(0)
    s = _search(eng, sims=10)
    s.engine.begin = lambda *a, **k: eng._state(d)
    s.run(d)
    root = s._root
    assert any("partial" in e.outcomes for e in root.edges.values())
    assert all(
        n.chosen != () for e in root.edges.values() for k, n in e.outcomes.items() if k == "partial"
    )
