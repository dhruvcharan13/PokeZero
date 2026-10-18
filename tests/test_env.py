import pytest

from pokezero.agents import HeuristicAgent, RandomAgent
from pokezero.env.base import Trajectory, validate_picks
from pokezero.env.cabt import CabtEnv
from pokezero.eval.arena import play_game


def test_reset_exposes_decision_and_hides_opponent_hand(first_decision):
    d = first_decision
    cur = d.obs["current"]
    assert d.num_options > 0 and d.min_count <= d.max_count
    assert cur["players"][d.player]["hand"] is not None
    assert cur["players"][1 - d.player]["hand"] is None  # hidden in real observations


def test_illegal_picks_rejected(first_decision):
    env = CabtEnv()
    d = env.reset()
    for bad in ([d.num_options], [-1], [0, 0], [0] * (d.max_count + 1)):
        with pytest.raises(ValueError):
            validate_picks(d, bad)
        with pytest.raises(ValueError):
            env.step(bad)
    env.close()


def test_full_game_terminates_with_zero_sum_reward():
    rewards, traj = play_game([RandomAgent(1), RandomAgent(2)], seed=1)
    assert rewards[0] == -rewards[1]
    assert traj.result in (0, 1, 2) and len(traj.steps) > 10


def test_step_after_game_over_raises():
    env = CabtEnv()
    d = env.reset()
    agent = RandomAgent(0)
    while d is not None:
        d = env.step(agent.act(d))
    assert env.done
    with pytest.raises(RuntimeError):
        env.step([0])


def test_seeded_agents_are_deterministic_given_same_decision(first_decision):
    a, b = RandomAgent(7), RandomAgent(7)
    assert [a.act(first_decision) for _ in range(5)] == [b.act(first_decision) for _ in range(5)]


def test_trajectory_roundtrip_and_picks_are_legal():
    _, traj = play_game([HeuristicAgent(0), RandomAgent(0)], seed=3)
    again = Trajectory.from_json(traj.to_json())
    assert again.to_json() == traj.to_json()
    for s in traj.steps:
        assert len(set(s.picks)) == len(s.picks) and all(0 <= p < s.num_options for p in s.picks)


def test_two_envs_can_coexist():
    e1, e2 = CabtEnv(), CabtEnv()
    d1, d2 = e1.reset(), e2.reset()
    e1.step([0] if d1.max_count == 1 else list(range(d1.max_count)))
    assert e2.decision is d2
    e1.close()
    e2.close()
