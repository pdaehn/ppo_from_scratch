import gymnasium as gym

from utils.seeding import SeedWrapper


def get_initial_observations(env, n_resets=5):
    """
    Call reset() n_resets times on env and return the list of
    initial observations (as tuples for hashability).
    """
    obs_list = []
    for _ in range(n_resets):
        obs, _ = env.reset()
        obs_list.append(tuple(obs.flatten()))
    return obs_list


def test_seedwrapper_reproducibility():
    """
    Wrappers with the same base seed must produce identical reset streams.
    """
    base_seed = 12345
    raw_env1 = gym.make("CartPole-v1")
    raw_env2 = gym.make("CartPole-v1")

    env1 = SeedWrapper(raw_env1, seed=base_seed)
    env2 = SeedWrapper(raw_env2, seed=base_seed)

    seq1 = get_initial_observations(env1, n_resets=5)
    seq2 = get_initial_observations(env2, n_resets=5)

    assert (
        seq1 == seq2
    ), "SeedWrapper with same seed did not reproduce the same obs sequence"


def test_seedwrapper_variation_between_seeds():
    """
    Wrappers with different base seeds must produce different streams.
    (We only require that at least one of the first few obs differs.)
    """
    raw_env1 = gym.make("CartPole-v1")
    raw_env2 = gym.make("CartPole-v1")

    env1 = SeedWrapper(raw_env1, seed=100)
    env2 = SeedWrapper(raw_env2, seed=200)

    seq1 = get_initial_observations(env1, n_resets=5)
    seq2 = get_initial_observations(env2, n_resets=5)

    assert any(
        o1 != o2 for o1, o2 in zip(seq1, seq2)
    ), "SeedWrapper with different seeds produced identical obs sequences"


def test_seedwrapper_internal_randomness():
    """
    Verify that successive resets on the same wrapper produce *different* observations,
    i.e. the wrapper is not fully deterministic between resets.
    """
    raw_env = gym.make("CartPole-v1")
    env = SeedWrapper(raw_env, seed=999)

    seq = get_initial_observations(env, n_resets=5)
    assert len(set(seq)) > 1, "SeedWrapper did not vary the initial obs across resets"
