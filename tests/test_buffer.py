import numpy as np
import pytest
import torch
from gymnasium.spaces import Box

from ppo.buffer import RolloutBuffer


def make_dummy_buffer(
    gamma=0.9, gae_lambda=0.95, num_envs=1, rollout_length=2, device=None
):
    """Helper to construct a small buffer and override its contents."""
    obs_space = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
    action_space = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
    buf = RolloutBuffer(
        obs_space,
        action_space,
        gamma,
        gae_lambda,
        num_envs,
        rollout_length,
        device=torch.device("cpu") if device is None else device,
    )
    return buf


def test_compute_gae_simple_episode():
    """
    2-step episode:
      - step 0: r0=0, V0=1, done1=0
      - step 1: r1=1, V1=2, done2=1
    bootstrap_value=0

    Manually:
      δ1 = 1 + 0.9*0*? - 2 = -1
      A1 = -1

      δ0 = 0 + 0.9*2*1 - 1 = 0.8
      A0 = 0.8 + 0.9*0.95*1*(-1) = 0.8 - 0.855 = -0.055

      R1 = A1 + V1 = -1 + 2 = 1
      R0 = A0 + V0 = -0.055 + 1 = 0.945
    """
    buf = make_dummy_buffer()
    device = buf.device

    buf.rew_buf.copy_(torch.tensor([[0.0], [1.0]], device=device))
    buf.val_buf.copy_(torch.tensor([[1.0], [2.0]], device=device))
    buf.done_buf.copy_(torch.tensor([[0.0], [1.0]], device=device))

    buf.compute_gae(bootstrap_value=torch.tensor([0.0], device=device))

    adv = buf.adv_buf.cpu().numpy().reshape(-1)
    ret = buf.ret_buf.cpu().numpy().reshape(-1)

    expected_adv = np.array([-0.055, -1.0])
    expected_ret = np.array([0.945, 1.0])

    assert np.allclose(adv, expected_adv, atol=1e-3), f"adv_buf {adv} ≠ {expected_adv}"
    assert np.allclose(ret, expected_ret, atol=1e-3), f"ret_buf {ret} ≠ {expected_ret}"


def test_add_transition_overflow_raises():
    """
    If we add more transitions than rollout_length, ptr >= rollout_length
    should trigger a RuntimeError.
    """
    buf = make_dummy_buffer(rollout_length=2)
    device = buf.device

    obs = torch.zeros((1, 1), device=device)
    action = torch.zeros((1, 1), device=device)
    logp = torch.zeros((1,), device=device)
    reward = torch.zeros((1,), device=device)
    done = torch.zeros((1,), device=device)
    value = torch.zeros((1,), device=device)

    buf.add_transition(obs, action, logp, reward, done, value)
    buf.add_transition(obs, action, logp, reward, done, value)

    with pytest.raises(RuntimeError):
        buf.add_transition(obs, action, logp, reward, done, value)
