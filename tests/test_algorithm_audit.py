import numpy as np
import torch
from src.models.iql import IQLActor
from src.models.common import GaussianActor


def test_iql_actor_is_unsquashed_gaussian_with_tanh_mean():
    actor = IQLActor(3, 1, [8, 8])
    s = torch.zeros(4, 3)
    dist = actor.distribution(s)
    assert torch.all(dist.loc <= 1.0)
    assert torch.all(dist.loc >= -1.0)
    assert actor.log_std.shape == (1,)


def test_cql_actor_log_prob_is_finite_for_dataset_actions():
    actor = GaussianActor(3, 1, [8, 8], state_dependent_std=True)
    s = torch.zeros(4, 3)
    a = torch.tensor([[-0.9], [-0.2], [0.2], [0.9]], dtype=torch.float32)
    lp = actor.log_prob(s, a, tanh=True)
    assert lp.shape == (4, 1)
    assert torch.isfinite(lp).all()
