from __future__ import annotations

import copy
import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
EPS = 1e-6


def mlp(input_dim: int, hidden_dims, output_dim: int, activate_final: bool = False):
    layers = []
    dims = [input_dim] + list(hidden_dims)
    for i in range(len(dims) - 1):
        layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
    layers.append(nn.Linear(dims[-1], output_dim))
    if activate_final:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class DoubleQ(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims):
        super().__init__()
        self.q1 = mlp(state_dim + action_dim, hidden_dims, 1)
        self.q2 = mlp(state_dim + action_dim, hidden_dims, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.q1(x), self.q2(x)

    def min(self, s, a):
        q1, q2 = self(s, a)
        return torch.minimum(q1, q2)


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dims):
        super().__init__()
        self.net = mlp(state_dim, hidden_dims, 1)

    def forward(self, s):
        return self.net(s)


class GaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims, state_dependent_std=True):
        super().__init__()
        layers = []
        dims = [state_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
        self.backbone = nn.Sequential(*layers)
        self.mean = nn.Linear(hidden_dims[-1], action_dim)
        self.state_dependent_std = state_dependent_std
        if state_dependent_std:
            self.log_std = nn.Linear(hidden_dims[-1], action_dim)
        else:
            self.log_std_param = nn.Parameter(torch.zeros(action_dim))

    def _params(self, s):
        h = self.backbone(s)
        mean = self.mean(h)
        if self.state_dependent_std:
            log_std = self.log_std(h)
        else:
            log_std = self.log_std_param.expand_as(mean)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, s, tanh=True):
        mean, log_std = self._params(s)
        std = log_std.exp()
        dist = Normal(mean, std)
        z = dist.rsample()
        if tanh:
            a = torch.tanh(z)
            log_prob = dist.log_prob(z) - torch.log(1 - a.pow(2) + EPS)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            mean_action = torch.tanh(mean)
        else:
            a = z
            log_prob = dist.log_prob(z).sum(dim=-1, keepdim=True)
            mean_action = mean
        return a, log_prob, mean_action

    def log_prob(self, s, action, tanh=True):
        mean, log_std = self._params(s)
        std = log_std.exp()
        dist = Normal(mean, std)
        if tanh:
            clipped = torch.clamp(action, -1 + EPS, 1 - EPS)
            z = torch.atanh(clipped)
            log_prob = dist.log_prob(z) - torch.log(1 - clipped.pow(2) + EPS)
        else:
            log_prob = dist.log_prob(action)
        return log_prob.sum(dim=-1, keepdim=True)

    def deterministic(self, s, tanh=True):
        mean, _ = self._params(s)
        return torch.tanh(mean) if tanh else torch.clamp(mean, -1.0, 1.0)


def soft_update(source: nn.Module, target: nn.Module, tau: float):
    with torch.no_grad():
        for p, tp in zip(source.parameters(), target.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * p.data)


def clone_module(module: nn.Module):
    return copy.deepcopy(module)


def minibatch_indices(n: int, batch_size: int, rng: np.random.Generator):
    return rng.integers(0, n, size=batch_size)
