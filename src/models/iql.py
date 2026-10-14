from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .common import DoubleQ, ValueNet, clone_module, soft_update, minibatch_indices
from .schedules import build_iql_actor_scheduler


class IQLActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims):
        super().__init__()
        layers = []
        dims = [state_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def distribution(self, s):
        h = self.backbone(s)
        mean = torch.tanh(self.mean_head(h))
        log_std = torch.clamp(self.log_std, -5.0, 2.0).expand_as(mean)
        return Normal(mean, log_std.exp())

    def log_prob(self, s, a):
        return self.distribution(s).log_prob(a).sum(dim=-1, keepdim=True)

    def deterministic(self, s):
        h = self.backbone(s)
        return torch.tanh(self.mean_head(h))


@dataclass
class IQLPolicy:
    actor: IQLActor
    state_scaler: object
    action_space: object
    device: str

    def normalized_action(self, state):
        z = self.state_scaler.transform(np.atleast_2d(state)).astype(np.float32)
        s = torch.tensor(z, dtype=torch.float32, device=self.device)
        self.actor.eval()
        with torch.no_grad():
            a = self.actor.deterministic(s).cpu().numpy()[0]
        return np.clip(a, -1.0, 1.0)

    def __call__(self, state):
        return self.action_space.decode(self.normalized_action(state))


def expectile_loss(diff, expectile):
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return (weight * diff.pow(2)).mean()


def train_iql(
    dataset,
    state_scaler,
    action_space,
    cfg,
    seed: int,
    updates: int,
    device="cpu",
    progress_callback=None,
):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    iql = cfg["offline_rl"]["iql"]
    gamma = float(cfg["offline_rl"]["gamma"])
    batch_size = int(cfg["offline_rl"]["batch_size"])
    tau = float(iql["target_tau"])
    expectile = float(iql["expectile"])
    temperature = float(iql["temperature"])
    max_adv_weight = float(iql["max_adv_weight"])

    S = state_scaler.transform(dataset.S).astype(np.float32)
    SN = state_scaler.transform(dataset.SN).astype(np.float32)
    A = dataset.A_normalized.astype(np.float32)
    R = dataset.R.astype(np.float32).reshape(-1, 1)
    M = (1.0 - dataset.terminal.astype(np.float32)).reshape(-1, 1)

    state_dim = S.shape[1]
    action_dim = A.shape[1]
    hidden = iql["hidden_dims"]

    critic = DoubleQ(state_dim, action_dim, hidden).to(device)
    target_critic = clone_module(critic).to(device)
    value = ValueNet(state_dim, hidden).to(device)
    actor = IQLActor(state_dim, action_dim, hidden).to(device)

    critic_opt = torch.optim.Adam(critic.parameters(), lr=float(iql["critic_lr"]))
    value_opt = torch.optim.Adam(value.parameters(), lr=float(iql["value_lr"]))
    actor_opt = torch.optim.Adam(actor.parameters(), lr=float(iql["actor_lr"]))
    actor_sched = build_iql_actor_scheduler(actor_opt, updates)

    logs = []
    n = len(dataset)
    for step in range(1, updates + 1):
        idx = minibatch_indices(n, batch_size, rng)
        s = torch.tensor(S[idx], device=device)
        sn = torch.tensor(SN[idx], device=device)
        a = torch.tensor(A[idx], device=device)
        r = torch.tensor(R[idx], device=device)
        m = torch.tensor(M[idx], device=device)

        with torch.no_grad():
            tq1, tq2 = target_critic(s, a)
            tq = torch.minimum(tq1, tq2)
        v = value(s)
        v_loss = expectile_loss(tq - v, expectile)
        value_opt.zero_grad()
        v_loss.backward()
        value_opt.step()

        with torch.no_grad():
            next_v = value(sn)
            q_target = r + gamma * m * next_v
        q1, q2 = critic(s, a)
        q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        critic_opt.zero_grad()
        q_loss.backward()
        critic_opt.step()

        with torch.no_grad():
            tq1, tq2 = target_critic(s, a)
            adv = torch.minimum(tq1, tq2) - value(s)
            weights = torch.exp(temperature * adv).clamp(max=max_adv_weight)
        log_prob = actor.log_prob(s, a)
        actor_loss = -(weights * log_prob).mean()
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()
        actor_sched.step()

        soft_update(critic, target_critic, tau)

        if step == 1 or step == updates or step % max(1, updates // 10) == 0:
            record = {
                "step": step,
                "value_loss": float(v_loss.detach().cpu()),
                "critic_loss": float(q_loss.detach().cpu()),
                "actor_loss": float(actor_loss.detach().cpu()),
                "mean_adv": float(adv.mean().detach().cpu()),
                "actor_lr": float(actor_opt.param_groups[0]["lr"]),
            }
            logs.append(record)
            if progress_callback is not None:
                progress_callback(step, updates, record)

    return IQLPolicy(actor, state_scaler, action_space, device), logs
