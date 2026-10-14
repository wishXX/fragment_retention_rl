from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import torch
import torch.nn.functional as F

from .common import DoubleQ, GaussianActor, clone_module, soft_update, minibatch_indices
from .schedules import cql_policy_eval_start_steps, cql_actor_phase


@dataclass
class CQLPolicy:
    actor: GaussianActor
    state_scaler: object
    action_space: object
    device: str

    def normalized_action(self, state):
        z = self.state_scaler.transform(np.atleast_2d(state)).astype(np.float32)
        s = torch.tensor(z, dtype=torch.float32, device=self.device)
        self.actor.eval()
        with torch.no_grad():
            a = self.actor.deterministic(s, tanh=True).cpu().numpy()[0]
        return np.clip(a, -1.0, 1.0)

    def __call__(self, state):
        return self.action_space.decode(self.normalized_action(state))


def _repeat_q(qnet, s, actions):
    b, n, ad = actions.shape
    sr = s[:, None, :].expand(b, n, s.shape[-1]).reshape(b * n, -1)
    ar = actions.reshape(b * n, ad)
    q1, q2 = qnet(sr, ar)
    return q1.view(b, n, 1), q2.view(b, n, 1)


def _sample_actor_n(actor, s, n):
    """Draw n independent policy samples per state in one vectorized forward pass."""
    b, sd = s.shape
    sr = s[:, None, :].expand(b, n, sd).reshape(b * n, sd)
    actions, logp, _ = actor.sample(sr, tanh=True)
    return actions.view(b, n, -1), logp.view(b, n, 1)


def train_cql(
    dataset,
    state_scaler,
    action_space,
    cfg,
    seed: int,
    updates: int,
    policy_lr: float,
    min_q_weight: float,
    device="cpu",
    progress_callback=None,
):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cql = cfg["offline_rl"]["cql"]
    gamma = float(cfg["offline_rl"]["gamma"])
    batch_size = int(cfg["offline_rl"]["batch_size"])
    tau = float(cql["target_tau"])
    temp = float(cql["cql_temperature"])
    num_random = int(cql["num_random"])
    deterministic_backup = bool(cql["deterministic_backup"])
    policy_eval_start = cql_policy_eval_start_steps(
        updates, float(cql["policy_eval_start_fraction"])
    )

    S_np = state_scaler.transform(dataset.S).astype(np.float32)
    SN_np = state_scaler.transform(dataset.SN).astype(np.float32)
    A_np = dataset.A_normalized.astype(np.float32)
    R_np = dataset.R.astype(np.float32).reshape(-1, 1)
    M_np = (1.0 - dataset.terminal.astype(np.float32)).reshape(-1, 1)

    S = torch.as_tensor(S_np, dtype=torch.float32, device=device)
    SN = torch.as_tensor(SN_np, dtype=torch.float32, device=device)
    A = torch.as_tensor(A_np, dtype=torch.float32, device=device)
    R = torch.as_tensor(R_np, dtype=torch.float32, device=device)
    M = torch.as_tensor(M_np, dtype=torch.float32, device=device)

    state_dim = S.shape[1]
    action_dim = A.shape[1]
    hidden = cql["hidden_dims"]

    critic = DoubleQ(state_dim, action_dim, hidden).to(device)
    target_critic = clone_module(critic).to(device)
    actor = GaussianActor(state_dim, action_dim, hidden, state_dependent_std=True).to(device)

    q_opt = torch.optim.Adam(critic.parameters(), lr=float(cql["q_lr"]))
    actor_opt = torch.optim.Adam(actor.parameters(), lr=float(policy_lr))

    target_entropy = -float(action_dim)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_opt = torch.optim.Adam([log_alpha], lr=float(policy_lr))

    logs = []
    n = len(dataset)
    random_density = math.log(0.5 ** action_dim)

    for step in range(1, updates + 1):
        idx = torch.as_tensor(
            minibatch_indices(n, batch_size, rng),
            dtype=torch.long,
            device=device,
        )
        s = S.index_select(0, idx)
        sn = SN.index_select(0, idx)
        a = A.index_select(0, idx)
        r = R.index_select(0, idx)
        m = M.index_select(0, idx)

        new_a, log_pi, _ = actor.sample(s, tanh=True)
        alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()
        alpha_opt.zero_grad()
        alpha_loss.backward()
        alpha_opt.step()
        alpha = log_alpha.exp().detach()

        q_new = critic.min(s, new_a)
        if step <= policy_eval_start:
            data_log_prob = actor.log_prob(s, a, tanh=True)
            actor_loss = (alpha * log_pi - data_log_prob).mean()
            actor_phase = cql_actor_phase(step, policy_eval_start)
        else:
            actor_loss = (alpha * log_pi - q_new).mean()
            actor_phase = cql_actor_phase(step, policy_eval_start)
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        with torch.no_grad():
            next_a, next_log_pi, _ = actor.sample(sn, tanh=True)
            target_q = target_critic.min(sn, next_a)
            if not deterministic_backup:
                target_q = target_q - alpha * next_log_pi
            y = r + gamma * m * target_q

        q1, q2 = critic(s, a)
        bellman1 = F.mse_loss(q1, y)
        bellman2 = F.mse_loss(q2, y)

        b = s.shape[0]
        rand_actions = torch.empty(
            b, num_random, action_dim, device=device
        ).uniform_(-1.0, 1.0)

        curr_actions, curr_logp = _sample_actor_n(actor, s, num_random)
        next_actions, next_logp = _sample_actor_n(actor, sn, num_random)

        candidate_actions = torch.cat(
            [rand_actions, curr_actions, next_actions],
            dim=1,
        )
        q1_candidates, q2_candidates = _repeat_q(
            critic, s, candidate_actions
        )
        q1_rand, q1_curr, q1_next = torch.split(
            q1_candidates, num_random, dim=1
        )
        q2_rand, q2_curr, q2_next = torch.split(
            q2_candidates, num_random, dim=1
        )

        cat_q1 = torch.cat([
            q1_rand - random_density,
            q1_next - next_logp.detach(),
            q1_curr - curr_logp.detach(),
        ], dim=1)
        cat_q2 = torch.cat([
            q2_rand - random_density,
            q2_next - next_logp.detach(),
            q2_curr - curr_logp.detach(),
        ], dim=1)

        cql1 = torch.logsumexp(cat_q1 / temp, dim=1).mean() * min_q_weight * temp - q1.mean() * min_q_weight
        cql2 = torch.logsumexp(cat_q2 / temp, dim=1).mean() * min_q_weight * temp - q2.mean() * min_q_weight
        q_loss = bellman1 + bellman2 + cql1 + cql2

        q_opt.zero_grad()
        q_loss.backward()
        q_opt.step()
        soft_update(critic, target_critic, tau)

        if step == 1 or step == updates or step % max(1, updates // 10) == 0:
            with torch.no_grad():
                dataset_q = critic.min(s, a).mean()
            record = {
                "step": step,
                "actor_loss": float(actor_loss.detach().cpu()),
                "q_loss": float(q_loss.detach().cpu()),
                "bellman_loss": float((bellman1 + bellman2).detach().cpu()),
                "cql_penalty": float((cql1 + cql2).detach().cpu()),
                "alpha": float(alpha.detach().cpu()),
                "dataset_q": float(dataset_q.detach().cpu()),
                "actor_phase": actor_phase,
                "policy_eval_start_steps": policy_eval_start,
            }
            logs.append(record)
            if progress_callback is not None:
                progress_callback(step, updates, record)

    return CQLPolicy(actor, state_scaler, action_space, device), logs
