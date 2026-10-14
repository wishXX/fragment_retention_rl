from __future__ import annotations

import torch


def cql_policy_eval_start_steps(updates: int, fraction: float) -> int:
    """Return the frozen CQL behavior-cloning warm-start length."""
    if updates < 1:
        raise ValueError("updates must be positive")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must lie in [0, 1]")
    return int(round(float(fraction) * int(updates)))


def cql_actor_phase(step: int, policy_eval_start_steps: int) -> str:
    """Return the formal CQL actor phase for a one-indexed optimization step."""
    if step < 1:
        raise ValueError("step must be one-indexed and positive")
    if policy_eval_start_steps < 0:
        raise ValueError("policy_eval_start_steps must be nonnegative")
    return "bc_warmstart" if step <= policy_eval_start_steps else "sac"


def build_iql_actor_scheduler(optimizer, updates: int):
    """Build the frozen IQL cosine actor-learning-rate schedule."""
    if updates < 1:
        raise ValueError("updates must be positive")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(updates),
        eta_min=0.0,
    )
