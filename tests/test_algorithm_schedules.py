import torch

from src.models.schedules import (
    build_iql_actor_scheduler,
    cql_actor_phase,
    cql_policy_eval_start_steps,
)


def test_c1_cql_warmstart_schedule_is_frozen():
    formal_updates = 335
    fraction = 0.04
    start_steps = cql_policy_eval_start_steps(formal_updates, fraction)

    assert start_steps == 13
    assert cql_actor_phase(1, start_steps) == "bc_warmstart"
    assert cql_actor_phase(13, start_steps) == "bc_warmstart"
    assert cql_actor_phase(14, start_steps) == "sac"
    assert cql_actor_phase(335, start_steps) == "sac"


def test_iql_cosine_scheduler_uses_full_formal_budget():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.Adam([parameter], lr=3e-4)
    scheduler = build_iql_actor_scheduler(optimizer, updates=335)

    assert scheduler.T_max == 335
    assert optimizer.param_groups[0]["lr"] == 3e-4

    for _ in range(334):
        optimizer.step()
        scheduler.step()

    assert optimizer.param_groups[0]["lr"] > 0.0

    optimizer.step()
    scheduler.step()
    assert abs(optimizer.param_groups[0]["lr"]) < 1e-15
