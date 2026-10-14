from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.pipeline import FormalPipeline
from src.models.iql import train_iql
from src.models.cql import train_cql
from src.utils import set_seed, ensure_dir
from src.evaluation import build_common_origins
from src.evaluator_protocol import build_cross_fitted_evaluators, evaluate_policy_cross_fitted


def main():
    cfg = load_config(ROOT / "config" / "experiment.yaml")
    set_seed(cfg["project"]["seed"], cfg["project"]["torch_threads"])
    pipe = FormalPipeline(cfg).build()
    pipe.assert_frozen_qa()

    condition = "C1"
    ds = pipe.condition_dataset(condition)
    scaler = pipe.state_scaler_for(condition)
    full_updates = pipe.training_updates(condition)
    smoke_updates = min(10, full_updates)

    iql_policy, iql_logs = train_iql(ds, scaler, pipe.action_space, cfg, seed=1, updates=smoke_updates)
    cql_policy, cql_logs = train_cql(
        ds, scaler, pipe.action_space, cfg, seed=1, updates=smoke_updates,
        policy_lr=1e-4, min_q_weight=5.0,
    )

    sample_state_physical = ds.S[0]
    iql_action = iql_policy(sample_state_physical)
    cql_action = cql_policy(sample_state_physical)

    bundles = build_cross_fitted_evaluators(pipe, random_state=int(cfg["project"]["seed"]))
    origins = build_common_origins(pipe, stride=100)
    smoke_origins = origins[:2]
    iql_eval = evaluate_policy_cross_fitted(
        iql_policy, bundles, smoke_origins, pipe.reward_fn, pipe.action_space, cfg
    )
    cql_eval = evaluate_policy_cross_fitted(
        cql_policy, bundles, smoke_origins, pipe.reward_fn, pipe.action_space, cfg
    )

    out = {
        "condition": condition,
        "transitions": len(ds),
        "formal_updates": full_updates,
        "smoke_updates": smoke_updates,
        "iql_action_min": float(iql_action.min()),
        "iql_action_max": float(iql_action.max()),
        "cql_action_min": float(cql_action.min()),
        "cql_action_max": float(cql_action.max()),
        "iql_last_log": iql_logs[-1],
        "cql_last_log": cql_logs[-1],
        "evaluation_origins_available": len(origins),
        "iql_smoke_eval_mean_return": float(sum(r["Discounted_Return"] for r in iql_eval) / len(iql_eval)),
        "cql_smoke_eval_mean_return": float(sum(r["Discounted_Return"] for r in cql_eval) / len(cql_eval)),
    }
    ensure_dir(ROOT / "outputs" / "qa")
    with (ROOT / "outputs" / "qa" / "smoke_test.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
