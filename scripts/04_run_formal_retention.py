from pathlib import Path
import sys
import argparse
import hashlib
import json
import pickle
import time

import pandas as pd
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.pipeline import FormalPipeline
from src.preprocessing import StateScaler
from src.models.iql import train_iql
from src.models.cql import train_cql
from src.evaluation import build_common_origins
from src.evaluator_protocol import (
    build_selected_cross_fitted_evaluators,
    evaluate_policy_cross_fitted,
)
from src.utils import set_seed, ensure_dir


def aggregate(rows, condition, algorithm, configuration, seed):
    """Aggregate origins within fragment, then weight evaluation fragments equally."""
    df = pd.DataFrame(rows)
    required = {
        "Evaluator", "Fragment", "Discounted_Return", "Mean_Reward",
        "Mean_Energy_Component", "Mean_Underheat_Component",
        "Mean_Support_Distance", "Max_Support_Distance",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError("Formal evaluation rows are missing fields: " + ", ".join(missing))

    out = []
    for evaluator, g in df.groupby("Evaluator", sort=False):
        fragment = (
            g.groupby("Fragment", sort=False)
            .agg(
                Discounted_Return=("Discounted_Return", "mean"),
                Mean_Reward=("Mean_Reward", "mean"),
                Mean_Energy_Component=("Mean_Energy_Component", "mean"),
                Mean_Underheat_Component=("Mean_Underheat_Component", "mean"),
                Mean_Support_Distance=("Mean_Support_Distance", "mean"),
                Max_Support_Distance=("Max_Support_Distance", "max"),
                N_Origins=("Discounted_Return", "size"),
            )
            .reset_index()
        )
        out.append({
            "Condition": condition,
            "Algorithm": algorithm,
            "Configuration": configuration,
            "Seed": seed,
            "Evaluator": evaluator,
            "N_Origins": int(len(g)),
            "N_Evaluation_Fragments": int(len(fragment)),
            "Aggregation_Unit": "equal_weight_evaluation_fragment",
            "Mean_Discounted_Return": float(fragment["Discounted_Return"].mean()),
            "SD_Discounted_Return_Across_Fragments": float(fragment["Discounted_Return"].std(ddof=1)),
            "Mean_Reward": float(fragment["Mean_Reward"].mean()),
            "Mean_Energy_Component": float(fragment["Mean_Energy_Component"].mean()),
            "Mean_Underheat_Component": float(fragment["Mean_Underheat_Component"].mean()),
            "Mean_Support_Distance": float(fragment["Mean_Support_Distance"].mean()),
            "Max_Support_Distance": float(fragment["Max_Support_Distance"].max()),
        })
    return out


def attach_origin_metadata(rows, condition, algorithm, configuration, seed):
    """Attach policy identity to origin-level OPE rows for lossless re-aggregation."""
    return [
        {
            "Condition": condition,
            "Algorithm": algorithm,
            "Configuration": configuration,
            "Seed": int(seed),
            **row,
        }
        for row in rows
    ]


def require_evaluator_acceptance():
    """Return the formally accepted evaluator channels."""
    path = ROOT / "outputs" / "results" / "evaluator" / "evaluator_validation_acceptance.json"
    if not path.exists():
        raise SystemExit(
            "Evaluator validation has not been formally accepted. "
            "Run the evaluator validation and acceptance sequence first."
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("accepted") is not True:
        raise SystemExit(
            "Evaluator validation acceptance artifact does not contain accepted=true."
        )

    accepted_channels = list(payload.get("accepted_channels", []))
    if not accepted_channels:
        raise SystemExit(
            "Evaluator acceptance artifact contains no accepted evaluator channels."
        )
    return accepted_channels


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_fingerprint(pipe, accepted_channels):
    """Fingerprint every input that can change cross-fitted evaluator models."""
    acceptance = ROOT / "outputs" / "results" / "evaluator" / "evaluator_validation_acceptance.json"
    config = ROOT / "config" / "experiment.yaml"
    evaluator_source = ROOT / "src" / "evaluator_protocol.py"

    payload = {
        "raw_file": str(pipe.raw_path.name),
        "raw_sha256": _sha256_file(pipe.raw_path),
        "config_sha256": _sha256_file(config),
        "evaluator_protocol_sha256": _sha256_file(evaluator_source),
        "acceptance_sha256": _sha256_file(acceptance),
        "accepted_channels": list(accepted_channels),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    payload["fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return payload


def load_or_build_evaluator_cache(pipe, cfg, accepted_channels):
    """Load a valid evaluator cache or rebuild only accepted evaluator channels."""
    cache_dir = ensure_dir(ROOT / "outputs" / "cache")
    cache_path = cache_dir / "accepted_cross_fitted_evaluators.pkl"
    manifest_path = cache_dir / "accepted_cross_fitted_evaluators_manifest.json"

    expected = _cache_fingerprint(pipe, accepted_channels)

    if cache_path.exists() and manifest_path.exists():
        try:
            observed = json.loads(manifest_path.read_text(encoding="utf-8"))
            if observed.get("fingerprint") == expected["fingerprint"]:
                print("[Evaluator cache] Valid cache found. Loading accepted evaluators...", flush=True)
                with cache_path.open("rb") as handle:
                    bundles = pickle.load(handle)
                print("[Evaluator cache] Loaded.", flush=True)
                return bundles
            print("[Evaluator cache] Inputs changed. Rebuilding cache.", flush=True)
        except Exception as exc:
            print(
                f"[Evaluator cache] Cache could not be loaded ({exc}). Rebuilding.",
                flush=True,
            )
    else:
        print("[Evaluator cache] No valid cache found. Building once.", flush=True)

    def fold_progress(channel, fold_index, n_folds, completed, total):
        print(
            f"[Evaluator cache] {channel} fold {fold_index}/{n_folds} "
            f"({completed}/{total} evaluator-fold fits)",
            flush=True,
        )

    started = time.perf_counter()
    bundles = build_selected_cross_fitted_evaluators(
        pipe,
        accepted_channels,
        random_state=int(cfg["project"]["seed"]),
        progress_callback=fold_progress,
    )
    elapsed = time.perf_counter() - started

    with cache_path.open("wb") as handle:
        pickle.dump(bundles, handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest_path.write_text(
        json.dumps(expected, indent=2),
        encoding="utf-8",
    )
    print(
        f"[Evaluator cache] Built and saved in {elapsed:.1f} s.",
        flush=True,
    )
    return bundles


def progress_printer(label):
    """Create a compact callback for approximately ten progress updates."""
    started = time.perf_counter()

    def callback(step, updates, record):
        pct = 100.0 * step / updates
        elapsed = time.perf_counter() - started
        phase = record.get("actor_phase")
        phase_text = f", phase={phase}" if phase is not None else ""
        print(
            f"    {label}: {step}/{updates} ({pct:5.1f}%), "
            f"elapsed={elapsed:.1f} s{phase_text}",
            flush=True,
        )

    return callback




def _result_key(condition, algorithm, configuration, seed):
    return (str(condition), str(algorithm), str(configuration), int(seed))


def load_completed_keys(result_path: Path):
    """Load completed training/evaluation units from an existing summary CSV."""
    if not result_path.exists():
        return set()
    df = pd.read_csv(result_path)
    required = {"Condition", "Algorithm", "Configuration", "Seed"}
    if not required.issubset(df.columns):
        raise SystemExit(
            "Existing formal-retention summary is missing required resume columns."
        )
    return {
        _result_key(row.Condition, row.Algorithm, row.Configuration, row.Seed)
        for row in df.itertuples(index=False)
    }


def checkpoint_results(result_path: Path, results):
    """Atomically checkpoint all completed summary rows."""
    tmp = result_path.with_suffix(".tmp.csv")
    pd.DataFrame(results).to_csv(tmp, index=False)
    tmp.replace(result_path)


def checkpoint_logs(log_path: Path, training_logs):
    """Atomically checkpoint all completed training logs."""
    tmp = log_path.with_suffix(".tmp.csv")
    pd.DataFrame(training_logs).to_csv(tmp, index=False)
    tmp.replace(log_path)


def checkpoint_origins(origin_path: Path, origin_rows):
    """Atomically checkpoint lossless origin-level formal OPE rows."""
    tmp = origin_path.with_suffix(".tmp.csv")
    pd.DataFrame(origin_rows).to_csv(tmp, index=False)
    tmp.replace(origin_path)


def policy_checkpoint_path(model_dir: Path, condition, algorithm, configuration, seed):
    """Return the canonical checkpoint path for one formal policy."""
    safe_configuration = str(configuration).replace(".", "p")
    filename = f"{str(condition).lower()}_{str(algorithm).lower()}_{safe_configuration}_seed{int(seed)}.pt"
    return model_dir / filename


def save_policy_checkpoint(policy, path: Path):
    """Atomically persist a trained formal policy."""
    tmp = path.with_suffix(".tmp.pt")
    torch.save(policy, tmp)
    tmp.replace(path)


FORMAL_CONDITIONS = ["C1", "C2", "C3", "C5", "C10", "C16", "C19", "C28"]
FORMAL_SEEDS = list(range(1, 21))
FORMAL_CQL_POLICY_LR = 1e-4
FORMAL_CQL_MIN_Q_WEIGHT = 5.0

FORMAL_RETENTION_TARGETS = {
    "C1": 0.0,
    "C2": 20.0,
    "C3": 40.0,
    "C5": 60.0,
    "C10": 80.0,
    "C16": 90.0,
    "C19": 95.0,
    "C28": 100.0,
}


def load_retention_metadata():
    """Load realized retention statistics for the frozen formal conditions."""
    path = ROOT / "outputs" / "fragments" / "formal_cumulative_conditions.csv"
    if not path.exists():
        raise SystemExit(f"Formal cumulative-condition artifact does not exist: {path}")

    df = pd.read_csv(path)
    required = {
        "Condition",
        "Fragments_Retained",
        "Cumulative_Observed_Hours",
        "Cumulative_Eligible_Transitions",
        "Total_Transition_Retention_pct",
        "Additional_Fragment_Retention_pct",
    }
    if not required.issubset(df.columns):
        missing = sorted(required.difference(df.columns))
        raise SystemExit(
            "Formal cumulative-condition artifact is missing required columns: "
            + ", ".join(missing)
        )

    indexed = df.set_index("Condition")
    missing_conditions = [c for c in FORMAL_CONDITIONS if c not in indexed.index]
    if missing_conditions:
        raise SystemExit(
            "Formal cumulative-condition artifact is missing frozen condition(s): "
            + ", ".join(missing_conditions)
        )

    metadata = {}
    for condition in FORMAL_CONDITIONS:
        row = indexed.loc[condition]
        metadata[condition] = {
            "Target_Additional_Fragment_Retention_pct": float(
                FORMAL_RETENTION_TARGETS[condition]
            ),
            "Actual_Additional_Fragment_Retention_pct": float(
                row["Additional_Fragment_Retention_pct"]
            ),
            "Total_Transition_Retention_pct": float(
                row["Total_Transition_Retention_pct"]
            ),
            "Fragments_Retained": int(row["Fragments_Retained"]),
            "Cumulative_Observed_Hours": int(row["Cumulative_Observed_Hours"]),
            "Cumulative_Eligible_Transitions": int(
                row["Cumulative_Eligible_Transitions"]
            ),
        }
    return metadata


def aggregate_formal(rows, condition, algorithm, configuration, seed, retention_metadata):
    """Aggregate evaluator rows and attach frozen retention coordinates."""
    out = aggregate(rows, condition, algorithm, configuration, seed)
    metadata = retention_metadata[condition]
    for row in out:
        row["Horizon_Hours"] = 2
        row.update(metadata)
    return out


def validate_resume_artifacts(result_path: Path, log_path: Path, origin_path: Path, resume: bool):
    """Enforce safe formal output semantics before expensive initialization."""
    result_exists = result_path.exists()
    log_exists = log_path.exists()
    origin_exists = origin_path.exists()

    if resume:
        if len({result_exists, log_exists, origin_exists}) != 1:
            raise SystemExit(
                "Refusing to resume from inconsistent formal artifacts. "
                "The summary, training-log, and origin-level OPE CSVs must either all exist or all be absent."
            )
        if not result_exists:
            raise SystemExit(
                "Refusing to resume because no formal retention artifacts exist."
            )
        return

    if result_exists or log_exists or origin_exists:
        existing = []
        if result_exists:
            existing.append(str(result_path))
        if log_exists:
            existing.append(str(log_path))
        if origin_exists:
            existing.append(str(origin_path))
        raise SystemExit(
            "Refusing to overwrite existing formal retention artifacts without --resume. "
            "Existing artifact(s): " + ", ".join(existing)
        )



def run_ope_only(pipe, cfg, evaluator_bundles, origins, retention_metadata, model_dir, result_path, origin_path, device):
    """Resume policy-level OPE and atomically publish only after complete QA.

    Staging is fingerprinted against evaluator prediction code and inputs.
    Published results are protected against accidental replacement.
    """
    cql_name = f"lr{FORMAL_CQL_POLICY_LR:g}_w{FORMAL_CQL_MIN_Q_WEIGHT:g}"
    specs = [(condition, algorithm, configuration, seed)
             for condition in FORMAL_CONDITIONS for seed in FORMAL_SEEDS
             for algorithm, configuration in (("IQL", "official"), ("CQL", cql_name))]
    paths = [policy_checkpoint_path(model_dir, *spec) for spec in specs]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"OPE-only requires all {len(specs)} checkpoints; missing: {missing[:10]}")

    stage = result_path.parent / "formal_ope_staging"
    stage.mkdir(parents=True, exist_ok=True)
    manifest_path = stage / "manifest.json"
    manifest = {
        "config": _sha256_file(ROOT / "config" / "experiment.yaml"),
        "evaluation_source": _sha256_file(ROOT / "src" / "evaluation.py"),
        "protocol_source": _sha256_file(ROOT / "src" / "evaluator_protocol.py"),
        "acceptance": _sha256_file(ROOT / "outputs" / "results" / "evaluator" / "evaluator_validation_acceptance.json"),
        "raw_data": _sha256_file(pipe.raw_path),
        "policy_checkpoints": [[str(p.relative_to(ROOT)), _sha256_file(p)] for p in paths],
        "origin_count": len(origins),
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise SystemExit("OPE staging inputs changed. Preserve/archive the staging folder before restarting.")
    elif any(stage.iterdir()):
        raise SystemExit("OPE staging contains files without a manifest; inspect it before proceeding.")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    started = time.perf_counter()
    expected_fragments = {origin.fragment for origin in origins}
    expected_origin_counts = pd.Series([origin.fragment for origin in origins]).value_counts().to_dict()
    for index, (spec, checkpoint) in enumerate(zip(specs, paths), 1):
        condition, algorithm, configuration, seed = spec
        tag = f"{condition.lower()}_{algorithm.lower()}_{configuration.replace('.', 'p')}_seed{seed}"
        policy_origin_path = stage / f"{tag}_origins.csv"
        policy_summary_path = stage / f"{tag}_summary.csv"
        if policy_origin_path.exists() and policy_summary_path.exists():
            continue
        if policy_origin_path.exists() or policy_summary_path.exists():
            raise SystemExit(f"Incomplete staged policy pair: {tag}; inspect before retrying")
        policy = torch.load(checkpoint, map_location=device, weights_only=False)
        eval_rows = evaluate_policy_cross_fitted(policy, evaluator_bundles, origins, pipe.reward_fn, pipe.action_space, cfg)
        attached = attach_origin_metadata(eval_rows, *spec)
        summary = aggregate_formal(eval_rows, *spec, retention_metadata)
        for evaluator in ("ARX", "XGBoost"):
            subset = [r for r in eval_rows if r["Evaluator"] == evaluator]
            observed = pd.Series([r["Fragment"] for r in subset]).value_counts().to_dict()
            if observed != expected_origin_counts:
                raise SystemExit(f"Origin coverage mismatch for {tag} / {evaluator}")
        if len(summary) != 2 or {r["Evaluator"] for r in summary} != {"ARX", "XGBoost"}:
            raise SystemExit(f"Summary evaluator coverage mismatch: {tag}")
        tmp_origin = policy_origin_path.with_suffix(".tmp")
        tmp_summary = policy_summary_path.with_suffix(".tmp")
        pd.DataFrame(attached).to_csv(tmp_origin, index=False)
        pd.DataFrame(summary).to_csv(tmp_summary, index=False)
        tmp_origin.replace(policy_origin_path)
        tmp_summary.replace(policy_summary_path)
        del policy, eval_rows, attached
        if index == 1 or index % 10 == 0 or index == len(specs):
            print(f"[OPE-only] {index}/{len(specs)} policies; elapsed={time.perf_counter()-started:.1f}s", flush=True)

    origin_parts = []
    summary_parts = []
    for condition, algorithm, configuration, seed in specs:
        tag = f"{condition.lower()}_{algorithm.lower()}_{configuration.replace('.', 'p')}_seed{seed}"
        origin_parts.append(pd.read_csv(stage / f"{tag}_origins.csv"))
        summary_parts.append(pd.read_csv(stage / f"{tag}_summary.csv"))
    origins_df = pd.concat(origin_parts, ignore_index=True)
    summary_df = pd.concat(summary_parts, ignore_index=True)
    keys = ["Condition", "Algorithm", "Configuration", "Seed", "Evaluator"]
    expected_groups = len(specs) * 2
    origin_groups = origins_df.groupby(keys, dropna=False).size()
    if len(summary_df) != expected_groups or len(origin_groups) != expected_groups:
        raise SystemExit("Formal OPE publication blocked: incomplete policy/evaluator groups")
    if summary_df.duplicated(keys).any() or not np.isfinite(summary_df.select_dtypes(include="number")).all().all():
        raise SystemExit("Formal OPE publication blocked: duplicate keys or non-finite summary")
    if not (summary_df["Aggregation_Unit"] == "equal_weight_evaluation_fragment").all():
        raise SystemExit("Formal OPE publication blocked: aggregation rule mismatch")
    for key, g in origins_df.groupby(keys, dropna=False):
        if g["Fragment"].value_counts().to_dict() != expected_origin_counts:
            raise SystemExit(f"Formal OPE publication blocked: origin coverage mismatch {key}")
        fragment_means = g.groupby("Fragment")["Discounted_Return"].mean()
        published = summary_df.set_index(keys).loc[key, "Mean_Discounted_Return"]
        if not np.isclose(fragment_means.mean(), published, rtol=0, atol=1e-9):
            raise SystemExit(f"Formal OPE publication blocked: fragment-equal mismatch {key}")

    # Do not silently overwrite accepted formal results or recreate retired backups.
    existing = [str(path) for path in (result_path, origin_path) if path.exists()]
    if existing:
        raise SystemExit(
            "Formal OPE publication blocked: published output already exists: "
            + ", ".join(existing)
            + ". Preserve the existing results and use a separately reviewed "
              "publication procedure if replacement is ever necessary."
        )
    tmp_summary = result_path.with_suffix(".publish.tmp.csv")
    tmp_origin = origin_path.with_suffix(".publish.tmp.csv")
    summary_df.to_csv(tmp_summary, index=False)
    origins_df.to_csv(tmp_origin, index=False)
    tmp_summary.replace(result_path)
    tmp_origin.replace(origin_path)
    print(f"Formal OPE published: {len(summary_df)} summary rows, {len(origins_df)} origin rows.", flush=True)
    print(f"Policy-level staging retained at {stage}.", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required to run the frozen formal retention experiment",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from completed condition/algorithm/configuration/seed rows",
    )
    parser.add_argument(
        "--ope-only",
        action="store_true",
        help="Load existing formal policy checkpoints and recompute OPE without retraining",
    )
    args = parser.parse_args()

    if not args.confirm:
        raise SystemExit(
            "Refusing to run. Add --confirm only after all formal QA gates pass."
        )

    result_dir = ensure_dir(ROOT / "outputs" / "results" / "retention")
    log_dir = ensure_dir(ROOT / "outputs" / "logs")
    model_dir = ensure_dir(ROOT / "outputs" / "models" / "formal_retention")
    result_path = result_dir / "formal_retention_summary.csv"
    log_path = log_dir / "formal_retention_training_logs.csv"
    origin_path = result_dir / "formal_retention_origin_level_ope.csv"

    # Preserve previous formal outputs. TC OPE-only writes dedicated artifacts.
    if args.ope_only:
        result_path = result_dir / "formal_retention_summary_tc.csv"
        origin_path = result_dir / "formal_retention_origin_level_ope_tc.csv"

    if args.ope_only and args.resume:
        raise SystemExit("--ope-only and --resume are mutually exclusive.")
    if not args.ope_only:
        validate_resume_artifacts(result_path, log_path, origin_path, args.resume)

    accepted_channels = require_evaluator_acceptance()
    cfg = load_config(ROOT / "config" / "experiment.yaml")
    device = str(cfg["project"]["device"])
    pipe = FormalPipeline(cfg).build()
    pipe.assert_frozen_qa()

    if int(cfg["evaluation"]["horizon_hours"]) != 2:
        raise SystemExit("Formal retention runner requires the frozen corrected evaluation horizon of 2 h.")

    required_channels = {"ARX", "XGBoost"}
    if set(accepted_channels) != required_channels:
        raise SystemExit(
            "Formal retention runner requires the two prespecified model-based evaluator "
            f"channels ARX and XGBoost. Observed: {accepted_channels}"
        )

    retention_metadata = load_retention_metadata()

    print("Run role: formal retention experiment", flush=True)
    print(f"Training device: {device}", flush=True)
    print(f"Evaluation horizon: {int(cfg['evaluation']['horizon_hours'])} h", flush=True)
    print(f"Accepted evaluator channels: {accepted_channels}", flush=True)
    print(f"Summary output: {result_path}", flush=True)
    print(f"Training-log output: {log_path}", flush=True)
    print(f"Origin-level OPE output: {origin_path}", flush=True)

    evaluator_bundles = load_or_build_evaluator_cache(
        pipe,
        cfg,
        accepted_channels,
    )

    origins = build_common_origins(pipe, stride=1)
    if args.ope_only:
        print("OPE-only mode: loading existing policy checkpoints; no policy training will run.", flush=True)
        run_ope_only(pipe, cfg, evaluator_bundles, origins, retention_metadata, model_dir, result_path, origin_path, device)
        return
    conditions = list(FORMAL_CONDITIONS)
    seeds = list(FORMAL_SEEDS)
    total_trainings = len(conditions) * len(seeds) * 2

    print(
        f"Run matrix: {len(conditions)} conditions x {len(seeds)} matched seeds "
        f"x 2 algorithms = {total_trainings} policy trainings",
        flush=True,
    )
    print(
        "Formal CQL configuration: "
        f"policy_lr={FORMAL_CQL_POLICY_LR:g}, "
        f"min_q_weight={FORMAL_CQL_MIN_Q_WEIGHT:g}",
        flush=True,
    )
    print(f"Common evaluation origins: {len(origins)}", flush=True)

    completed_keys = load_completed_keys(result_path) if args.resume else set()
    results = (
        pd.read_csv(result_path).to_dict("records")
        if args.resume and result_path.exists()
        else []
    )
    training_logs = (
        pd.read_csv(log_path).to_dict("records")
        if args.resume and log_path.exists()
        else []
    )
    origin_rows = (
        pd.read_csv(origin_path).to_dict("records")
        if args.resume and origin_path.exists()
        else []
    )

    if completed_keys:
        print(
            f"Resume mode: {len(completed_keys)} completed training units detected.",
            flush=True,
        )

    training_index = 0
    run_started = time.perf_counter()

    for condition_index, condition in enumerate(conditions, start=1):
        ds = pipe.condition_dataset(condition)
        scaler = StateScaler.fit(ds.S)
        updates = pipe.training_updates(condition)

        expected_transitions = retention_metadata[condition][
            "Cumulative_Eligible_Transitions"
        ]
        if len(ds) != expected_transitions:
            raise SystemExit(
                f"Frozen retention mismatch for {condition}: "
                f"pipeline transitions={len(ds)}, "
                f"formal artifact transitions={expected_transitions}."
            )

        print(
            f"\n[Condition {condition_index}/{len(conditions)}] "
            f"{condition}: transitions={len(ds)}, updates={updates}, "
            f"target_additional_retention="
            f"{retention_metadata[condition]['Target_Additional_Fragment_Retention_pct']:.0f}%, "
            f"actual_additional_retention="
            f"{retention_metadata[condition]['Actual_Additional_Fragment_Retention_pct']:.2f}%",
            flush=True,
        )

        for seed_index, seed in enumerate(seeds, start=1):
            print(
                f"  [Seed {seed_index}/{len(seeds)}] seed={seed}",
                flush=True,
            )

            training_index += 1
            iql_label = (
                f"IQL {condition} seed={seed} "
                f"[training {training_index}/{total_trainings}]"
            )
            iql_key = _result_key(condition, "IQL", "official", seed)

            if iql_key in completed_keys:
                print(f"  Skipping completed {iql_label}", flush=True)
            else:
                print(f"  Starting {iql_label}", flush=True)
                set_seed(seed, cfg["project"]["torch_threads"])
                started = time.perf_counter()
                policy, logs = train_iql(
                    ds,
                    scaler,
                    pipe.action_space,
                    cfg,
                    seed,
                    updates,
                    device=device,
                    progress_callback=progress_printer(iql_label),
                )
                print(
                    f"  Finished {iql_label} in "
                    f"{time.perf_counter() - started:.1f} s. Evaluating...",
                    flush=True,
                )
                iql_checkpoint = policy_checkpoint_path(model_dir, condition, "IQL", "official", seed)
                save_policy_checkpoint(policy, iql_checkpoint)

                eval_rows = evaluate_policy_cross_fitted(
                    policy,
                    evaluator_bundles,
                    origins,
                    pipe.reward_fn,
                    pipe.action_space,
                    cfg,
                )
                origin_rows.extend(attach_origin_metadata(eval_rows, condition, "IQL", "official", seed))
                results.extend(
                    aggregate_formal(
                        eval_rows,
                        condition,
                        "IQL",
                        "official",
                        seed,
                        retention_metadata,
                    )
                )
                for row in logs:
                    training_logs.append({
                        "Condition": condition,
                        "Algorithm": "IQL",
                        "Configuration": "official",
                        "Seed": seed,
                        **retention_metadata[condition],
                        **row,
                    })

                completed_keys.add(iql_key)
                checkpoint_results(result_path, results)
                checkpoint_logs(log_path, training_logs)
                checkpoint_origins(origin_path, origin_rows)
                print(
                    f"  Evaluation complete for {iql_label}. Checkpoint saved.",
                    flush=True,
                )

            cql_name = (
                f"lr{FORMAL_CQL_POLICY_LR:g}_w{FORMAL_CQL_MIN_Q_WEIGHT:g}"
            )
            training_index += 1
            cql_label = (
                f"CQL {cql_name} {condition} seed={seed} "
                f"[training {training_index}/{total_trainings}]"
            )
            cql_key = _result_key(condition, "CQL", cql_name, seed)

            if cql_key in completed_keys:
                print(f"  Skipping completed {cql_label}", flush=True)
                continue

            print(f"  Starting {cql_label}", flush=True)
            set_seed(seed, cfg["project"]["torch_threads"])
            started = time.perf_counter()
            policy, logs = train_cql(
                ds,
                scaler,
                pipe.action_space,
                cfg,
                seed,
                updates,
                policy_lr=float(FORMAL_CQL_POLICY_LR),
                min_q_weight=float(FORMAL_CQL_MIN_Q_WEIGHT),
                device=device,
                progress_callback=progress_printer(cql_label),
            )
            print(
                f"  Finished {cql_label} in "
                f"{time.perf_counter() - started:.1f} s. Evaluating...",
                flush=True,
            )
            cql_checkpoint = policy_checkpoint_path(model_dir, condition, "CQL", cql_name, seed)
            save_policy_checkpoint(policy, cql_checkpoint)

            eval_rows = evaluate_policy_cross_fitted(
                policy,
                evaluator_bundles,
                origins,
                pipe.reward_fn,
                pipe.action_space,
                cfg,
            )
            origin_rows.extend(attach_origin_metadata(eval_rows, condition, "CQL", cql_name, seed))
            results.extend(
                aggregate_formal(
                    eval_rows,
                    condition,
                    "CQL",
                    cql_name,
                    seed,
                    retention_metadata,
                )
            )
            for row in logs:
                training_logs.append({
                    "Condition": condition,
                    "Algorithm": "CQL",
                    "Configuration": cql_name,
                    "Seed": seed,
                    **retention_metadata[condition],
                    **row,
                })

            completed_keys.add(cql_key)
            checkpoint_results(result_path, results)
            checkpoint_logs(log_path, training_logs)
            checkpoint_origins(origin_path, origin_rows)
            print(
                f"  Evaluation complete for {cql_label}. Checkpoint saved.",
                flush=True,
            )

    checkpoint_results(result_path, results)
    checkpoint_logs(log_path, training_logs)
    checkpoint_origins(origin_path, origin_rows)

    elapsed = time.perf_counter() - run_started
    print("\nFormal retention runner completed.", flush=True)
    print(
        f"Completed {len(conditions)} conditions x {len(seeds)} matched seeds "
        f"x 2 algorithms = {total_trainings} policy trainings",
        flush=True,
    )
    print(f"Accepted evaluator channels: {accepted_channels}", flush=True)
    print(f"Common evaluation origins: {len(origins)}", flush=True)
    print(f"Training/evaluation elapsed: {elapsed:.1f} s", flush=True)
    print(result_path, flush=True)


if __name__ == "__main__":
    main()
