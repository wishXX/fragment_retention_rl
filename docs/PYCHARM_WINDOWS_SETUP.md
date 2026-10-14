# Windows and PyCharm Setup

This guide describes how to run the public repository on Windows and how authorized users can reconstruct the full experiment with the private BMS dataset.

## Project Setup

Open the repository root in PyCharm and set it as the working directory for run configurations. The root contains `README.md`, `requirements.txt`, `config/`, `data/`, `docs/`, `outputs/`, `scripts/`, `src/`, and `tests/`.

Use an appropriate Python environment and install the recorded dependencies:

```powershell
python -m pip install -r requirements.txt
```

Keep a record of package versions and the compute device when attempting full experimental reproduction. The formal training configuration specifies CUDA and 2 PyTorch threads. The public aggregate verification commands do not require retraining.

## Public Repository Checks

Run these commands from the repository root:

```powershell
python -m pytest -q
python scripts/06_generate_results.py
python scripts/07_evaluate_gamma095_sensitivity.py
python scripts/08_verify_final_inference.py
```

Tests that require the private dataset are skipped when it is unavailable. Script `06` validates released aggregate retention results and regenerates figures. Script `07` recalculates the evaluation discount sensitivity from released policy-level data. Script `08` independently checks released statistics and provenance, explicitly skipping origin-level checks when private records are absent.

These commands do not train policies, refit evaluators, or modify the underlying BMS data. Script `08` is read only.

## Private Data Requirements

Full reconstruction requires authorized access to the original BMS telemetry, stored under `data/raw/`. The original source is immutable. Do not interpolate missing intervals, clip logged FTS actions to the policy envelope, or construct transitions across real gaps.

The public release intentionally excludes raw telemetry, detailed fragment chronology, origin-level OPE, evaluator folds, and trained policy checkpoints. Without those assets, the full dataset, evaluator fitting, and training stages cannot be reproduced from the public repository alone.

## Full Reconstruction Workflow

**Only run the following commands in an authorized complete environment with the required private inputs.** They are not required for public aggregate verification.

```powershell
python scripts/00_capture_runtime_provenance.py
python scripts/01_build_dataset.py
python -m pytest -q
python scripts/02_run_smoke_test.py
python scripts/03a_validate_evaluators.py
python scripts/03b_aggregate_evaluator_validation.py
python scripts/03c_accept_evaluators.py
python scripts/04_run_formal_retention.py --confirm
python scripts/05_run_formal_statistics.py
python scripts/06_generate_results.py
python scripts/07_evaluate_gamma095_sensitivity.py
python scripts/08_verify_final_inference.py
```

The formal training matrix contains 8 retention conditions, 2 algorithms, and 20 seeds, producing 320 policies. Do not rerun training merely to regenerate statistics, figures, or discount sensitivity outputs.

## OPE Only Rerun

If all required trained policies and evaluator artifacts already exist in the authorized environment, formal OPE can be rerun without training:

```powershell
python scripts/04_run_formal_retention.py --confirm --ope-only
```

Do not use this option when the required checkpoints are absent.

## Validation and Failure Handling

Evaluator acceptance requires complete leave one natural fragment out coverage and positive recursive 2-hour RMSE skill against matched persistence for `Tin_Mean` and `RTP_Total`. See [`EVALUATOR_PROTOCOL.md`](EVALUATOR_PROTOCOL.md) for the protocol.

Stop full reconstruction if raw data QA, tests, smoke tests, evaluator acceptance, or required artifact checks fail. Diagnose failures rather than bypassing them. Private source data and scientific artifacts must not be copied into the public repository.
