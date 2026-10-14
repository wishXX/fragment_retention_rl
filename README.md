# Fragment Retention RL

This repository studies a question in offline reinforcement learning (RL):

> Given naturally fragmented historical control logs, how much disconnected but valid experience should be retained beyond the longest contiguous segment?

The experiment uses historical building management system (BMS) heating logs. Missing intervals are not reconstructed, and transitions never cross real telemetry gaps. Retention conditions add complete natural fragments rather than randomly sampled observations. This is not a conventional dataset size ablation.

## Experiment Overview

The screened dataset contains 1,288 valid hourly observations across 40 natural fragments and 1,098 eligible transitions. The longest complete fragment defines the C1 baseline. The formal retention conditions are `C1`, `C2`, `C3`, `C5`, `C10`, `C16`, `C19`, and `C28`.

IQL and CQL are trained with 20 matched seeds per condition, producing **320 policies**. Both algorithms use a training discount factor of `gamma = 0.99`. Formal offline policy evaluation (OPE) uses a 2-hour horizon and `gamma = 0.99`.

Two learned dynamics evaluators are reported separately: `ARX_TC_RIDGE_LOFO` and `XGBOOST_DEFAULT_LOFO`. Neither is treated as ground truth. Policy returns are aggregated within each evaluation fragment and then averaged with equal fragment weights. Formal statistical comparisons pair each nonreference condition with C28 using matched seeds.

A downstream sensitivity analysis recalculates 2-hour returns with evaluation `gamma = 0.95`, using the same policies and evaluator rollouts. It does not retrain policies or refit evaluators.

## Repository Structure

```text
fragment_retention_rl/
├── config/                         # Experiment configuration
├── data/                           # Local input directory (private data excluded)
├── docs/                           # Protocol, parameters, and setup
├── outputs/
│   ├── fragments/                  # Cumulative retention definitions
│   ├── qa/                         # Dataset and runtime diagnostics
│   └── results/
│       ├── evaluator/              # Evaluator validation summaries
│       ├── retention/              # Formal OPE results and comparisons
│       └── sensitivity/
│           └── eval_gamma_095/     # Evaluation discount sensitivity
├── scripts/                        # Experiment and analysis commands
├── src/                            # Core implementation
└── tests/                          # Automated tests
```

## Public Reproducibility

The original BMS telemetry, detailed fragment chronology, origin-level evaluation records, fitted evaluators, and trained policy checkpoints are not distributed. Consequently, the public repository cannot independently reproduce data preparation, evaluator fitting, or policy training from the original telemetry.

The published aggregate results support downstream verification and figure generation without the private data:

```powershell
python -m pytest -q
python scripts/06_generate_results.py
python scripts/07_evaluate_gamma095_sensitivity.py
python scripts/08_verify_final_inference.py
```

Tests requiring proprietary data are skipped when it is unavailable. The read-only final verifier checks published policy-level statistics and provenance; checks requiring private origin-level records are explicitly skipped.

## Documentation

- [`docs/EVALUATOR_PROTOCOL.md`](docs/EVALUATOR_PROTOCOL.md): State and action representation, evaluator design, validation, and interpretation limits.
- [`docs/PARAMETER_PROVENANCE.md`](docs/PARAMETER_PROVENANCE.md): Formal experiment parameters and their recorded status.
- [`docs/PYCHARM_WINDOWS_SETUP.md`](docs/PYCHARM_WINDOWS_SETUP.md): Windows setup and commands for public and authorized private environments.

## Data Availability

The underlying BMS telemetry is proprietary and is not included in this repository. Source code, experiment configurations, protocols, and selected aggregate results are provided for research transparency. Authorized users with the original data can follow the full workflow described in the setup guide.
