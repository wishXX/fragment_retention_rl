# Evaluator Protocol

This document specifies the learned dynamics evaluators used for formal offline policy evaluation (OPE). Training hyperparameters and statistical settings are recorded in [`PARAMETER_PROVENANCE.md`](PARAMETER_PROVENANCE.md).

## Evaluation Design

The formal rollout horizon is **2 hours**, with evaluation `gamma = 0.99`. A downstream sensitivity analysis changes only the evaluation discount to `gamma = 0.95` using the same policy rollouts.

Both evaluators use 1,098 eligible historical transitions. The policy action envelope of 30–50 °C constrains generated actions; it does not invalidate otherwise valid logged historical actions. No transition crosses a real telemetry gap.

The two evaluation channels are `ARX_TC_RIDGE_LOFO` and `XGBOOST_DEFAULT_LOFO`. They are qualified and reported separately, without averaging their estimated policy values or selecting an evaluator based on the resulting retention curve.

## State and Action Representation

The state has **80 dimensions**. Its observation block consists of the current hourly slice and 2 hourly lags. Each slice contains 16 quantities:

- Indoor temperature: `Tin_Mean`, `Tin_Min`, and `Tin_Std`.
- Derived underheating quantity: `Comfort_U`.
- Heating power: `RTP_01` through `RTP_09`, plus `RTP_Total`.
- Outdoor temperature: `Tout_Mean` and `Tout_Std`.

Thus, the observation block has 16 × 3 = 48 dimensions. The FTS history contains 8 controller channels over 4 historical hours, adding 8 × 4 = 32 dimensions. The current action is an 8-dimensional FTS vector. The evaluator therefore receives an **88-dimensional input** (80 state dimensions + 8 action dimensions).

## Successor Outputs

The evaluators produce **13 physical successor predictions**: `Tin_Mean`, `Tin_Min`, `Tin_Std`, `RTP_01` through `RTP_09`, and `RTP_Total`. They also predict **1 derived underheating quantity**, `Comfort_U`, calculated from indoor temperature sensor data rather than directly measured by the BMS. Together these form the existing 14-dimensional model output vector.

`Tout_Mean` and `Tout_Std` are supplied as exogenous inputs rather than learned successor outputs. The existing model and rollout implementation are unchanged.

## Natural Fragment Cross Validation

Leave one natural fragment out (LOFO) validation excludes each held-out fragment from the corresponding evaluator fit. Predictions and policy evaluations for that fragment use its held-out evaluator. Complete fragment coverage is required, and no historical transition crosses a telemetry gap.

## ARX Evaluator

The linear channel uses standardized Ridge regression with an additional temporally correlated (TC) penalty on FTS history coefficients. Its formal parameters are base Ridge alpha `1.0`, TC lambda `3.0`, and TC rho `0.99`. These settings are not selected from the retention return curve.

## XGBoost Evaluator

XGBoost is a separate nonlinear dynamics channel using the documented library default parameter policy. It follows the same state, action, successor output, cross validation, and rollout conventions as ARX. It is not an ensemble component or replacement for ARX.

## Qualification and Acceptance

Both evaluators must cover every held-out natural fragment and achieve positive **2-hour recursive RMSE skill against matched persistence** for `Tin_Mean` and `RTP_Total`. Other target errors are diagnostic, not independent acceptance criteria. `Comfort_U` is identified separately as a derived quantity rather than a directly measured physical target.

The recorded acceptance artifact marks both channels as accepted. Its documented primary results are:

| Evaluator | `Tin_Mean` RMSE | `Tin_Mean` RMSE Skill | `RTP_Total` RMSE | `RTP_Total` RMSE Skill |
| --- | ---: | ---: | ---: | ---: |
| ARX | 0.1900556267 | 0.5770792914 | 235.0250432 | 0.2444166626 |
| XGBoost | 0.3255949624 | 0.2754707944 | 221.4048875 | 0.2882041780 |

The acceptance record reports 1,098 one-step transitions and 1,075 recursive 2-hour validation origins per evaluator. These metrics qualify the dynamics models; they are not policy return estimates.

## OPE Aggregation and Limits

Returns are averaged across evaluation origins within each natural fragment and then averaged equally across fragments. Fold-standardized nearest neighbor state and action support distances are recorded as diagnostics, not added to the reward.

Learned dynamics OPE is not ground truth. Evaluator disagreement and target policy support limitations must remain visible in interpretation.

## Released Validation Artifacts

```text
outputs/results/evaluator/evaluator_validation_acceptance.json
outputs/results/evaluator/evaluator_validation.json
outputs/results/evaluator/evaluator_one_step_summary.csv
outputs/results/evaluator/evaluator_recursive_2h_summary.csv
outputs/results/evaluator/persistence_matched_one_step_summary.csv
outputs/results/evaluator/persistence_matched_recursive_2h_summary.csv
```

Private fitted evaluator folds and origin-level records are not distributed.
