# Parameter Provenance

This record lists the settings used in the formal experiment. For model definitions and validation criteria, see [`EVALUATOR_PROTOCOL.md`](EVALUATOR_PROTOCOL.md). Values are descriptive of the completed experiment; changing a configuration file does not retroactively change trained policies or saved results.

## Dataset and Retention

| Parameter | Formal Value or Rule |
| --- | --- |
| Natural fragments | Contiguous valid hourly observations; no cross-gap transitions |
| Fragment ordering | Longest first, then descending valid duration |
| Retention conditions | C1, C2, C3, C5, C10, C16, C19, C28 |
| Eligible transitions | 1,098 |
| Scalar observation lags | 2 hourly lags |
| FTS history | 4 hours |
| PM eligibility | Offsets -2, -1, 0, +1; required value 100%; tolerance 1e-6 |
| Policy FTS action envelope | 30–50 °C |
| State normalization | Fitted separately for each retention condition |

The FTS envelope constrains policy generated actions but is not a filter for otherwise valid logged historical actions. Historical actions are not clipped to this envelope.

## State, Action, and Reward

| Parameter | Formal Value or Rule |
| --- | --- |
| State representation | 80 dimensions: 3 × 16 observation quantities + 4 × 8 FTS history values |
| Current FTS action | 8 dimensions |
| Evaluator input | 88 dimensions |
| Derived underheating quantity | `Comfort_U`; not a directly measured BMS variable |
| Comfort lower threshold | 18 °C |
| Underheating penalty | Quadratic below 18 °C |
| Underheating alpha | 1.5008240210556434 |
| Explicit overheating penalty | None |
| RTP normalization reference | Mean next state RTP over eligible C1 transitions |
| RTP normalization value | 1178.965671641791 |

The evaluators predict 13 physical successor quantities and 1 derived underheating quantity (`Comfort_U`). This distinction describes the existing implementation without changing its output dimensions or evaluation procedure.

## Offline RL Training

| Parameter | Formal Value |
| --- | --- |
| Algorithms | IQL and CQL |
| Training discount | `gamma = 0.99` |
| Matched seeds | 1–20 |
| Retention conditions | 8 |
| Trained policies | 320 |
| Batch size | 256 |
| Training budget | 256 expected sample exposures per retained transition |

### IQL

| Parameter | Formal Value |
| --- | --- |
| Actor, value, and critic learning rates | `3e-4` each |
| Hidden dimensions | `[256, 256]` |
| Expectile | 0.7 |
| Temperature | 3.0 |
| Target tau | 0.005 |
| Maximum advantage weight | 100 |

### CQL

| Parameter | Formal Value |
| --- | --- |
| Policy learning rate | `1e-4` |
| Q learning rate | `3e-4` |
| Minimum Q weight | 5 |
| Hidden dimensions | `[256, 256, 256]` |
| Target tau | 0.005 |
| Temperature | 1.0 |
| Random actions | 10 |
| Automatic entropy tuning | True |
| Deterministic backup | True |
| Minimum Q version | 3 |
| Policy evaluation / behavior cloning warm start fraction | 0.04 |

The configuration retains earlier CQL candidate policy learning rates `{1e-4, 3e-4}` and minimum Q weights `{5, 10}`. The formal retention runner uses the single fixed configuration above.

## Evaluators and OPE

| Parameter | Formal Value or Rule |
| --- | --- |
| Evaluation horizon | 2 hours |
| Primary evaluation discount | `gamma = 0.99` |
| Validation | Leave one natural fragment out |
| Evaluators | `ARX_TC_RIDGE_LOFO`, `XGBOOST_DEFAULT_LOFO` |
| ARX base Ridge alpha | 1.0 |
| ARX TC lambda | 3.0 |
| ARX TC rho | 0.99 |
| XGBoost settings | Documented library defaults |
| Evaluator successor outputs | 13 physical predictions + 1 derived `Comfort_U` quantity (14 dimensions total) |
| Required persistence skill | Positive 2-hour RMSE skill for `Tin_Mean` and `RTP_Total` |
| Support diagnostic | Fold standardized nearest neighbor distance |
| Return aggregation | Mean within fragment, then equal mean across fragments |

The two evaluator channels remain separate and are not selected using retention performance.

## Statistical Comparisons

| Parameter | Formal Value or Rule |
| --- | --- |
| Reference | C28 |
| Comparisons | Each of C1, C2, C3, C5, C10, C16, C19 against C28 |
| Pairing | Matched seeds within each algorithm and evaluator |
| Confidence intervals | 20,000 paired bootstrap resamples; 95% interval |
| Hypothesis test | Two sided paired Wilcoxon signed rank |
| Effect size | Paired Cohen's `dz` |
| Multiple comparison adjustment | Holm over 7 comparisons separately per algorithm and evaluator |
| Other diagnostics | Mean, median, and paired seed direction counts |

No best condition is selected inferentially from the observed return curve.

## Evaluation Discount Sensitivity

Both training and primary OPE use `gamma = 0.99`. The sensitivity analysis changes **only the evaluation discount** to `gamma = 0.95`, retaining the same 320 policies, fitted evaluators, and 2-hour rollouts. It does not retrain policies, refit evaluators, or generate new rollouts.

## Interpretation Limits

The evaluators estimate policy values using learned dynamics, not direct environment interaction. Their predictive qualification, support diagnostics, and disagreements should be considered when interpreting results.
