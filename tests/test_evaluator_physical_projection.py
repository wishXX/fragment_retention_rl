import numpy as np
import pytest

from src.evaluation import (
    DynamicsEvaluator,
    _project_endogenous_to_physical_domain,
    _project_arx_endogenous_to_physical_domain,
    _next_state_from_prediction,
)


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return np.asarray([self.value], dtype=np.float64)


@pytest.mark.parametrize("name", ["ARX_TC_RIDGE_LOFO", "XGBOOST_DEFAULT_LOFO"])
def test_evaluators_use_identical_projection(name):
    raw = np.array([19.0, 17.0, -1.0, -0.5, -2.0, 3.0, 4.0,
                    0.0, 5.0, -1.0, 2.0, 1.0, 0.5, 20.0])
    evaluator = DynamicsEvaluator(name, tuple(ConstantModel(x) for x in raw))
    actual = evaluator.predict_next(np.zeros(80), np.zeros(8))
    expected = _project_endogenous_to_physical_domain(raw)
    np.testing.assert_array_equal(actual, expected)
    assert actual[2] >= 0 and actual[3] >= 0
    assert np.all(actual[4:13] >= 0)
    assert actual[13] == pytest.approx(actual[4:13].sum(), abs=1e-12)
    np.testing.assert_array_equal(actual[:2], raw[:2])


def test_arx_projection_backward_compatible():
    raw = np.array([20., 18., -0.1, -0.2, 1., 2., 3., 4., 5., 6., 7., 8., 9., 50.])
    np.testing.assert_array_equal(
        _project_arx_endogenous_to_physical_domain(raw),
        _project_endogenous_to_physical_domain(raw),
    )


def test_unknown_evaluator_keeps_raw_predictions():
    raw = np.arange(14, dtype=float) - 4
    evaluator = DynamicsEvaluator("UNPROJECTED_TEST", tuple(ConstantModel(x) for x in raw))
    np.testing.assert_array_equal(evaluator.predict_next(np.zeros(80), np.zeros(8)), raw)


def test_projected_endogenous_enters_recursive_state():
    raw = np.array([19., 17., -1., -2., *([-3.] * 9), -27.])
    projected = _project_endogenous_to_physical_domain(raw)
    cfg = {"state": {"scalar_lags": 2, "fts_history_hours": 4},
           "signals": {"action_controllers": [str(i) for i in range(8)]}}
    state = np.zeros(80)
    next_state = _next_state_from_prediction(state, np.full(8, 40.),
                                              np.array([0., 1.]), projected, cfg)
    np.testing.assert_array_equal(next_state[:14], projected)
    np.testing.assert_array_equal(next_state[14:16], [0., 1.])
    np.testing.assert_array_equal(next_state[48:56], np.full(8, 40.))


def test_projection_rejects_nonfinite():
    raw = np.zeros(14)
    raw[3] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        _project_endogenous_to_physical_domain(raw)
