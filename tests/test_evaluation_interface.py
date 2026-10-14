import numpy as np

from src.evaluation import EvaluationOrigin, _next_state_from_prediction


def test_next_state_preserves_expected_dimension():
    cfg = {
        "state": {"scalar_lags": 2, "fts_history_hours": 4},
        "signals": {"action_controllers": [f"a{i}" for i in range(8)]},
    }
    state = np.arange(80, dtype=np.float64)
    action = np.full(8, 40.0, dtype=np.float64)
    next_exogenous = np.array([-10.0, 1.5], dtype=np.float64)
    next_endogenous = np.arange(14, dtype=np.float64)

    nxt = _next_state_from_prediction(
        state,
        action,
        next_exogenous,
        next_endogenous,
        cfg,
    )

    assert nxt.shape == (80,)
    np.testing.assert_allclose(nxt[:14], next_endogenous)
    np.testing.assert_allclose(nxt[14:16], next_exogenous)
    np.testing.assert_allclose(nxt[16:48], state[:32])
    np.testing.assert_allclose(nxt[48:56], action)
    np.testing.assert_allclose(nxt[56:], state[48:72])


def test_evaluation_origin_keeps_physical_state():
    state = np.arange(80, dtype=np.float64)
    future_exogenous = np.array(
        [[-10.0, 1.0], [-9.0, 1.2]],
        dtype=np.float64,
    )
    origin = EvaluationOrigin(
        fragment="F1",
        time="t0",
        state_physical=state,
        future_exogenous=future_exogenous,
    )

    np.testing.assert_array_equal(origin.state_physical, state)
    np.testing.assert_array_equal(origin.future_exogenous, future_exogenous)
    assert origin.fragment == "F1"
    assert origin.time == "t0"
