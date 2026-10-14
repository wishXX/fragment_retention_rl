import numpy as np

from src.evaluation import DynamicsEvaluator


def test_dynamics_evaluator_feature_contract_uses_physical_actions():
    class CaptureModel:
        def __init__(self, value):
            self.value = float(value)
            self.last_x = None

        def predict(self, x):
            self.last_x = np.asarray(x, dtype=np.float64)
            return np.array([self.value], dtype=np.float64)

    models = tuple(CaptureModel(i) for i in range(14))
    evaluator = DynamicsEvaluator("capture", models)

    state = np.arange(80, dtype=np.float64)
    action = np.arange(30.0, 38.0, dtype=np.float64)
    pred = evaluator.predict_next(state, action)

    assert pred.shape == (14,)
    np.testing.assert_allclose(pred, np.arange(14, dtype=np.float64))

    expected_x = np.concatenate([state, action]).reshape(1, -1)
    assert expected_x.shape == (1, 88)
    for model in models:
        np.testing.assert_allclose(model.last_x, expected_x)
