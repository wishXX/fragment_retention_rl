import inspect

from src.evaluation import rollout_policy


def test_rollout_policy_accepts_support_index_argument():
    signature = inspect.signature(rollout_policy)

    assert "support_index" in signature.parameters
    assert signature.parameters["support_index"].default is None
