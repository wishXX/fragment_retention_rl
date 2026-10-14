from src.evaluator_protocol import _historical_dynamics_indices


def test_historical_evaluator_keeps_real_actions_outside_policy_envelope(pipeline):
    found = False

    for fragment in pipeline.fragments:
        times = list(fragment.times)
        for i in _historical_dynamics_indices(pipeline, fragment.name):
            action = pipeline.data[times[i]]["fts"]
            if not pipeline.action_space.contains(action):
                found = True
                break
        if found:
            break

    assert found, (
        "The current raw dataset is expected to contain at least one valid logged "
        "historical FTS outside the 30-50 C policy action envelope."
    )
