import numpy as np


def test_action_mapping_round_trip(pipeline):
    a = np.array([30.0, 32.5, 35.0, 37.5, 40.0, 42.5, 45.0, 50.0])
    z = pipeline.action_space.encode(a)
    back = pipeline.action_space.decode(z)
    assert np.allclose(a, back)
    assert np.isclose(z[0], -1.0)
    assert np.isclose(z[-1], 1.0)


def test_historical_actions_are_not_filtered_or_clipped_by_policy_envelope(pipeline):
    ds = pipeline.all_dataset()

    assert len(ds) == 1098

    below_primary = ds.A_physical < pipeline.action_space.lower_c
    above_primary = ds.A_physical > pipeline.action_space.upper_c

    assert below_primary.any()
    assert not above_primary.any()

    assert (ds.A_normalized[below_primary] < -1.0).all()

    decoded = pipeline.action_space.decode(ds.A_normalized)
    assert np.allclose(decoded, ds.A_physical)


def test_policy_action_clipping_respects_primary_envelope(pipeline):
    z = np.array([-1.5, -1.0, 0.0, 1.0, 1.5])
    clipped = pipeline.action_space.clip_normalized(z)

    assert np.all(clipped >= -1.0)
    assert np.all(clipped <= 1.0)

    physical = pipeline.action_space.decode(clipped)
    assert np.all(physical >= pipeline.action_space.lower_c)
    assert np.all(physical <= pipeline.action_space.upper_c)