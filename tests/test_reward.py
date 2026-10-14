import numpy as np


def test_zero_underheating_has_no_comfort_penalty(pipeline):
    reward = pipeline.reward_fn(0.0, pipeline.rtp_scale)
    assert np.isclose(reward, -1.0)


def test_underheating_statistic_is_weighted_linearly(pipeline):
    r0 = pipeline.reward_fn(0.0, pipeline.rtp_scale)
    r1 = pipeline.reward_fn(1.0, pipeline.rtp_scale)
    expected_extra = pipeline.cfg["reward"]["underheat_alpha"]
    assert np.isclose(r0 - r1, expected_extra)


def test_negative_predicted_underheating_is_clipped_for_reward(pipeline):
    r0 = pipeline.reward_fn(0.0, pipeline.rtp_scale)
    rneg = pipeline.reward_fn(-1.0, pipeline.rtp_scale)
    assert np.isclose(r0, rneg)
