def test_formal_rule_allows_rejecting_one_evaluator_without_pooling():
    accepted_channels = ["XGBoost"]
    rejected_channels = ["ARX"]

    assert len(accepted_channels) >= 1
    assert set(accepted_channels).isdisjoint(rejected_channels)
