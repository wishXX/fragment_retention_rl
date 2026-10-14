def test_state_action_shapes(pipeline):
    ds = pipeline.condition_dataset("C1")
    assert ds.S.shape == (335, 80)
    assert ds.A_physical.shape == (335, 8)
    assert ds.A_normalized.shape == (335, 8)
    assert ds.R.shape == (335,)
    assert ds.SN.shape == (335, 80)
