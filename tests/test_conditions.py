def test_primary_path_is_duration_descending(pipeline):
    hours = [f.observed_hours for f in pipeline.primary_order]
    assert hours[0] == max(hours)
    assert hours[1:] == sorted(hours[1:], reverse=True)