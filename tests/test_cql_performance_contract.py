from pathlib import Path


def test_cql_candidate_sampling_is_vectorized():
    root = Path(__file__).resolve().parents[1]
    text = (root / "src" / "models" / "cql.py").read_text()

    assert "def _sample_actor_n" in text
    assert "for _ in range(num_random)" not in text
    assert "candidate_actions = torch.cat" in text
    assert "torch.split" in text


def test_cql_dataset_tensors_are_device_resident():
    root = Path(__file__).resolve().parents[1]
    text = (root / "src" / "models" / "cql.py").read_text()

    assert "torch.as_tensor(S_np" in text
    assert "torch.as_tensor(SN_np" in text
    assert "index_select" in text
