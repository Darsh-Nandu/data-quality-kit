import pandas as pd


def test_from_dataframe_summary():
    from dqk.core.dataset import DQKDataset

    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    ds = DQKDataset.from_dataframe(df)
    summary = ds.summary()

    assert summary["n_rows"] == 3
    assert summary["n_cols"] == 2
    names = {c["name"] for c in summary["columns"]}
    assert "a" in names
    assert "b" in names


def test_sample_and_len():
    from dqk.core.dataset import DQKDataset

    df = pd.DataFrame({"x": list(range(10))})
    ds = DQKDataset.from_dataframe(df)
    assert len(ds) == 10
    sampled = ds.sample(5)
    assert isinstance(sampled, DQKDataset)
    assert len(sampled) <= 5
