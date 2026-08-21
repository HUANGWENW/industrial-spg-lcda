import pytest

pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from spg_lcda.training.e5_f_trainer import SourceGroupedBatchSampler


def test_source_groups_stay_together_and_each_sample_appears_once() -> None:
    groups = [list(range(start, start + 4)) for start in range(0, 12, 4)]
    batches = list(SourceGroupedBatchSampler(groups, batch_size=8, seed=42))

    assert sorted(index for batch in batches for index in batch) == list(range(12))
    for group in groups:
        assert any(set(group).issubset(batch) for batch in map(set, batches))
