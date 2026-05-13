"""Tests for data pipeline. These run on CPU without network access."""
import pytest
import numpy as np
import torch
from src.data.dataset import (
    FER_EMOTION_TO_STRESS, FER_INT_TO_EMOTION,
    STRESS_CLASS_NAMES, FATIGUE_CLASS_NAMES,
    _hash_subject, SplitIndices,
)


class TestLabelMaps:
    def test_emotion_to_stress_covers_all_classes(self):
        for emotion in FER_INT_TO_EMOTION.values():
            assert emotion in FER_EMOTION_TO_STRESS

    def test_stress_class_count(self):
        assert len(STRESS_CLASS_NAMES) == 3
        assert STRESS_CLASS_NAMES == ["low", "moderate", "high"]

    def test_fatigue_class_count(self):
        assert len(FATIGUE_CLASS_NAMES) == 2

    def test_stress_values_in_range(self):
        for v in FER_EMOTION_TO_STRESS.values():
            assert 0 <= v <= 2

    def test_hash_subject_deterministic(self):
        assert _hash_subject("foo.jpg") == _hash_subject("foo.jpg")
        assert _hash_subject("foo.jpg") != _hash_subject("bar.jpg")


class TestSplitIndices:
    def test_dataclass_construction(self):
        s = SplitIndices(train=[1, 2], val=[3], test=[4])
        assert s.train == [1, 2]
        assert s.val == [3]
        assert s.test == [4]


def test_transforms_import():
    """Importing transforms shouldn't crash even without GPU."""
    from src.data.transforms import build_inference_transform
    tf = build_inference_transform(224)
    assert tf is not None
