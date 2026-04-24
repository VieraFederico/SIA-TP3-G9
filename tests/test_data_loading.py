import pytest
import numpy as np
from pathlib import Path

from src.data.dataset import Dataset
from src.data.loaders import CSVDatasetLoader
from src.data.preprocessing import StandardScaler, Normalizer, OneHotEncoder
from src.data.splitter import HoldOutSplitter, KFoldSplitter
from src.utils.random_state import make_rng


class TestDataset:
    def test_len(self):
        pytest.skip("pending")

    def test_shape_mismatch_raises(self):
        pytest.skip("pending")


class TestCSVDatasetLoader:
    def test_load_missing_file_raises(self):
        pytest.skip("pending")

    def test_load_returns_correct_shapes(self):
        pytest.skip("pending")

    def test_load_unknown_target_column_raises(self):
        pytest.skip("pending")


class TestStandardScaler:
    def test_fit_stores_mean_std(self):
        pytest.skip("pending")

    def test_transform_zero_mean(self):
        pytest.skip("pending")

    def test_transform_unit_variance(self):
        pytest.skip("pending")

    def test_fit_transform_consistent(self):
        pytest.skip("pending")


class TestNormalizer:
    def test_transform_range_zero_one(self):
        pytest.skip("pending")


class TestOneHotEncoder:
    def test_transform_shape(self):
        pytest.skip("pending")

    def test_inverse_transform_roundtrip(self):
        pytest.skip("pending")


class TestHoldOutSplitter:
    def test_sizes_sum_to_total(self):
        pytest.skip("pending")

    def test_no_overlap_between_splits(self):
        pytest.skip("pending")


class TestKFoldSplitter:
    def test_returns_k_pairs(self):
        pytest.skip("pending")

    def test_val_sizes_equal(self):
        pytest.skip("pending")
