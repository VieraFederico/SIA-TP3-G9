import pytest
import numpy as np

from src.losses.mse import MSELoss
from src.losses.binary_cross_entropy import BinaryCrossEntropyLoss
from src.losses.categorical_cross_entropy import CategoricalCrossEntropyLoss


class TestMSELoss:
    def test_compute_perfect_prediction_is_zero(self):
        pytest.skip("pending")

    def test_compute_known_value(self):
        pytest.skip("pending")

    def test_gradient_shape(self):
        pytest.skip("pending")

    def test_gradient_direction(self):
        pytest.skip("pending")


class TestBinaryCrossEntropyLoss:
    def test_compute_known_value(self):
        pytest.skip("pending")

    def test_compute_perfect_prediction_near_zero(self):
        pytest.skip("pending")

    def test_gradient_shape(self):
        pytest.skip("pending")


class TestCategoricalCrossEntropyLoss:
    def test_compute_known_value(self):
        pytest.skip("pending")

    def test_gradient_shape(self):
        pytest.skip("pending")
