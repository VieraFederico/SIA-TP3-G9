import pytest
import numpy as np

from src.metrics.regression import MSEMetric, MAEMetric, R2Metric
from src.metrics.classification import AccuracyMetric, PrecisionMetric, RecallMetric, F1Metric
from src.metrics.confusion_matrix import ConfusionMatrix


class TestMSEMetric:
    def test_perfect_prediction_is_zero(self):
        pytest.skip("pending")

    def test_known_value(self):
        pytest.skip("pending")


class TestMAEMetric:
    def test_perfect_prediction_is_zero(self):
        pytest.skip("pending")

    def test_known_value(self):
        pytest.skip("pending")


class TestR2Metric:
    def test_perfect_prediction_is_one(self):
        pytest.skip("pending")

    def test_mean_predictor_is_zero(self):
        pytest.skip("pending")


class TestAccuracyMetric:
    def test_all_correct(self):
        pytest.skip("pending")

    def test_all_wrong(self):
        pytest.skip("pending")

    def test_partial_accuracy(self):
        pytest.skip("pending")


class TestPrecisionMetric:
    def test_binary_precision(self):
        pytest.skip("pending")

    def test_macro_average(self):
        pytest.skip("pending")


class TestRecallMetric:
    def test_binary_recall(self):
        pytest.skip("pending")


class TestF1Metric:
    def test_binary_f1(self):
        pytest.skip("pending")

    def test_perfect_f1_is_one(self):
        pytest.skip("pending")


class TestConfusionMatrix:
    def test_shape(self):
        pytest.skip("pending")

    def test_diagonal_correct_predictions(self):
        pytest.skip("pending")
