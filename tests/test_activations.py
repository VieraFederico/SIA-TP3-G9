import pytest
import numpy as np

from src.activations.step import StepActivation
from src.activations.identity import IdentityActivation
from src.activations.tanh import TanhActivation
from src.activations.logistic import LogisticActivation
from src.activations.relu import ReLUActivation


class TestStepActivation:
    def test_forward_positive(self):
        pytest.skip("pending")

    def test_forward_negative(self):
        pytest.skip("pending")

    def test_forward_zero(self):
        pytest.skip("pending")

    def test_is_not_differentiable(self):
        pytest.skip("pending")

    def test_derivative_raises(self):
        pytest.skip("pending")


class TestIdentityActivation:
    def test_forward_returns_input(self):
        pytest.skip("pending")

    def test_derivative_is_ones(self):
        pytest.skip("pending")

    def test_is_differentiable(self):
        pytest.skip("pending")


class TestTanhActivation:
    def test_forward_beta_1(self):
        pytest.skip("pending")

    def test_forward_beta_custom(self):
        pytest.skip("pending")

    def test_derivative_formula(self):
        pytest.skip("pending")

    def test_derivative_range(self):
        pytest.skip("pending")

    def test_is_differentiable(self):
        pytest.skip("pending")


class TestLogisticActivation:
    def test_forward_output_range(self):
        pytest.skip("pending")

    def test_forward_zero_gives_half(self):
        pytest.skip("pending")

    def test_derivative_formula(self):
        pytest.skip("pending")

    def test_is_differentiable(self):
        pytest.skip("pending")


class TestReLUActivation:
    def test_forward_positive(self):
        pytest.skip("pending")

    def test_forward_negative_is_zero(self):
        pytest.skip("pending")

    def test_derivative_positive(self):
        pytest.skip("pending")

    def test_derivative_negative_is_zero(self):
        pytest.skip("pending")
