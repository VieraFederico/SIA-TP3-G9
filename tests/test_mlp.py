import pytest
import numpy as np

from src.models.mlp import MultilayerPerceptron
from src.activations.tanh import TanhActivation
from src.activations.identity import IdentityActivation
from src.utils.random_state import make_rng


class TestMLPInit:
    def test_correct_number_of_layers(self):
        pytest.skip("pending")

    def test_layer_sizes(self):
        pytest.skip("pending")


class TestMLPForward:
    def test_output_shape(self):
        pytest.skip("pending")

    def test_output_shape_single_sample(self):
        pytest.skip("pending")

    def test_forward_deterministic(self):
        pytest.skip("pending")


class TestMLPBackward:
    def test_grads_populated_after_backward(self):
        pytest.skip("pending")

    def test_zero_grad_clears_all_layers(self):
        pytest.skip("pending")


class TestMLPGetSetParams:
    def test_params_count(self):
        pytest.skip("pending")

    def test_set_get_roundtrip(self):
        pytest.skip("pending")
