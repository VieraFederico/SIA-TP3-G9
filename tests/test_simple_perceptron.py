import pytest
import numpy as np

from src.models.simple_perceptron import SimplePerceptron
from src.activations.step import StepActivation
from src.activations.identity import IdentityActivation
from src.activations.tanh import TanhActivation
from src.utils.random_state import make_rng


class TestSimplePerceptronInit:
    def test_weights_shape(self):
        pytest.skip("pending")

    def test_bias_shape(self):
        pytest.skip("pending")

    def test_xavier_init_variance(self):
        pytest.skip("pending")


class TestSimplePerceptronForward:
    def test_output_shape_single_sample(self):
        pytest.skip("pending")

    def test_output_shape_batch(self):
        pytest.skip("pending")

    def test_step_output_binary(self):
        pytest.skip("pending")

    def test_identity_output_linear(self):
        pytest.skip("pending")


class TestSimplePerceptronBackward:
    def test_grad_shapes_match_params(self):
        pytest.skip("pending")

    def test_zero_grad_clears_accumulators(self):
        pytest.skip("pending")


class TestSimplePerceptronSetGetParams:
    def test_set_get_roundtrip(self):
        pytest.skip("pending")
