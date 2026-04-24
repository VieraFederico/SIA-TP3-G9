import pytest
import numpy as np

from src.optimizers.gradient_descent import GradientDescentOptimizer
from src.optimizers.momentum import MomentumOptimizer
from src.optimizers.adam import AdamOptimizer


class TestGradientDescentOptimizer:
    def test_update_decreases_params(self):
        pytest.skip("pending")

    def test_update_step_size(self):
        pytest.skip("pending")

    def test_reset_no_op(self):
        pytest.skip("pending")


class TestMomentumOptimizer:
    def test_first_update_matches_gd(self):
        pytest.skip("pending")

    def test_velocity_accumulates(self):
        pytest.skip("pending")

    def test_reset_clears_velocities(self):
        pytest.skip("pending")


class TestAdamOptimizer:
    def test_update_changes_params(self):
        pytest.skip("pending")

    def test_step_counter_increments(self):
        pytest.skip("pending")

    def test_reset_clears_moments(self):
        pytest.skip("pending")

    def test_bias_correction_first_step(self):
        pytest.skip("pending")
