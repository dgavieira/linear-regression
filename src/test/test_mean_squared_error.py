import numpy as np
import pytest

from metrics.mean_squared_error import mean_squared_error_custom

def test_mean_squared_error_custom_equal_arrays():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])
    assert mean_squared_error_custom(y_true, y_pred) == pytest.approx(0.0)

def test_mean_squared_error_custom_different_arrays():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([2, 3, 4, 5, 6])
    assert mean_squared_error_custom(y_true, y_pred) == pytest.approx(1.0)

def test_mean_squared_error_custom_empty_input():
    y_true = np.array([])
    y_pred = np.array([])
    assert mean_squared_error_custom(y_true, y_pred) == pytest.approx(0.0)

def test_mean_squared_error_custom_single_value():
    y_true = np.array([5])
    y_pred = np.array([3])
    assert mean_squared_error_custom(y_true, y_pred) == pytest.approx(4.0)

def test_mean_squared_error_custom_large_arrays():
    y_true = np.random.rand(1000)
    y_pred = np.random.rand(1000)
    mse_custom = mean_squared_error_custom(y_true, y_pred)
    mse_numpy = np.mean((y_true - y_pred) ** 2)
    assert mse_custom == pytest.approx(mse_numpy)
