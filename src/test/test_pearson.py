import numpy as np
import pytest

from metrics.pearson import pearsonr_custom

def test_pearsonr_custom_equal_arrays():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5])
    assert pearsonr_custom(x, y) == pytest.approx(1.0)

def test_pearsonr_custom_opposite_arrays():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([5, 4, 3, 2, 1])
    assert pearsonr_custom(x, y) == pytest.approx(-1.0)

def test_pearsonr_custom_random_arrays():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    assert pearsonr_custom(x, y) == pytest.approx(1.0)

def test_pearsonr_custom_zero_std():
    x = np.array([1, 1, 1, 1, 1])
    y = np.array([2, 4, 6, 8, 10])
    assert pearsonr_custom(x, y) == pytest.approx(0.0)

def test_pearsonr_custom_empty_input():
    x = np.array([])
    y = np.array([])
    assert pearsonr_custom(x, y) == pytest.approx(0.0)
