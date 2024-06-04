import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression

from models.linear_regression import LinearRegression

@pytest.fixture
def linear_regression():
    return LinearRegression()

@pytest.fixture
def sample_data():
    X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)
    return X, y

def test_fit(linear_regression, sample_data):
    X, y = sample_data
    linear_regression.fit(X, y)
    assert linear_regression.weights is not None

def test_predict(linear_regression, sample_data):
    X, y = sample_data
    linear_regression.fit(X, y)
    y_pred = linear_regression.predict(X)
    assert len(y_pred) == len(y)

def test_cross_val_predict(linear_regression, sample_data):
    X, y = sample_data
    predictions, fold_indices = linear_regression.cross_val_predict(X, y, cv=5)
    assert len(predictions) == len(y)
    assert len(fold_indices) == 5

def test_predict_without_fit(linear_regression, sample_data):
    X, y = sample_data
    with pytest.raises(ValueError):
        linear_regression.predict(X)

def test_fit_with_wrong_input(linear_regression):
    with pytest.raises(ValueError):
        linear_regression.fit(np.array([[1, 2], [3, 4]]), np.array([1, 2]))

def test_cross_val_predict_with_pandas_input(linear_regression):
    X = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1]})
    y = pd.Series([1, 2, 3, 4, 5])
    predictions, fold_indices = linear_regression.cross_val_predict(X, y, cv=5)
    assert len(predictions) == len(y)
    assert len(fold_indices) == 5
