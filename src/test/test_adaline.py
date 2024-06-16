import pytest
import numpy as np
from models.adaline import Adaline

# Função auxiliar para criar dados de teste
def create_test_data():
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([1, 2, 3, 4])
    return X, y

# Teste para a inicialização da classe Adaline
def test_adaline_init():
    adaline = Adaline(learning_rate=0.05, epochs=100, random_state=42)
    assert adaline.learning_rate == 0.05
    assert adaline.epochs == 100
    assert adaline.random_state == 42

# Teste para a função de ativação linear
def test_purelin():
    adaline = Adaline()
    y = np.array([1, 2, 3])
    assert np.array_equal(adaline.purelin(y), y)

# Teste para o método predict
def test_predict():
    adaline = Adaline()
    X, y = create_test_data()
    adaline.fit(X, y)
    predictions = adaline.predict(X)
    assert predictions.shape == (4,)

# Teste para o método fit
def test_fit():
    adaline = Adaline(learning_rate=0.01, epochs=10, random_state=1)
    X, y = create_test_data()
    adaline.fit(X, y)
    assert len(adaline.cost_) == 10
    assert adaline.w_.shape == (2,)
    assert isinstance(adaline.b_, np.ndarray)

# Teste para o método get_weights
def test_get_weights():
    adaline = Adaline()
    X, y = create_test_data()
    adaline.fit(X, y)
    weights = adaline.get_weights()
    assert weights.shape == (2,)

# Teste para o método get_bias
def test_get_bias():
    adaline = Adaline()
    X, y = create_test_data()
    adaline.fit(X, y)
    bias = adaline.get_bias()
    assert isinstance(bias, np.ndarray)
    assert bias.shape == (1,)

# Teste para o método get_cost
def test_get_cost():
    adaline = Adaline()
    X, y = create_test_data()
    adaline.fit(X, y)
    cost = adaline.get_cost()
    assert len(cost) == adaline.epochs

# Executar todos os testes
if __name__ == "__main__":
    pytest.main()

