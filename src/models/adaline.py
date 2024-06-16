import numpy as np

class Adaline:
    def __init__(self, learning_rate=0.01, epochs=50, random_state=None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state

    def purelin(self, y):
        return y

    def predict(self, X):
        return self.purelin(np.dot(X, self.w_) + self.b_)

    def fit(self, X, y, learning_rate=None, epochs=None, random_state=None):
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if epochs is not None:
            self.epochs = epochs
        if random_state is not None:
            self.random_state = random_state

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.1, size=X.shape[1])
        self.b_ = rgen.normal(loc=0.0, scale=0.1, size=1)
        self.cost_ = []

        for _ in range(self.epochs):
            cost = 0
            for xi, target in zip(X, y):
                output = self.predict(xi)
                error = target - output
                self.w_ += 2 * self.learning_rate * error * xi
                self.b_ += 2 * self.learning_rate * error
                cost += error**2
            self.cost_.append(cost)
        return self

    def get_weights(self):
        return self.w_

    def get_bias(self):
        return self.b_

    def get_cost(self):
        return self.cost_

# Exemplo de uso
if __name__ == "__main__":
    # Gerando dados de exemplo
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([1, 2, 3, 4])

    # Criando a instância do modelo Adaline
    adaline = Adaline()

    # Treinando o modelo Adaline com parâmetros especificados
    adaline.fit(X, y, learning_rate=0.01, epochs=10, random_state=7)

    print("Pesos:", adaline.get_weights())
    print("Bias:", adaline.get_bias())
    print("Custo por época:", adaline.get_cost())
