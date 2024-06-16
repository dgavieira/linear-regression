import numpy as np

class Adaline:
    """
    Adaline (Adaptive Linear Neuron) class for regression tasks.

    Attributes:
    ----------
    learning_rate : float
        The learning rate for weight updates.
    epochs : int
        The number of passes over the training dataset.
    random_state : int or None
        The seed for random number generation.
    """
    def __init__(self, learning_rate=0.01, epochs=50, random_state=None):
        """
        Initializes the Adaline model with given learning rate, epochs, and random state.

        Parameters:
        ----------
        learning_rate : float, optional (default=0.01)
            The learning rate for weight updates.
        epochs : int, optional (default=50)
            The number of passes over the training dataset.
        random_state : int or None, optional (default=None)
            The seed for random number generation.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state

    def purelin(self, y):
        """
        Linear activation function (identity function).

        Parameters:
        ----------
        y : array-like
            Input array.

        Returns:
        -------
        array-like
            The same input array.
        """
        return y

    def predict(self, X):
        """
        Predicts the output for given input data.

        Parameters:
        ----------
        X : array-like
            Input data.

        Returns:
        -------
        array-like
            Predicted output.
        """
        return self.purelin(np.dot(X, self.w_) + self.b_)

    def fit(self, X, y, learning_rate=None, epochs=None, random_state=None):
        """
        Trains the Adaline model on the given dataset.

        Parameters:
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data.
        y : array-like, shape = [n_samples]
            Target values.
        learning_rate : float, optional
            The learning rate for weight updates.
        epochs : int, optional
            The number of passes over the training dataset.
        random_state : int or None, optional
            The seed for random number generation.
        """
        # Update instance attributes if parameters are provided
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if epochs is not None:
            self.epochs = epochs
        if random_state is not None:
            self.random_state = random_state

        # Initialize weights and bias
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.1, size=X.shape[1])
        self.b_ = rgen.normal(loc=0.0, scale=0.1, size=1)
        self.cost_ = []

        # Training loop
        for _ in range(self.epochs):
            cost = 0
            for xi, target in zip(X, y):
                # Compute prediction
                output = self.predict(xi)
                # Compute error
                error = target - output
                # Update weights and bias
                self.w_ += 2 * self.learning_rate * error * xi
                self.b_ += 2 * self.learning_rate * error
                # Accumulate squared error
                cost += error**2
            # Store total cost for the epoch
            self.cost_.append(cost)
        return self

    def get_weights(self):
        """
        Returns the weights of the model.

        Returns:
        -------
        array-like
            Model weights.
        """
        return self.w_

    def get_bias(self):
        """
        Returns the bias of the model.

        Returns:
        -------
        array-like
            Model bias.
        """
        return self.b_

    def get_cost(self):
        """
        Returns the cost (sum of squared errors) for each epoch.

        Returns:
        -------
        list
            Cost for each epoch.
        """
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

    # Imprimindo os resultados
    print("Pesos:", adaline.get_weights())
    print("Bias:", adaline.get_bias())
    print("Custo por época:", adaline.get_cost())

