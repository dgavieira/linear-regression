import numpy as np

class Adaline:
    """
    Adaptive Linear Neuron Classifier.
    
    Parameters
    ----------
    learning_rate : float
        Learning rate (between 0.0 and 1.0)
    epochs : int
        Number of passes over the training dataset.
    random_state : int
        Seed for random number generator.
    """
    
    def __init__(self, learning_rate=0.01, epochs=50, random_state=1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
    
    def fit(self, X, y):
        """
        Fit training data.
        
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        
        Returns
        -------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.)
        self.cost_ = []
        
        for _ in range(self.epochs):
            cost = 0
            for xi, target in zip(X, y):
                output = self.activation(self.net_input(xi))
                error = target - output
                self.w_ += self.learning_rate * xi * error
                self.b_ += self.learning_rate * error
                cost += 0.5 * error**2
            self.cost_.append(cost)
            if np.isnan(cost) or np.isinf(cost):
                raise ValueError("Cost function returned NaN or Inf. Adjust learning rate.")
        return self
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_
    
    def activation(self, X):
        """Compute linear activation"""
        return X
    
    def predict(self, X):
        """Return class label after unit step"""
        return self.activation(self.net_input(X))
