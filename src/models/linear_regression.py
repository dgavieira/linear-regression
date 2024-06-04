import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class LinearRegression:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        """
        Ajusta o modelo de regressão linear aos dados de treinamento.
        
        Args:
        X (numpy.ndarray): Matriz de características de treinamento.
        y (numpy.ndarray): Vetor de valores alvo de treinamento.
        """
        # Adicionar a coluna de 1s para o termo de bias
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Calcular a pseudoinversa de X_b
        X_b_pseudo_inverse = np.linalg.pinv(X_b)
        
        # Calcular os pesos (coeficientes) usando a fórmula da pseudoinversa
        self.weights = X_b_pseudo_inverse.dot(y)

    def predict(self, X):
        """
        Faz previsões utilizando o modelo ajustado.
        
        Args:
        X (numpy.ndarray): Matriz de características para predição.
        
        Returns:
        numpy.ndarray: Vetor de previsões.
        """
        if self.weights is None:
            raise ValueError("O modelo não foi ajustado. Chame 'fit' primeiro.")
        
        # Adicionar a coluna de 1s para o termo de bias
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Calcular as previsões
        y_pred = X_b.dot(self.weights)
        
        return y_pred
    
    def cross_val_predict(self, X, y, cv=5):
        """
        Realiza predições usando validação cruzada.
        
        Args:
        X (numpy.ndarray or pandas.DataFrame): Matriz de características.
        y (numpy.ndarray or pandas.Series): Vetor de valores alvo.
        cv (int): Número de folds para a validação cruzada.
        
        Returns:
        tuple: Vetor de previsões e lista de índices de cada fold.
        """
        # Convertendo X e y para numpy arrays, se não forem
        if isinstance(X, pd.DataFrame):
            X = X.astype(float).values
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y = y.astype(float).values
        
        kf = KFold(n_splits=cv)
        predictions = np.zeros(y.shape)
        fold_indices = []
        
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train = y[train_index]
            
            self.fit(X_train, y_train)
            predictions[test_index] = self.predict(X_test)
            fold_indices.append(test_index)
        
        return predictions, fold_indices
    