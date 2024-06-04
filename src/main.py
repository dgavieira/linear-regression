from models.linear_regression import LinearRegression
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from metrics.pearson import pearsonr_custom


def main():
    # Carregar a base de dados
    url = './data/hospital.xls'
    df = pd.read_excel(url)
    
    # Definir as variáveis independentes (preditoras) e a variável dependente
    X = df[['Sex', 'Age', 'Weight', 'Smoker']]
    y = df[['BloodPressure_1', 'BloodPressure_2']]
    
    # Convertendo variáveis categóricas em variáveis dummy
    X = pd.get_dummies(X, drop_first=True)

    model = LinearRegression()
    
    # Realizar a validação cruzada com 5 pastas
    y_pred, fold_indices = model.cross_val_predict(X, y, cv=5)

    # Calcular e imprimir os coeficientes de Pearson e os erros médios quadráticos para cada pasta
    for i, indices in enumerate(fold_indices):
        pearson = pearsonr_custom(y.iloc[indices].values.flatten(), y_pred[indices].flatten())
        mse = mean_squared_error(y.iloc[indices].values.flatten(), y_pred[indices].flatten())
        print(f'Pasta {i + 1}:')
        print(f'   Coeficiente de Pearson: {pearson}')
        print(f'   Erro médio quadrático: {mse}')

if __name__ == "__main__":
    main()