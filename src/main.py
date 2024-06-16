import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from models.standard_scaler import StandardScaler
from models.adaline import Adaline

def main():
    # Carregar a base de dados hospital.xls
    df = pd.read_excel('./data/hospital.xls')

    # Pré-processamento das variáveis categóricas
    df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})
    df['Smoker'] = df['Smoker'].map({True: 1, False: 0})

    # Separar variáveis de predição e a variável alvo
    X = df[['Sex', 'Age', 'Weight', 'Smoker']].values
    y = df['BloodPressure_1'].values

    # Configuração para a validação cruzada de 5 pastas
    kf = KFold(n_splits=5, shuffle=True, random_state=1)

    # Listas para armazenar os resultados de cada pasta
    pearson_coeffs = []
    mean_squared_errors = []

    # Inicializar o scaler
    scaler = StandardScaler()

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Normalizar os dados
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Inicializar o modelo Adaline
        adaline = Adaline(learning_rate=0.01, epochs=50, random_state=1)
        
        # Treinar o modelo Adaline
        adaline.fit(X_train_scaled, y_train)
        
        # Fazer previsões
        y_pred = adaline.predict(X_test_scaled)
        
        # Calcular o coeficiente de Pearson
        pearson_coeff, _ = pearsonr(y_test, y_pred)
        pearson_coeffs.append(pearson_coeff)
        
        # Calcular o erro médio quadrático
        mse = np.mean((y_test - y_pred)**2)
        mean_squared_errors.append(mse)

    # Calcular as médias dos resultados
    mean_pearson_coeff = np.mean(pearson_coeffs)
    mean_mse = np.mean(mean_squared_errors)

    print(f'Média dos coeficientes de Pearson: {mean_pearson_coeff:.4f}')
    print(f'Média dos erros médios quadráticos: {mean_mse:.4f}')

    # Salvar os resultados em um arquivo CSV
    results_df = pd.DataFrame({'pearson_coeffs': pearson_coeffs, 'mean_squared_errors': mean_squared_errors})
    results_df.to_csv('results.csv', index=False)

if __name__ == '__main__':
    main()