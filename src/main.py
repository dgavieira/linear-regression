import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from models.standard_scaler import StandardScaler
from models.adaline import Adaline

def load_data():
    df = pd.read_excel('./data/hospital.xls')
    df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})
    df['Smoker'] = df['Smoker'].map({True: 1, False: 0})
    return df

def preprocess_data(df):
    X = df[['Sex', 'Age', 'Weight', 'Smoker']].values
    y = df['BloodPressure_1'].values
    return X, y

def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    adaline = Adaline(learning_rate=0.01, epochs=50, random_state=1)
    adaline.fit(X_train_scaled, y_train, verbose=False)
    return adaline, scaler

def evaluate_model(model, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    pearson_coeff, _ = pearsonr(y_test, y_pred)
    mse = np.mean((y_test - y_pred)**2)
    return pearson_coeff, mse

def cross_validate(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    pearson_coeffs = []
    mean_squared_errors = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model, scaler = train_model(X_train, y_train)
        pearson_coeff, mse = evaluate_model(model, scaler, X_test, y_test)
        pearson_coeffs.append(pearson_coeff)
        mean_squared_errors.append(mse)
    return pearson_coeffs, mean_squared_errors

def save_results(pearson_coeffs, mean_squared_errors):
    results_df = pd.DataFrame({'folder': range(1, len(pearson_coeffs)+1),
                               'pearson_coeffs': pearson_coeffs,
                               'mean_squared_errors': mean_squared_errors})
    mean_pearson_coeff = np.mean(pearson_coeffs)
    mean_mse = np.mean(mean_squared_errors)
    mean_row = pd.DataFrame({'folder': 'Media', 'pearson_coeffs': mean_pearson_coeff, 'mean_squared_errors': mean_mse}, index=[0])
    results_df = pd.concat([results_df, mean_row], ignore_index=True)
    results_df.to_csv('./data/results.csv', index=False)

def plot_results(y, y_pred_all):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=y, y=y_pred_all)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel('Valores reais')
    plt.ylabel('Valores previstos')
    plt.title('Gráfico de dispersão com linha de regressão')
    plt.grid(True)
    plt.savefig('./images/scatter_plot.png')

    residuals = y - y_pred_all
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=y_pred_all, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Valores previstos')
    plt.ylabel('Resíduos')
    plt.title('Gráfico de resíduos')
    plt.grid(True)
    plt.savefig('./images/residual_plot.png')

def main():
    df = load_data()
    X, y = preprocess_data(df)
    pearson_coeffs, mean_squared_errors = cross_validate(X, y)
    save_results(pearson_coeffs, mean_squared_errors)

    # Predict for all data
    model, scaler = train_model(X, y)
    X_scaled = scaler.transform(X)
    y_pred_all = model.predict(X_scaled)
    plot_results(y, y_pred_all)

if __name__ == '__main__':
    main()
