import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from joblib import dump
import pathlib

# Cargar el dataset
df = pd.read_csv(pathlib.Path('data/WineQT.csv'))

# Definir la variable objetivo ('quality')
y = df['quality']

# Seleccionar las características relevantes para el entrenamiento
selected_features = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]
X = df[selected_features]

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Crear un modelo de Random Forest
print('Entrenando el modelo...')
# Puedes ajustar los hiperparámetros aquí
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X_train, y_train)

# Evaluar el modelo en el conjunto de prueba
y_pred = regressor.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print('Error absoluto medio (MAE):', mae)

# Guardar el modelo entrenado
print('Guardando el modelo...')
dump(regressor, pathlib.Path('model/WineQT-v1.joblib'))
