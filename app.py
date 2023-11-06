from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib

app = FastAPI(title='Wine Quality Prediction')

app.add_middleware(
    CORSMiddleware,
    # Reemplaza con los orígenes permitidos en tu aplicación
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load(pathlib.Path('model/WineQT-v1.joblib'))


class InputData(BaseModel):
    # Define las características de entrada para la predicción
    fixed_acidity: float = 7.4
    volatile_acidity: float = 0.7
    citric_acid: float = 0.0
    residual_sugar: float = 1.9
    chlorides: float = 0.076
    free_sulfur_dioxide: float = 11.0
    total_sulfur_dioxide: float = 34.0
    density: float = 0.9978
    pH: float = 3.51
    sulphates: float = 0.56
    alcohol: float = 9.4


class OutputData(BaseModel):
    quality: float = 5  # Cambia el nombre del campo a 'quality' para reflejar el resultado


@app.post('/predict', response_model=OutputData)
def predict(data: InputData):
    # Convertir los datos de entrada en una matriz NumPy
    model_input = np.array(list(data.dict().values())).reshape(1, -1)

    # Realizar la predicción
    result = model.predict(model_input)

    # Crear una instancia de OutputData con el resultado
    # Asumiendo que el resultado es un valor flotante
    output_data = OutputData(quality=result[0])

    return output_data
