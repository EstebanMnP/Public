'''
Proyecto de Machine Learning realizado por Esteban Mangado.

Se busca comparar el resultado a la hora de intentar predecir la victoria en una partida de LoL
empleando datos de partidas hasta el minuto 10 y hasta el minuto 15

Después de entrenar el modelo, la clase está programda para calcular el Accuracy y la matriz de confusión
'''

# Librerías
from funciones import *
from models import Models
import pandas as pd
import numpy as np


# Cargar datos procesados
datos_15min = pd.read_csv('../data/processed_data/Challenger_Ranked_Games_15minute_processed.csv')
datos_10min = pd.read_csv('../data/processed_data/Challenger_Ranked_Games_10minute_processed.csv')

# División de los datos
X_train_15min, X_test_15min, y_train_15min, y_test_15min, _, _ = dividir_datos(datos_15min)
X_train_10min, X_test_10min, y_train_10min, y_test_10min, _, _ = dividir_datos(datos_10min)

#Se crean los objetos
partidas_15min = Models(X_train_15min, X_test_15min, y_train_15min, y_test_15min)
partidas_10min = Models(X_train_10min, X_test_10min, y_train_10min, y_test_10min)

# Entrenamiento 
modelo_entrenado_15min, _, _, _, _ = partidas_15min.randomforest(metricas = False)
modelo_entrenado_10min, _, _, _, _ = partidas_10min.randomforest(metricas = False)

# Guardar modelos
guardar_modelo(modelo_entrenado_15min,"randomforest_15min")
guardar_modelo(modelo_entrenado_15min,"randomforest_10min")