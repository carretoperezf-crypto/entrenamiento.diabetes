import os
from datetime import date

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

archivo = "data/Diabetes_Mexico_DATASET_SMOTE.xlsx"
df = pd.read_excel(archivo)

print("\n === DISTRIBUCION DE CLASES ====")
print(df["riesgo_diabetes_cat"].value_counts())

columnas = [
    "sexo",
    "edad",
    "Peso",
    "Estatura(cm)",
    "IMC",
    "Antecedente diabetes familiar",
    "Antecedente hipertension familiar",
    "Hipertension",
    "Actividad fisica",
    "Tabaquismo"
]

X = df[columnas]
Y = df["riesgo_diabetes_cat"]

print("Variables de entrada:", X.shape[1])
print("Registros:", X.shape[0])

#COLUMNAS UTILIZADAS
for columna in X.columns:
    print("-", columna)

# ==========================================
# XGBOOST
# ==========================================
modelo_xgb = XGBClassifier(
    random_state=42,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    objective="multi:softmax",
    eval_metric="mlogloss",
    num_class=3
)

# ==========================================
# 1. PROMEDIO (VALIDACIÓN CRUZADA, 5 FOLDS)
#    Aquí se declara puntajes
# ==========================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
puntajes = cross_val_score(modelo_xgb, X, Y, cv=cv, scoring="f1_macro")

print("\n=== PROMEDIO (VALIDACIÓN CRUZADA, 5 FOLDS) ===")
print("F1 de cada entrenamiento:", np.round(puntajes, 4))
print("F1 promedio:", round(puntajes.mean(), 4), "±", round(puntajes.std(), 4))

# ==========================================
# 2. ENTRENAMIENTO FINAL CON TODOS LOS DATOS
# ==========================================
print("\n=== ENTRENANDO XGBOOST ===")
modelo_xgb.fit(X, Y)
print("XGBoost terminado")

# Variables que más influyen en el modelo
importancias = pd.Series(
    modelo_xgb.feature_importances_,
    index=columnas
).sort_values(ascending=False)

print("\n=== IMPORTANCIA DE LAS VARIABLES ===")
print(importancias.round(4).to_string())

# ==========================================
# 3. GUARDADO DEL PAQUETE
# ==========================================
os.makedirs("modelos", exist_ok=True)   # crea la carpeta si no existe

archivo_modelo = "modelos/modelo_xgboost.pkl"

paquete = {
    "modelo": modelo_xgb,
    "columnas": columnas,
    "clases": {0: "Riesgo bajo", 1: "Riesgo medio", 2: "Riesgo alto"},
    "f1_promedio": puntajes.mean(),
    "fecha": str(date.today())
}

joblib.dump(paquete, archivo_modelo)
print("\nArchivo del modelo guardado en:", archivo_modelo)

# ==========================================
# 4. PRUEBA DE CARGA (así se carga en la app)
# ==========================================
paquete_cargado = joblib.load(archivo_modelo)
modelo_cargado = paquete_cargado["modelo"]

print("\n=== MODELO XGBOOST CARGADO ===")
print("Fecha de entrenamiento:", paquete_cargado["fecha"])
print("F1 promedio:", round(paquete_cargado["f1_promedio"], 4))
print("Columnas esperadas:", len(paquete_cargado["columnas"]))

# Prueba rápida: predecir el primer registro del dataset
ejemplo = X.iloc[[0]][paquete_cargado["columnas"]]
probabilidades = modelo_cargado.predict_proba(ejemplo)[0]
clase = int(np.argmax(probabilidades))

print("\n=== PRUEBA DE PREDICCIÓN ===")
print("Resultado:", paquete_cargado["clases"][clase])
for numero, nombre in paquete_cargado["clases"].items():
    print(f"  {nombre}: {probabilidades[numero] * 100:.1f}%")