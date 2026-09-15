import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
#METRICAS 
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

archivo= "data/Diabetes_Mexico_DATASET_SMOTE.xlsx"
df=pd.read_excel(archivo)

print("\n === DISTRIBUCION DE CLASES ====")
print(df["riesgo_diabetes_cat"].value_counts())

columnas= [
    "sexo",
    "edad",
    "Peso",
    "Estatura(cm)",
    "IMC",
    "muestra_suero",
    "insulina",
    "glu_suero",
    "creat",
    "colest",
    "trig",
    "Antecedente diabetes familiar",
    "Antecedente hipertension familiar",
    "Tabaquismo",
    "Actividad fisica"]

X = df[columnas]
Y = df["riesgo_diabetes_cat"]

print("Variables de entrada:", X.shape[1])
print("Registros:", X.shape[0])

#COLUMNAS UTILIZADAS 
for columna in X.columns:
    print("-",columna)

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

print("\n=== ENTRENANDO XGBOOST ===")
modelo_xgb.fit( X, Y)
print("XGBoost terminado")

archivo_modelo = "modelos/modelo_xgboost.pkl"
joblib.dump(modelo_xgb, archivo_modelo)
print("Archivo del modelo guardado en:", archivo_modelo)

modelo_cargado = joblib.load(archivo_modelo)
print("\n=== MODELO XGBOOST CARGADO ===")


