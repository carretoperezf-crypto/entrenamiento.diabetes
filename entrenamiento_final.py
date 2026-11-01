import os
from datetime import date

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score, recall_score
from xgboost import XGBClassifier

archivo = "data/Diabetes_Mexico_DATASET_codificado.xlsx"
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

#VARIABLES CATEGÓRICAS (para SMOTENC)
categoricas = [
    "sexo", "Antecedente diabetes familiar", "Antecedente hipertension familiar",
    "Hipertension", "Actividad fisica", "Tabaquismo"
]
indices_categoricas = [columnas.index(c) for c in categoricas]

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
    objective="multi:softprob",
    eval_metric="mlogloss",
    num_class=3
)

pipe=Pipeline([
    ("smote", SMOTENC(categorical_features=indices_categoricas, random_state=42)),
    ("xgb", modelo_xgb)
])  

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
puntajes = cross_val_score(pipe, X, Y, cv=cv, scoring="f1_macro")

print("\n=== PROMEDIO (VALIDACIÓN CRUZADA, 5 FOLDS) ===")
print("F1 de cada entrenamiento:", np.round(puntajes, 4))
print("F1 promedio:", round(puntajes.mean(), 4), "±", round(puntajes.std(), 4))

# Probabilidades "honestas": cada paciente es predicho por un modelo que no lo vio
proba_cv = cross_val_predict(pipe, X, Y, cv=cv, method="predict_proba")

def clasificar_con_umbrales(proba, umbral_medio, umbral_alto):
    pred = np.zeros(len(proba), dtype=int)                  # por defecto: bajo
    pred[(proba[:, 1] + proba[:, 2]) >= umbral_medio] = 1   # al menos medio
    pred[proba[:, 2] >= umbral_alto] = 2                    # alto
    return pred

# Referencia: cómo le va a argmax
pred_argmax = proba_cv.argmax(axis=1)
print("\n=== ARGMAX ===")
print("F1 macro:", round(f1_score(Y, pred_argmax, average="macro"), 4))
print("Recall riesgo alto:", round(recall_score(Y, pred_argmax, labels=[2], average=None)[0], 4))

# Búsqueda de umbrales
RECALL_MINIMO_ALTO = 0.85   # detectar al menos 85% de los casos de riesgo alto
mejor = None

for u_medio in np.arange(0.20, 0.81, 0.05):
    for u_alto in np.arange(0.15, 0.81, 0.05):
        pred = clasificar_con_umbrales(proba_cv, u_medio, u_alto)
        recall_alto = recall_score(Y, pred, labels=[2], average=None)[0]
        if recall_alto < RECALL_MINIMO_ALTO:
            continue
        f1 = f1_score(Y, pred, average="macro")
        if mejor is None or f1 > mejor["f1"]:
            mejor = {"f1": f1, "medio": round(u_medio, 2),
                     "alto": round(u_alto, 2), "recall_alto": recall_alto}

if mejor is None:
    print("\nNingún par de umbrales alcanzó el recall mínimo; se usará argmax.")
    umbrales = None
else:
    umbrales = {"medio": mejor["medio"], "alto": mejor["alto"]}
    print("\n=== UMBRALES ELEGIDOS ===")
    print("Umbral medio:", umbrales["medio"], "| Umbral alto:", umbrales["alto"])
    print("F1 macro:", round(mejor["f1"], 4))
    print("Recall riesgo alto:", round(mejor["recall_alto"], 4))

# ENTRENAMIENTO FINAL CON TODOS LOS DATOS
#print("\n=== ENTRENANDO XGBOOST ===")
#modelo_xgb.fit(X, Y)
#print("XGBoost terminado")
pipe.fit(X, Y)
modelo_final = pipe.named_steps["xgb"]

# Variables que más influyen en el modelo
importancias = pd.Series(
    modelo_final.feature_importances_,
    index=columnas
).sort_values(ascending=False)

print("\n=== IMPORTANCIA DE LAS VARIABLES ===")
print(importancias.round(4).to_string())


# GUARDADO DEL PAQUETE
os.makedirs("modelos", exist_ok=True)   # crea la carpeta si no existe
paquete = {
    "modelo": modelo_final,
    "columnas": columnas,
    "clases": {0: "Riesgo bajo", 1: "Riesgo medio", 2: "Riesgo alto"},
    "umbrales": umbrales,
    "f1_promedio": puntajes.mean(),
    "fecha": str(date.today())
}

archivo_modelo = "modelos/modelo_xgboost.pkl"
joblib.dump(paquete, archivo_modelo)
print("\nArchivo del modelo guardado en:", archivo_modelo)


# PRUEBA DE CARGA (así se carga en la app)
paquete_cargado = joblib.load(archivo_modelo)
modelo_cargado = paquete_cargado["modelo"]

print("\n=== MODELO XGBOOST CARGADO ===")
print("Fecha de entrenamiento:", paquete_cargado["fecha"])
print("F1 promedio:", round(paquete_cargado["f1_promedio"], 4))
print("Columnas esperadas:", len(paquete_cargado["columnas"]))

# Prueba rápida: predecir el primer registro del dataset
ejemplo = X.iloc[[0]][paquete_cargado["columnas"]]
probabilidades = modelo_cargado.predict_proba(ejemplo)[0]

umbrales_cargados = paquete_cargado["umbrales"]
if umbrales_cargados:
    clase = int(clasificar_con_umbrales(
        probabilidades.reshape(1, -1),
        umbrales_cargados["medio"], umbrales_cargados["alto"]
    )[0])
else:
    clase = int(np.argmax(probabilidades))

print("\n=== PRUEBA DE PREDICCIÓN ===")
print("Resultado:", paquete_cargado["clases"][clase])
for numero, nombre in paquete_cargado["clases"].items():
    print(f"  {nombre}: {probabilidades[numero] * 100:.1f}%")

# Cuántos datos usó el entrenamiento final
X_bal, Y_bal = SMOTENC(categorical_features=indices_categoricas,
                       random_state=42).fit_resample(X, Y)

print("\n=== DATOS DEL ENTRENAMIENTO FINAL ===")
print("Registros originales:", len(X))
print("Registros después de SMOTENC:", len(X_bal))
print("Registros sintéticos creados:", len(X_bal) - len(X))
print("Distribución balanceada:")
print(Y_bal.value_counts().sort_index())