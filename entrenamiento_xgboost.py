import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

# =====================================================================
# 1. CARGA DE DATOS
# =====================================================================
archivo = "data/Diabetes_Mexico_DATASET_SMOTE.xlsx"
OBJETIVO = "riesgo_diabetes_cat"

df = pd.read_excel(archivo)

# =====================================================================
# 2. ETIQUETAS DE TEXTO PARA riesgo_diabetes_cat
# =====================================================================
# Orden en que quieres ver las categorías (de menor a mayor riesgo).
ORDEN_RIESGO = ["Bajo", "Medio", "Alto"]

# Solo se usa si en el Excel la columna viene con números en lugar de texto.
# REVISA que cada número corresponda a la categoría correcta en tu dataset.
NOMBRES_SI_ES_NUMERICO = {0: "Bajo", 1: "Medio", 2: "Alto"}

if pd.api.types.is_numeric_dtype(df[OBJETIVO]):
    y_texto = df[OBJETIVO].map(NOMBRES_SI_ES_NUMERICO)
    sin_nombre = df.loc[y_texto.isna(), OBJETIVO].unique()
    if len(sin_nombre) > 0:
        raise ValueError(f"Códigos sin nombre en NOMBRES_SI_ES_NUMERICO: {sin_nombre}")
else:
    y_texto = df[OBJETIVO].astype(str).str.strip()

# Respeta ORDEN_RIESGO si coincide con los valores del dataset
# (sin importar mayúsculas); si no, usa orden alfabético.
valores = list(y_texto.unique())
por_minuscula = {v.lower(): v for v in valores}
if set(por_minuscula) <= {o.lower() for o in ORDEN_RIESGO}:
    nombres_clases = [por_minuscula[o.lower()] for o in ORDEN_RIESGO if o.lower() in por_minuscula]
else:
    nombres_clases = sorted(valores)
    print(f"Aviso: las categorías {nombres_clases} no coinciden con ORDEN_RIESGO; "
          "se usará orden alfabético.")

# XGBoost necesita clases numéricas 0, 1, 2...; el texto se usa solo para mostrar.
codigo_clase = {nombre: i for i, nombre in enumerate(nombres_clases)}

print("\n=== CORRESPONDENCIA DE CLASES ===")
for nombre, codigo in codigo_clase.items():
    print(f"{codigo} -> {nombre}")

print("\n=== DISTRIBUCION DE CLASES ===")
print(y_texto.value_counts().reindex(nombres_clases))

X = df.drop(OBJETIVO, axis=1)
Y = y_texto.map(codigo_clase)

print("\n=== DATOS SEPARADOS ===")
print("X:", X.shape)
print("Y:", Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42,
    stratify=Y,
)

# No se usa StandardScaler: XGBoost (árboles de decisión) no necesita escalado.

print("\n=== DIVISION TRAIN / TEST ===")
print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)
print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)

print("\n=== DISTRIBUCION TRAIN ===")
print(Y_train.map(dict(enumerate(nombres_clases))).value_counts().reindex(nombres_clases))
print("\n=== DISTRIBUCION TEST ===")
print(Y_test.map(dict(enumerate(nombres_clases))).value_counts().reindex(nombres_clases))

# =====================================================================
# 4. MODELO
# =====================================================================
modelo = XGBClassifier(
    random_state=42,
    n_estimators=15,
    learning_rate=0.1,
    max_depth=1,
    objective="multi:softprob",
    eval_metric="mlogloss",
)

print("\n=== ENTRENANDO XGBOOST ===")
modelo.fit(
    X_train,
    Y_train,
    eval_set=[(X_train, Y_train), (X_test, Y_test)],
    verbose=False,
)
print("Entrenamiento terminado correctamente")

# =====================================================================
# 5. PREDICCIONES
# =====================================================================
y_pred = modelo.predict(X_test)
y_proba = modelo.predict_proba(X_test)
clases = modelo.classes_
etiquetas = [nombres_clases[c] for c in clases]

print("\n=== PREDICCIONES REALIZADAS (primeras 20) ===")
print([nombres_clases[p] for p in y_pred[:20]])

# =====================================================================
# 6. METRICAS
# =====================================================================
print("\n=== METRICAS GENERALES (CONJUNTO DE PRUEBA) ===")
print(f"Accuracy:                {accuracy_score(Y_test, y_pred):.4f}")
print(f"Balanced accuracy:       {balanced_accuracy_score(Y_test, y_pred):.4f}")
print(f"Precision (macro):       {precision_score(Y_test, y_pred, average='macro'):.4f}")
print(f"Recall (macro):          {recall_score(Y_test, y_pred, average='macro'):.4f}")
print(f"F1-score (macro):        {f1_score(Y_test, y_pred, average='macro'):.4f}")
print(f"F1-score (ponderado):    {f1_score(Y_test, y_pred, average='weighted'):.4f}")
print(f"Log loss:                {log_loss(Y_test, y_proba, labels=clases):.4f}")
print(f"ROC AUC (OvR, macro):    {roc_auc_score(Y_test, y_proba, multi_class='ovr', average='macro'):.4f}")

print("\n=== REPORTE DE CLASIFICACION POR CLASE ===")
print(classification_report(Y_test, y_pred, labels=clases, target_names=etiquetas, digits=4))

print("=== MATRIZ DE CONFUSION ===")
print("(filas = clase real, columnas = clase predicha)")
matriz = confusion_matrix(Y_test, y_pred, labels=clases)
print(pd.DataFrame(matriz,
                   index=[f"Real {e}" for e in etiquetas],
                   columns=[f"Pred {e}" for e in etiquetas]))

# mlogloss al final del entrenamiento (sirve para detectar sobreajuste)
resultados = modelo.evals_result()
mlogloss_train = resultados["validation_0"]["mlogloss"][-1]
mlogloss_test = resultados["validation_1"]["mlogloss"][-1]
print("\n=== MLOGLOSS FINAL ===")
print(f"Entrenamiento: {mlogloss_train:.4f}")
print(f"Prueba:        {mlogloss_test:.4f}")


y_test_bin = label_binarize(Y_test, classes=clases)

plt.figure(figsize=(8, 6))

fpr_por_clase, tpr_por_clase = {}, {}
for i, etiqueta in enumerate(etiquetas):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    fpr_por_clase[i], tpr_por_clase[i] = fpr, tpr
    plt.plot(fpr, tpr, lw=2, color=plt.cm.tab10(i),
             label=f"Riesgo {etiqueta} (AUC = {auc(fpr, tpr):.3f})")

# Promedio macro: interpola las curvas de cada clase sobre una misma malla
fpr_comun = np.unique(np.concatenate(list(fpr_por_clase.values())))
tpr_promedio = np.zeros_like(fpr_comun)
for i in range(len(clases)):
    tpr_promedio += np.interp(fpr_comun, fpr_por_clase[i], tpr_por_clase[i])
tpr_promedio /= len(clases)
plt.plot(fpr_comun, tpr_promedio, lw=2, linestyle=":", color="black",
         label=f"Promedio macro (AUC = {auc(fpr_comun, tpr_promedio):.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", lw=1, color="gray",
         label="Clasificador aleatorio (AUC = 0.500)")

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Tasa de falsos positivos")
plt.ylabel("Tasa de verdaderos positivos")
plt.title("Curva ROC multiclase - XGBoost - Riesgo de diabetes")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()

os.makedirs("resultados", exist_ok=True)
plt.savefig("resultados/curva_roc_xgboost.png", dpi=300)
print("\nCurva ROC guardada en resultados/curva_roc_xgboost.png")
plt.show()