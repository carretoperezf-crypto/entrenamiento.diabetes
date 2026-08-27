# ============================================================
# PROYECTO DIABETES MÉXICO
# MLP - MULTILAYER PERCEPTRON
# SIN CIUDAD
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PASO 1 - CARGAR DATASET
# ============================================================

print("=" * 50)
print("PASO 1 - CARGANDO DATASET")
print("=" * 50)

archivo = "Diabetes_Mexico_IMPUTADO_SIN_CIUDAD.xlsx"

df = pd.read_excel(archivo)

print("\n==========================================")
print("DATASET CARGADO")
print("==========================================")

print(f"Archivo: {archivo}")
print(f"Filas: {df.shape[0]}")
print(f"Columnas: {df.shape[1]}")


# ============================================================
# PASO 2 - VARIABLES PREDICTORAS
# ============================================================

print("\n==========================================")
print("PASO 2 - VARIABLES PREDICTORAS")
print("==========================================")

variables_predictoras = [
    "sexo",
    "edad",
    "Peso",
    "Estatura (cm)",
    "IMC",
    "ac_urico",
    "albu",
    "col_hdl",
    "col_ldl",
    "colest",
    "creat",
    "glu_suero",
    "insulina",
    "trig",
    "hb1ac"
]

for variable in variables_predictoras:
    print(f"- {variable}")

print(f"\nNúmero de variables predictoras: {len(variables_predictoras)}")


# ============================================================
# PASO 3 - VARIABLE OBJETIVO
# ============================================================

print("\n==========================================")
print("PASO 3 - VARIABLE OBJETIVO")
print("==========================================")

variable_objetivo = "riesgo_diabetes_cat"

print(f"Variable objetivo: {variable_objetivo}")


# ============================================================
# PASO 4 - CREAR X E Y
# ============================================================

print("\n==========================================")
print("PASO 4 - CREANDO X E Y")
print("==========================================")

X = df[variables_predictoras].copy()
y = df[variable_objetivo].copy()

print(f"Dimensiones de X: {X.shape}")
print(f"Dimensiones de y: {y.shape}")


# ============================================================
# COMPROBACIÓN DE DATOS
# ============================================================

print("\n==========================================")
print("COMPROBACIÓN DE DATOS")
print("==========================================")

print("\nValores faltantes en X:")
print(X.isnull().sum())

print("\nValores faltantes en y:")
print(y.isnull().sum())

print("\nTotal de faltantes en X:")
print(X.isnull().sum().sum())

print("\nTotal de faltantes en y:")
print(y.isnull().sum())


# ============================================================
# PASO 5 - DISTRIBUCIÓN DE LA VARIABLE OBJETIVO
# ============================================================

print("\n==========================================")
print("PASO 5 - DISTRIBUCIÓN DE CLASES")
print("==========================================")

distribucion = y.value_counts().sort_index()

print(distribucion)

print("\nPorcentajes:")

porcentajes = y.value_counts(
    normalize=True
).sort_index() * 100

print(porcentajes.round(2))

print("\nClases:")
print(sorted(y.unique()))


# ============================================================
# PASO 6 - DIVISIÓN TRAIN / TEST
# ============================================================

print("\n==========================================")
print("PASO 6 - DIVISIÓN TRAIN / TEST")
print("==========================================")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nEntrenamiento: 70 %")
print("Prueba: 30 %")
print("Random state: 42")
print("Estratificación: activada")

print("\n==========================================")
print("RESULTADOS DE LA DIVISIÓN")
print("==========================================")

print(f"X_train: {X_train.shape}")
print(f"X_test : {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test : {y_test.shape}")


# ============================================================
# DISTRIBUCIÓN TRAIN
# ============================================================

print("\n==========================================")
print("DISTRIBUCIÓN EN TRAIN")
print("==========================================")

print(y_train.value_counts().sort_index())

print("\nPorcentajes TRAIN:")

print(
    (y_train.value_counts(normalize=True).sort_index() * 100)
    .round(2)
)


# ============================================================
# DISTRIBUCIÓN TEST
# ============================================================

print("\n==========================================")
print("DISTRIBUCIÓN EN TEST")
print("==========================================")

print(y_test.value_counts().sort_index())

print("\nPorcentajes TEST:")

print(
    (y_test.value_counts(normalize=True).sort_index() * 100)
    .round(2)
)


# ============================================================
# COMPROBACIÓN DE LA DIVISIÓN
# ============================================================

print("\n==========================================")
print("COMPROBACIÓN FINAL")
print("==========================================")

print(f"Registros originales: {len(df)}")
print(f"Registros TRAIN: {len(X_train)}")
print(f"Registros TEST: {len(X_test)}")
print(f"TRAIN + TEST: {len(X_train) + len(X_test)}")

if len(X_train) + len(X_test) == len(df):
    print("✓ La división conserva todos los registros.")
else:
    print("✗ ERROR EN LA DIVISIÓN")


# ============================================================
# PASO 7 - ESTANDARIZACIÓN
# ============================================================

print("\n==========================================")
print("PASO 7 - ESTANDARIZACIÓN")
print("==========================================")

print("\nCreando StandardScaler...")

scaler = StandardScaler()

print("StandardScaler creado correctamente.")

print("\nAjustando escalador únicamente con X_train...")

scaler.fit(X_train)

print("Escalador ajustado correctamente.")


# ============================================================
# ESTANDARIZAR TRAIN
# ============================================================

X_train_scaled = scaler.transform(X_train)

print("\nX_train estandarizado correctamente.")


# ============================================================
# ESTANDARIZAR TEST
# ============================================================

X_test_scaled = scaler.transform(X_test)

print("X_test estandarizado correctamente.")


# ============================================================
# COMPROBACIÓN DE ESTANDARIZACIÓN
# ============================================================

print("\n==========================================")
print("COMPROBACIÓN DE ESTANDARIZACIÓN")
print("==========================================")

print("\nDimensiones X_train_scaled:")
print(X_train_scaled.shape)

print("\nDimensiones X_test_scaled:")
print(X_test_scaled.shape)

print("\nMedia X_train:")

print(
    np.round(
        X_train_scaled.mean(axis=0),
        4
    )
)

print("\nDesviación estándar X_train:")

print(
    np.round(
        X_train_scaled.std(axis=0),
        4
    )
)


# ============================================================
# COMPROBAR NaN
# ============================================================

print("\n==========================================")
print("COMPROBACIÓN DE NaN")
print("==========================================")

print(
    "NaN X_train:",
    np.isnan(X_train_scaled).sum()
)

print(
    "NaN X_test:",
    np.isnan(X_test_scaled).sum()
)


# ============================================================
# COMPROBAR INFINITOS
# ============================================================

print("\n==========================================")
print("COMPROBACIÓN DE INFINITOS")
print("==========================================")

print(
    "Infinitos X_train:",
    np.isinf(X_train_scaled).sum()
)

print(
    "Infinitos X_test:",
    np.isinf(X_test_scaled).sum()
)


# ============================================================
# PASO 8 - CREAR MLP
# ============================================================

print("\n==========================================")
print("PASO 8 - CREACIÓN DEL MLP")
print("==========================================")

mlp = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    batch_size=32,
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42
)

print("MLP creado correctamente.")


# ============================================================
# CONFIGURACIÓN
# ============================================================

print("\n==========================================")
print("CONFIGURACIÓN DEL MLP")
print("==========================================")

print("Tipo de modelo: MLPClassifier")
print("Capas ocultas: (32, 16)")
print("Activación: ReLU")
print("Optimizador: Adam")
print("Alpha: 0.0001")
print("Batch size: 32")
print("Learning rate inicial: 0.001")
print("Máximo de iteraciones: 500")
print("Random state: 42")


# ============================================================
# PASO 9 - ENTRENAMIENTO
# ============================================================

print("\n==========================================")
print("PASO 9 - ENTRENAMIENTO DEL MLP")
print("==========================================")

print("\nEntrenando MLP...")

mlp.fit(
    X_train_scaled,
    y_train
)

print("\nMLP entrenado correctamente.")


# ============================================================
# INFORMACIÓN DEL MODELO
# ============================================================

print("\n==========================================")
print("INFORMACIÓN DEL MODELO")
print("==========================================")

print(
    "Número de clases:",
    len(mlp.classes_)
)

print(
    "Clases aprendidas:",
    mlp.classes_
)

print(
    "Número de iteraciones realizadas:",
    mlp.n_iter_
)

print(
    "Número de muestras utilizadas:",
    mlp.t_
)

print(
    "Número de capas:",
    len(mlp.coefs_)
)


# ============================================================
# PASO 10 - PREDICCIONES
# ============================================================

print("\n==========================================")
print("PASO 10 - REALIZANDO PREDICCIONES")
print("==========================================")

y_pred = mlp.predict(X_test_scaled)

print("Predicciones realizadas correctamente.")

print(
    "Número de predicciones:",
    len(y_pred)
)


# ============================================================
# DISTRIBUCIÓN DE PREDICCIONES
# ============================================================

print("\n==========================================")
print("DISTRIBUCIÓN DE PREDICCIONES")
print("==========================================")

predicciones = pd.Series(
    y_pred,
    name="Prediccion"
)

print(
    predicciones.value_counts().sort_index()
)


# ============================================================
# REAL VS PREDICCIÓN
# ============================================================

print("\n==========================================")
print("REAL VS PREDICCIÓN")
print("==========================================")

comparacion = pd.DataFrame({
    "Real": y_test.values,
    "Prediccion": y_pred
})

print(
    comparacion.head(20)
)


# ============================================================
# COMPROBACIÓN DE PREDICCIONES
# ============================================================

print("\n==========================================")
print("COMPROBACIÓN DE PREDICCIONES")
print("==========================================")

correctas = np.sum(
    y_test.values == y_pred
)

incorrectas = np.sum(
    y_test.values != y_pred
)

print(
    "Predicciones correctas:",
    correctas
)

print(
    "Predicciones incorrectas:",
    incorrectas
)


# ============================================================
# PASO 11 - MATRIZ DE CONFUSIÓN
# ============================================================

print("\n==========================================")
print("PASO 11 - MATRIZ DE CONFUSIÓN")
print("==========================================")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1, 2]
)

print("\nFilas = valores reales")
print("Columnas = valores predichos\n")

print(cm)


# ============================================================
# PASO 12 - CÁLCULO DE MÉTRICAS
# ============================================================

print("\n==========================================")
print("PASO 12 - MÉTRICAS DEL MLP BASE")
print("==========================================")


# ------------------------------------------------------------
# ACCURACY
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)


# ------------------------------------------------------------
# PRECISION PONDERADA
# ------------------------------------------------------------

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


# ------------------------------------------------------------
# RECALL PONDERADO
# ------------------------------------------------------------

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


# ------------------------------------------------------------
# F1 SCORE PONDERADO
# ------------------------------------------------------------

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


# ------------------------------------------------------------
# BALANCED ACCURACY
# ------------------------------------------------------------

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)


# ------------------------------------------------------------
# F1 POR CLASE
# ------------------------------------------------------------

f1_clases = f1_score(
    y_test,
    y_pred,
    labels=[0, 1, 2],
    average=None,
    zero_division=0
)


# ------------------------------------------------------------
# RECALL POR CLASE
# ------------------------------------------------------------

recall_clases = recall_score(
    y_test,
    y_pred,
    labels=[0, 1, 2],
    average=None,
    zero_division=0
)


# ============================================================
# RESULTADOS
# ============================================================

print("\n==========================================")
print("RESULTADOS")
print("==========================================")

print(
    f"Accuracy:             {accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Precision ponderada:  {precision:.4f} "
    f"({precision * 100:.2f}%)"
)

print(
    f"Recall ponderado:     {recall:.4f} "
    f"({recall * 100:.2f}%)"
)

print(
    f"F1-score ponderado:   {f1:.4f} "
    f"({f1 * 100:.2f}%)"
)

print(
    f"Balanced Accuracy:    {balanced_accuracy:.4f} "
    f"({balanced_accuracy * 100:.2f}%)"
)


# ============================================================
# MÉTRICAS POR CLASE
# ============================================================

print("\n==========================================")
print("MÉTRICAS POR CLASE")
print("==========================================")

print(
    f"F1 Clase 0:            {f1_clases[0]:.4f} "
    f"({f1_clases[0] * 100:.2f}%)"
)

print(
    f"F1 Clase 1:            {f1_clases[1]:.4f} "
    f"({f1_clases[1] * 100:.2f}%)"
)

print(
    f"F1 Clase 2:            {f1_clases[2]:.4f} "
    f"({f1_clases[2] * 100:.2f}%)"
)

print(
    f"Recall Clase 0:        {recall_clases[0]:.4f} "
    f"({recall_clases[0] * 100:.2f}%)"
)

print(
    f"Recall Clase 1:        {recall_clases[1]:.4f} "
    f"({recall_clases[1] * 100:.2f}%)"
)

print(
    f"Recall Clase 2:        {recall_clases[2]:.4f} "
    f"({recall_clases[2] * 100:.2f}%)"
)


# ============================================================
# TABLA FINAL
# ============================================================

print("\n==========================================")
print("TABLA FINAL DE MÉTRICAS")
print("==========================================")

tabla_metricas = pd.DataFrame({
    "Métrica": [
        "Accuracy",
        "Precision ponderada",
        "Recall ponderado",
        "F1-score ponderado",
        "Balanced Accuracy",
        "F1 Clase 0",
        "F1 Clase 1",
        "F1 Clase 2",
        "Recall Clase 0",
        "Recall Clase 1",
        "Recall Clase 2"
    ],

    "MLP Base": [
        accuracy,
        precision,
        recall,
        f1,
        balanced_accuracy,
        f1_clases[0],
        f1_clases[1],
        f1_clases[2],
        recall_clases[0],
        recall_clases[1],
        recall_clases[2]
    ]
})


print(
    tabla_metricas.to_string(index=False)
)


# ============================================================
# TABLA EN PORCENTAJES
# ============================================================

print("\n==========================================")
print("TABLA FINAL EN PORCENTAJES")
print("==========================================")

tabla_porcentajes = tabla_metricas.copy()

tabla_porcentajes["MLP Base"] = (
    tabla_porcentajes["MLP Base"] * 100
).round(2)

tabla_porcentajes["MLP Base"] = (
    tabla_porcentajes["MLP Base"].astype(str) + "%"
)

print(
    tabla_porcentajes.to_string(index=False)
)


# ============================================================
# REPORTE DE CLASIFICACIÓN
# ============================================================

print("\n==========================================")
print("REPORTE DE CLASIFICACIÓN")
print("==========================================")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        zero_division=0
    )
)


# ============================================================
# RESUMEN FINAL
# ============================================================

print("\n==========================================")
print("RESUMEN FINAL DEL MLP")
print("==========================================")

print("Dataset:")
print(archivo)

print("\nRegistros totales:")
print(len(df))

print("\nVariables predictoras:")
print(len(variables_predictoras))

print("\nVariables:")
for variable in variables_predictoras:
    print(f"- {variable}")

print("\nVariable objetivo:")
print(variable_objetivo)

print("\nTrain:")
print(X_train.shape)

print("\nTest:")
print(X_test.shape)

print("\nArquitectura:")
print("(32, 16)")

print("\nClases:")
print(mlp.classes_)

print("\nPredicciones:")
print(len(y_pred))

print("\nPredicciones correctas:")
print(correctas)

print("\nPredicciones incorrectas:")
print(incorrectas)

print("\nAccuracy final:")
print(f"{accuracy * 100:.2f}%")

print("\nBalanced Accuracy final:")
print(f"{balanced_accuracy * 100:.2f}%")


# ============================================================
# FIN
# ============================================================

print("\n==========================================")
print("PROCESO COMPLETADO")
print("==========================================")

print("✓ Dataset cargado")
print("✓ Variables seleccionadas")
print("✓ Train/Test realizado")
print("✓ Estandarización realizada")
print("✓ MLP creado")
print("✓ MLP entrenado")
print("✓ Predicciones realizadas")
print("✓ Matriz de confusión calculada")
print("✓ Métricas calculadas")
print("✓ Tabla final generada")

print("\nMLP BASE TERMINADO.")