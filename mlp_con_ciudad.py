# ============================================================
# MLP BASE CON CIUDAD
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
    classification_report
)

# ============================================================
# PASO 1 - CARGAR DATASET
# ============================================================

print("=" * 50)
print("PASO 1 - PREPARACIÓN MLP CON CIUDAD")
print("=" * 50)

archivo = "Diabetes_Mexico_IMPUTADO.xlsx"

df = pd.read_excel(archivo)

print("\n==========================================")
print("DATASET CARGADO")
print("==========================================")

print("Archivo:", archivo)
print("Filas:", df.shape[0])
print("Columnas:", df.shape[1])


# ============================================================
# PASO 2 - VARIABLES
# ============================================================

print("\n==========================================")
print("PASO 2 - VARIABLES PREDICTORAS")
print("==========================================")

variables_numericas = [
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

variable_categorica = "Ciudad"

variable_objetivo = "riesgo_diabetes_cat"

for variable in variables_numericas:
    print("-", variable)

print("-", variable_categorica)

print("\n==========================================")
print("VARIABLE OBJETIVO")
print("==========================================")

print(variable_objetivo)


# ============================================================
# PASO 3 - CODIFICAR CIUDAD
# ============================================================

print("\n==========================================")
print("PASO 3 - CODIFICACIÓN DE CIUDAD")
print("==========================================")

# Comprobar que exista Ciudad

if variable_categorica not in df.columns:

    print("ERROR: No existe la columna Ciudad.")
    exit()

# Convertir Ciudad a texto

df["Ciudad"] = df["Ciudad"].astype(str)

# Obtener categorías ordenadas

categorias_ciudad = sorted(df["Ciudad"].unique())

# Crear diccionario

codigo_ciudad = {
    ciudad: i + 1
    for i, ciudad in enumerate(categorias_ciudad)
}

# Crear nueva columna temporal

df["Ciudad_Codigo"] = df["Ciudad"].map(codigo_ciudad)

print("Número de ciudades:", len(categorias_ciudad))

print("\nPrimeras 20 ciudades:")

for ciudad, codigo in list(codigo_ciudad.items())[:20]:

    print(f"{codigo} -> {ciudad}")


# ============================================================
# COMPROBACIÓN DE CODIFICACIÓN
# ============================================================

print("\n==========================================")
print("COMPROBACIÓN DE CODIFICACIÓN")
print("==========================================")

tabla_ciudades = pd.DataFrame({
    "Ciudad": categorias_ciudad,
    "Codigo": range(1, len(categorias_ciudad) + 1)
})

print(tabla_ciudades.head(20))


# ============================================================
# PASO 4 - CREAR X E Y
# ============================================================

print("\n==========================================")
print("PASO 4 - CREANDO X E Y")
print("==========================================")

# Variables clínicas

X_numericas = df[variables_numericas].copy()

# Ciudad codificada

X_ciudad = df[["Ciudad_Codigo"]].copy()

# Variable objetivo

y = df[variable_objetivo].copy()

print("Variables numéricas:", X_numericas.shape)
print("Ciudad:", X_ciudad.shape)
print("Variable objetivo:", y.shape)


# ============================================================
# COMPROBACIÓN DE VALORES FALTANTES
# ============================================================

print("\n==========================================")
print("COMPROBACIÓN DE VALORES FALTANTES")
print("==========================================")

print("Faltantes variables numéricas:")
print(X_numericas.isnull().sum())

print("\nFaltantes Ciudad:")
print(X_ciudad.isnull().sum())

print("\nFaltantes variable objetivo:")
print(y.isnull().sum())


# ============================================================
# PASO 5 - DIVISIÓN TRAIN / TEST
# ============================================================

print("\n==========================================")
print("PASO 5 - DIVISIÓN TRAIN / TEST")
print("==========================================")

X_train_num, X_test_num, y_train, y_test = train_test_split(
    X_numericas,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Obtener Ciudad usando los mismos índices

X_train_ciudad = X_ciudad.loc[X_train_num.index]

X_test_ciudad = X_ciudad.loc[X_test_num.index]

print("X_train numérico:", X_train_num.shape)
print("X_test numérico :", X_test_num.shape)

print("Ciudad TRAIN:", X_train_ciudad.shape)
print("Ciudad TEST :", X_test_ciudad.shape)

print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ============================================================
# DISTRIBUCIÓN DE CLASES
# ============================================================

print("\n==========================================")
print("DISTRIBUCIÓN DE CLASES")
print("==========================================")

print("\nTRAIN:")
print(y_train.value_counts().sort_index())

print("\nTEST:")
print(y_test.value_counts().sort_index())


# ============================================================
# PASO 6 - ESTANDARIZACIÓN
# ============================================================

print("\n==========================================")
print("PASO 6 - ESTANDARIZACIÓN")
print("==========================================")

scaler = StandardScaler()

# Ajustar SOLO con TRAIN

X_train_num_scaled = scaler.fit_transform(X_train_num)

# Transformar TEST

X_test_num_scaled = scaler.transform(X_test_num)

print("Variables clínicas estandarizadas correctamente.")


# ============================================================
# PASO 7 - AGREGAR CIUDAD
# ============================================================

print("\n==========================================")
print("PASO 7 - AGREGANDO CIUDAD")
print("==========================================")

# Convertir a DataFrame

X_train_num_scaled = pd.DataFrame(
    X_train_num_scaled,
    index=X_train_num.index,
    columns=variables_numericas
)

X_test_num_scaled = pd.DataFrame(
    X_test_num_scaled,
    index=X_test_num.index,
    columns=variables_numericas
)

# Ciudad permanece SIN estandarizar

X_train_final = pd.concat(
    [
        X_train_num_scaled,
        X_train_ciudad
    ],
    axis=1
)

X_test_final = pd.concat(
    [
        X_test_num_scaled,
        X_test_ciudad
    ],
    axis=1
)

print("X_train final:", X_train_final.shape)
print("X_test final :", X_test_final.shape)

print("\nCiudad permanece con códigos 1-267.")

print("\nMínimo Ciudad:")
print(X_train_final["Ciudad_Codigo"].min())

print("\nMáximo Ciudad:")
print(X_train_final["Ciudad_Codigo"].max())


# ============================================================
# COMPROBACIÓN DE DATOS
# ============================================================

print("\n==========================================")
print("COMPROBACIÓN DE DATOS")
print("==========================================")

print("Valores faltantes X_train:",
      X_train_final.isnull().sum().sum())

print("Valores faltantes X_test:",
      X_test_final.isnull().sum().sum())

print("Valores faltantes y_train:",
      y_train.isnull().sum())

print("Valores faltantes y_test:",
      y_test.isnull().sum())


# ============================================================
# PASO 8 - CREAR MLP
# ============================================================

print("\n==========================================")
print("PASO 8 - CREANDO MLP")
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
print("PASO 9 - ENTRENANDO MLP")
print("==========================================")

mlp.fit(
    X_train_final,
    y_train
)

print("MLP entrenado correctamente.")


# ============================================================
# INFORMACIÓN DEL MODELO
# ============================================================

print("\n==========================================")
print("INFORMACIÓN DEL MODELO")
print("==========================================")

print("Número de clases:", len(mlp.classes_))

print("Clases aprendidas:")
print(mlp.classes_)

print("Número de iteraciones realizadas:")
print(mlp.n_iter_)

print("Número de muestras utilizadas:")
print(mlp.t_)

print("Número de capas:")
print(mlp.n_layers_)


# ============================================================
# PASO 10 - PREDICCIONES
# ============================================================

print("\n==========================================")
print("PASO 10 - REALIZANDO PREDICCIONES")
print("==========================================")

y_pred = mlp.predict(X_test_final)

print("Predicciones realizadas correctamente.")

print("Número de predicciones:")
print(len(y_pred))


# ============================================================
# DISTRIBUCIÓN DE PREDICCIONES
# ============================================================

print("\n==========================================")
print("DISTRIBUCIÓN DE PREDICCIONES")
print("==========================================")

predicciones_df = pd.Series(
    y_pred,
    name="Prediccion"
)

print(
    predicciones_df
    .value_counts()
    .sort_index()
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

print(comparacion.head(20))


# ============================================================
# COMPROBAR PREDICCIONES
# ============================================================

correctas = (y_test.values == y_pred).sum()

incorrectas = (y_test.values != y_pred).sum()

print("\n==========================================")
print("COMPROBACIÓN DE PREDICCIONES")
print("==========================================")

print("Predicciones correctas:", correctas)

print("Predicciones incorrectas:", incorrectas)


# ============================================================
# PASO 11 - MÉTRICAS
# ============================================================

print("\n==========================================")
print("PASO 11 - MÉTRICAS DEL MLP CON CIUDAD")
print("==========================================")


# Accuracy

accuracy = accuracy_score(
    y_test,
    y_pred
)


# Precision ponderada

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


# Recall ponderado

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


# F1 ponderado

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


# Balanced Accuracy

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)


# F1 por clase

f1_clase = f1_score(
    y_test,
    y_pred,
    average=None,
    labels=[0, 1, 2],
    zero_division=0
)


# Recall por clase

recall_clase = recall_score(
    y_test,
    y_pred,
    average=None,
    labels=[0, 1, 2],
    zero_division=0
)


# ============================================================
# MOSTRAR RESULTADOS
# ============================================================

print("\n==========================================")
print("RESULTADOS")
print("==========================================")

print(
    f"Accuracy:             {accuracy:.4f} "
    f"({accuracy*100:.2f}%)"
)

print(
    f"Precision ponderada:  {precision:.4f} "
    f"({precision*100:.2f}%)"
)

print(
    f"Recall ponderado:     {recall:.4f} "
    f"({recall*100:.2f}%)"
)

print(
    f"F1-score ponderado:   {f1:.4f} "
    f"({f1*100:.2f}%)"
)

print(
    f"Balanced Accuracy:    {balanced_accuracy:.4f} "
    f"({balanced_accuracy*100:.2f}%)"
)


# ============================================================
# MÉTRICAS POR CLASE
# ============================================================

print("\n==========================================")
print("MÉTRICAS POR CLASE")
print("==========================================")

print(
    f"F1 Clase 0:            {f1_clase[0]:.4f} "
    f"({f1_clase[0]*100:.2f}%)"
)

print(
    f"F1 Clase 1:            {f1_clase[1]:.4f} "
    f"({f1_clase[1]*100:.2f}%)"
)

print(
    f"F1 Clase 2:            {f1_clase[2]:.4f} "
    f"({f1_clase[2]*100:.2f}%)"
)

print(
    f"Recall Clase 0:        {recall_clase[0]:.4f} "
    f"({recall_clase[0]*100:.2f}%)"
)

print(
    f"Recall Clase 1:        {recall_clase[1]:.4f} "
    f"({recall_clase[1]*100:.2f}%)"
)

print(
    f"Recall Clase 2:        {recall_clase[2]:.4f} "
    f"({recall_clase[2]*100:.2f}%)"
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

    "MLP Con Ciudad": [

        accuracy,

        precision,

        recall,

        f1,

        balanced_accuracy,

        f1_clase[0],

        f1_clase[1],

        f1_clase[2],

        recall_clase[0],

        recall_clase[1],

        recall_clase[2]

    ]

})


print(tabla_metricas.to_string(index=False))


# ============================================================
# TABLA EN PORCENTAJES
# ============================================================

print("\n==========================================")
print("TABLA FINAL EN PORCENTAJES")
print("==========================================")

tabla_porcentajes = tabla_metricas.copy()

tabla_porcentajes["MLP Con Ciudad"] = (
    tabla_porcentajes["MLP Con Ciudad"] * 100
).round(2).astype(str) + "%"

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
print("RESUMEN FINAL DEL MLP CON CIUDAD")
print("==========================================")

print("Dataset:", archivo)

print("Registros:", df.shape[0])

print("Variables clínicas:", len(variables_numericas))

print("Variable categórica: Ciudad")

print("Categorías de Ciudad:", len(categorias_ciudad))

print("Variables finales:", X_train_final.shape[1])

print("Train:", X_train_final.shape)

print("Test:", X_test_final.shape)

print("Arquitectura MLP: (32, 16)")

print(f"Accuracy: {accuracy*100:.2f}%")

print(f"F1 ponderado: {f1*100:.2f}%")

print(f"Balanced Accuracy: {balanced_accuracy*100:.2f}%")

print(f"Predicciones correctas: {correctas}")

print(f"Predicciones incorrectas: {incorrectas}")

print("\n==========================================")
print("PROCESO TERMINADO")
print("==========================================")

print("✓ Ciudad incluida.")

print("✓ Ciudad codificada de 1 a 267.")

print("✓ Variables clínicas estandarizadas.")

print("✓ Ciudad NO estandarizada.")

print("✓ Train/Test preparados.")

print("✓ MLP entrenado.")

print("✓ Predicciones realizadas.")

print("✓ Métricas calculadas.")

print("✓ Tabla final generada.")