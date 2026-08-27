import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

# ============================================================
# PASO 5 - ENTRENAMIENTO DEL PERCEPTRÓN
# ============================================================

print("=" * 50)
print("PASO 5 - ENTRENAMIENTO DEL PERCEPTRÓN")
print("=" * 50)


# ============================================================
# 1. CARGAR DATASET
# ============================================================

archivo = "Diabetes_Mexico_IMPUTADO.xlsx"

df = pd.read_excel(archivo)

print("\n==========================================")
print("DATASET CARGADO")
print("==========================================")

print("Archivo:", archivo)
print("Filas:", df.shape[0])
print("Columnas:", df.shape[1])


# ============================================================
# 2. VARIABLES
# ============================================================

variable_categorica = "Ciudad"
variable_objetivo = "riesgo_diabetes_cat"

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


# ============================================================
# 3. CODIFICAR CIUDAD
# ============================================================

print("\n==========================================")
print("CODIFICANDO CIUDAD")
print("==========================================")

categorias_ciudad = sorted(
    df[variable_categorica].astype(str).unique()
)

mapa_ciudades = {
    ciudad: codigo
    for codigo, ciudad in enumerate(categorias_ciudad, start=1)
}

df["Ciudad_Codigo"] = (
    df[variable_categorica]
    .astype(str)
    .map(mapa_ciudades)
)

print("Número de ciudades:", len(categorias_ciudad))

print("\nEjemplo:")

for codigo, ciudad in list(
    enumerate(categorias_ciudad, start=1)
)[:10]:

    print(codigo, "->", ciudad)


# ============================================================
# 4. CREAR X E Y
# ============================================================

print("\n==========================================")
print("CREANDO X E Y")
print("==========================================")

X_numerico = df[variables_numericas].copy()

X_ciudad = df[["Ciudad_Codigo"]].copy()

y = df[variable_objetivo].copy()

print("Variables numéricas:", X_numerico.shape)
print("Ciudad:", X_ciudad.shape)
print("Variable objetivo:", y.shape)


# ============================================================
# 5. DIVISIÓN TRAIN / TEST
# ============================================================

print("\n==========================================")
print("DIVISIÓN TRAIN / TEST")
print("==========================================")

X_train_num, X_test_num, X_train_ciudad, X_test_ciudad, y_train, y_test = train_test_split(

    X_numerico,
    X_ciudad,
    y,

    test_size=0.30,

    random_state=42,

    stratify=y
)


print("X_train:", X_train_num.shape)
print("X_test :", X_test_num.shape)

print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ============================================================
# 6. MOSTRAR DISTRIBUCIÓN DE CLASES
# ============================================================

print("\n==========================================")
print("DISTRIBUCIÓN DE CLASES")
print("==========================================")

print("\nTRAIN:")

print(
    y_train.value_counts()
    .sort_index()
)

print("\nTEST:")

print(
    y_test.value_counts()
    .sort_index()
)


# ============================================================
# 7. ESTANDARIZAR VARIABLES NUMÉRICAS
# ============================================================

print("\n==========================================")
print("ESTANDARIZACIÓN")
print("==========================================")

scaler = StandardScaler()

X_train_num_scaled = scaler.fit_transform(
    X_train_num
)

X_test_num_scaled = scaler.transform(
    X_test_num
)

print("Variables numéricas estandarizadas correctamente.")


# ============================================================
# 8. CONVERTIR A DATAFRAME
# ============================================================

X_train_num_scaled = pd.DataFrame(

    X_train_num_scaled,

    columns=variables_numericas,

    index=X_train_num.index
)

X_test_num_scaled = pd.DataFrame(

    X_test_num_scaled,

    columns=variables_numericas,

    index=X_test_num.index
)


# ============================================================
# 9. AGREGAR CIUDAD
# ============================================================

print("\n==========================================")
print("AGREGANDO CIUDAD")
print("==========================================")

X_train = pd.concat(

    [
        X_train_num_scaled,
        X_train_ciudad
    ],

    axis=1
)

X_test = pd.concat(

    [
        X_test_num_scaled,
        X_test_ciudad
    ],

    axis=1
)


print("X_train final:", X_train.shape)
print("X_test final :", X_test.shape)


# ============================================================
# 10. COMPROBAR DATOS
# ============================================================

print("\n==========================================")
print("COMPROBACIÓN DE DATOS")
print("==========================================")

print(
    "Valores faltantes X_train:",
    X_train.isnull().sum().sum()
)

print(
    "Valores faltantes X_test:",
    X_test.isnull().sum().sum()
)

print(
    "Valores faltantes y_train:",
    y_train.isnull().sum()
)

print(
    "Valores faltantes y_test:",
    y_test.isnull().sum()
)


# ============================================================
# 11. CREAR PERCEPTRÓN
# ============================================================

print("\n==========================================")
print("CREANDO PERCEPTRÓN")
print("==========================================")

modelo = Perceptron(

    max_iter=1000,

    tol=0.001,

    random_state=42
)


print("Modelo creado correctamente.")

print("\nConfiguración:")

print("Tipo: Perceptrón")
print("max_iter:", modelo.max_iter)
print("tol:", modelo.tol)
print("random_state:", modelo.random_state)


# ============================================================
# 12. ENTRENAR PERCEPTRÓN
# ============================================================

print("\n==========================================")
print("ENTRENANDO PERCEPTRÓN")
print("==========================================")

modelo.fit(

    X_train,
    y_train
)

print("Perceptrón entrenado correctamente.")


# ============================================================
# 13. CLASES APRENDIDAS
# ============================================================

print("\n==========================================")
print("CLASES APRENDIDAS")
print("==========================================")

print(modelo.classes_)


# ============================================================
# 14. INFORMACIÓN DEL MODELO
# ============================================================

print("\n==========================================")
print("INFORMACIÓN DEL MODELO")
print("==========================================")

print("Número de clases:",
      len(modelo.classes_))

print("Número de variables:",
      X_train.shape[1])

print("Número de iteraciones realizadas:",
      modelo.n_iter_)

print("Número de muestras utilizadas:",
      modelo.t_)


# ============================================================
# 15. PREDICCIONES
# ============================================================

print("\n==========================================")
print("REALIZANDO PREDICCIONES")
print("==========================================")

y_pred = modelo.predict(X_test)

print(
    "Número de predicciones:",
    len(y_pred)
)


# ============================================================
# 16. DISTRIBUCIÓN DE PREDICCIONES
# ============================================================

print("\n==========================================")
print("DISTRIBUCIÓN DE PREDICCIONES")
print("==========================================")

predicciones = pd.Series(
    y_pred,
    name="Prediccion"
)

print(
    predicciones
    .value_counts()
    .sort_index()
)


# ============================================================
# 17. COMPARACIÓN REAL VS PREDICCIÓN
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
# 18. COMPROBACIÓN DE PREDICCIONES
# ============================================================

print("\n==========================================")
print("COMPROBACIÓN DE PREDICCIONES")
print("==========================================")

print(
    "Predicciones correctas:",
    (y_pred == y_test.values).sum()
)

print(
    "Predicciones incorrectas:",
    (y_pred != y_test.values).sum()
)


# ============================================================
# 19. RESUMEN
# ============================================================

print("\n==========================================")
print("RESUMEN DEL PASO 5")
print("==========================================")

print("Dataset:", archivo)

print("Registros:", len(df))

print("Variables numéricas:",
      len(variables_numericas))

print("Variable categórica: Ciudad")

print("Categorías de Ciudad:",
      len(categorias_ciudad))

print("Variables finales:",
      X_train.shape[1])

print("Train:",
      X_train.shape)

print("Test:",
      X_test.shape)

print("Clases:",
      modelo.classes_)

print("Predicciones:",
      len(y_pred))


print("\n==========================================")
print("PASO 5 TERMINADO")
print("==========================================")

print("✓ Ciudad codificada 1-267.")
print("✓ Variables clínicas estandarizadas.")
print("✓ Train/Test preparados.")
print("✓ Perceptrón entrenado.")
print("✓ Predicciones realizadas.")

print("\nTodavía NO se han calculado las métricas finales.")

# ============================================================
# PASO 6 - MÉTRICAS DEL PERCEPTRÓN BASE
# ============================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score
)

print("\n==========================================")
print("PASO 6 - MÉTRICAS DEL PERCEPTRÓN BASE")
print("==========================================")

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Precision ponderada
precision_ponderada = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

# Recall ponderado
recall_ponderado = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

# F1 ponderado
f1_ponderado = f1_score(
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
f1_por_clase = f1_score(
    y_test,
    y_pred,
    average=None,
    labels=[0, 1, 2],
    zero_division=0
)

# Recall por clase
recall_por_clase = recall_score(
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

print(f"Accuracy:             {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision ponderada:  {precision_ponderada:.4f} ({precision_ponderada*100:.2f}%)")
print(f"Recall ponderado:     {recall_ponderado:.4f} ({recall_ponderado*100:.2f}%)")
print(f"F1-score ponderado:   {f1_ponderado:.4f} ({f1_ponderado*100:.2f}%)")
print(f"Balanced Accuracy:    {balanced_accuracy:.4f} ({balanced_accuracy*100:.2f}%)")

print("\n==========================================")
print("MÉTRICAS POR CLASE")
print("==========================================")

print(f"F1 Clase 0:            {f1_por_clase[0]:.4f} ({f1_por_clase[0]*100:.2f}%)")
print(f"F1 Clase 1:            {f1_por_clase[1]:.4f} ({f1_por_clase[1]*100:.2f}%)")
print(f"F1 Clase 2:            {f1_por_clase[2]:.4f} ({f1_por_clase[2]*100:.2f}%)")

print(f"Recall Clase 0:        {recall_por_clase[0]:.4f} ({recall_por_clase[0]*100:.2f}%)")
print(f"Recall Clase 1:        {recall_por_clase[1]:.4f} ({recall_por_clase[1]*100:.2f}%)")
print(f"Recall Clase 2:        {recall_por_clase[2]:.4f} ({recall_por_clase[2]*100:.2f}%)")


# ============================================================
# TABLA FINAL
# ============================================================

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

    "Perceptrón Base": [
        accuracy,
        precision_ponderada,
        recall_ponderado,
        f1_ponderado,
        balanced_accuracy,
        f1_por_clase[0],
        f1_por_clase[1],
        f1_por_clase[2],
        recall_por_clase[0],
        recall_por_clase[1],
        recall_por_clase[2]
    ]
})


print("\n==========================================")
print("TABLA FINAL DE MÉTRICAS")
print("==========================================")

print(
    tabla_metricas.to_string(index=False)
)

print("\n==========================================")
print("PASO 6 TERMINADO")
print("==========================================")