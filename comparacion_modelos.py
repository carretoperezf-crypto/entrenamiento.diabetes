import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron

#RED NEURONAL
from sklearn.neural_network import MLPClassifier

#METRICAS 
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from xgboost import XGBClassifier

archivo= "data/Diabetes_Mexico_DATASET_SMOTE.xlsx"
df=pd.read_excel(archivo)

print("\n === DISTRIBUCION DE CLASES ====")
print(df["riesgo_diabetes_cat"].value_counts())

X = df.drop("riesgo_diabetes_cat", axis=1)
Y = df["riesgo_diabetes_cat"]

print("\n=== DATOS SEPARADOS ===")
print("X:", X.shape)
print("Y:", Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42,
    stratify=Y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n=== DIVISIÓN TRAIN / TEST ===")

print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)

print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)


print("\n=== DISTRIBUCIÓN TRAIN ===")
print(Y_train.value_counts())
print("\n=== DISTRIBUCIÓN TEST ===")
print(Y_test.value_counts())
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#==========================================
#PERCEPTRON
#==========================================
modelo_perceptron = Perceptron(
    max_iter=2000,
    eta0=0.01,
    random_state=42
)

#==========================================
#ARBOL DE DECISION
#==========================================
modelo_arbol = DecisionTreeClassifier(
    max_depth=5,
    random_state=42,
)


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
# RANDOM FOREST
# ==========================================

modelo_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

# ==========================================
# RED NEURONAL 
# ==========================================
modelo_red_neuronal= MLPClassifier(
    hidden_layer_sizes=(64,32),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.15,
    alpha=0.0001,
    n_iter_no_change=30
)

# ==========================================
# ENTRENAMIENTO
# ==========================================
print("\n=== ENTRENANDO PERCEPTRON ====")
modelo_perceptron.fit(X_train_scaled,Y_train)
print("Perceptron termiando")

print("\n=== ENTRENANDO ARBOL DE DECISION ====")
modelo_arbol.fit(X_train,Y_train)
print("Arbol de decision termiando")

print("\n=== ENTRENANDO XGBOOST ===")
modelo_xgb.fit( X_train, Y_train)
print("XGBoost terminado")

print("\n=== ENTRENANDO RANDOM FOREST ===")
modelo_rf.fit(X_train,Y_train)
print("Random Forest terminado")

print("\n=== ENTRENANDO RED NEURONAL ===")
modelo_red_neuronal.fit(X_train_scaled,Y_train)
print("Red Nueronal terminado")

# ==========================================
# PREDICCIONES
# ==========================================

pred_xgb = modelo_xgb.predict(X_test)
pred_rf = modelo_rf.predict(X_test)
pred_perceptron = modelo_perceptron.predict(X_test_scaled)
pred_arbol = modelo_arbol.predict(X_test)
pred_red_neuronal = modelo_red_neuronal.predict(X_test_scaled)



# ==========================================
# FUNCIÓN DE EVALUACIÓN
# ==========================================

def evaluar_modelo(nombre, y_real, y_pred):

    print("\n======================================")
    print(nombre)
    print("======================================")

    accuracy = accuracy_score(
        y_real,
        y_pred
    )

    precision = precision_score(
        y_real,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_real,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_real,
        y_pred,
        average="macro",
        zero_division=0
    )

    print("Accuracy:", round(accuracy, 4))
    print("Precision macro:", round(precision, 4))
    print("Recall macro:", round(recall, 4))
    print("F1 macro:", round(f1, 4))

    print("\n=== REPORTE DE CLASIFICACIÓN ===")

    print(
        classification_report(
            y_real,
            y_pred,
            zero_division=0
        )
    )


evaluar_modelo( "XGBOOST", Y_test, pred_xgb)
evaluar_modelo("RANDOM FOREST",Y_test,pred_rf)
evaluar_modelo("PERCEPTRON",Y_test, pred_perceptron)
evaluar_modelo("ARBOL DE DECISION", Y_test, pred_arbol)
evaluar_modelo("RED NEURONAL - RELU + ADAM",Y_test, pred_red_neuronal)

resultados = []
modelos = [
    ("Perceptrón", pred_perceptron),
    ("Árbol de Decisión", pred_arbol),
    ("Random Forest", pred_rf),
    ("XGBoost", pred_xgb),
    ("Red Neuronal", pred_red_neuronal)
]

for nombre, predicciones in modelos:

    resultados.append({

        "Modelo": nombre,

        "Accuracy": accuracy_score(
            Y_test,
            predicciones
        ),

        "Precision": precision_score(
            Y_test,
            predicciones,
            average="macro",
            zero_division=0
        ),

        "Recall": recall_score(
            Y_test,
            predicciones,
            average="macro",
            zero_division=0
        ),

        "F1": f1_score(
            Y_test,
            predicciones,
            average="macro",
            zero_division=0
        )
    })


df_resultados = pd.DataFrame(
    resultados
)

df_resultados = df_resultados.sort_values(
    by="F1",
    ascending=False
)
print("COMPARACIÓN FINAL DE MODELOS")
print(
    df_resultados.to_string(
        index=False
    )
)

df_predicciones = pd.DataFrame({
    "Real": Y_test,
    "Perceptron": pred_perceptron,
    "Árbol de Decisión": pred_arbol,
    "Random Forest": pred_rf,
    "XGBoost": pred_xgb,
    "Red Neuronal": pred_red_neuronal
})

df_predicciones.to_csv(
    "data/comparacion_modelos_predicciones.csv",
    index=False
)

print("\n==========================================")
print("COMPARACIÓN TERMINADA")
print("==========================================")

print(
    "Resultados guardados en:"
)

print(
    "data/comparacion_modelos_predicciones.csv"
)
