import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from xgboost import XGBClassifier


# ==========================================
# CARGAR DATOS
# ==========================================

X_train = pd.read_csv("data/X_train_smote.csv")
Y_train = pd.read_csv("data/Y_train_smote.csv").squeeze()

X_test = pd.read_csv("data/X_test.csv")
Y_test = pd.read_csv("data/Y_test.csv").squeeze()


print("=== DATOS CARGADOS ===")

print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)

print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)


# ==========================================
# MODELO 1: XGBOOST
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
# MODELO 2: RANDOM FOREST
# ==========================================

modelo_rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)


# ==========================================
# MODELO 3: REGRESION LOGISTICA
# ==========================================

modelo_lr = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)


# ==========================================
# ENTRENAMIENTO
# ==========================================

print("\n=== ENTRENANDO XGBOOST ===")

modelo_xgb.fit(
    X_train,
    Y_train
)

print("XGBoost terminado")


print("\n=== ENTRENANDO RANDOM FOREST ===")

modelo_rf.fit(
    X_train,
    Y_train
)

print("Random Forest terminado")


print("\n=== ENTRENANDO REGRESION LOGISTICA ===")

modelo_lr.fit(
    X_train,
    Y_train
)

print("Regresión Logística terminada")


# ==========================================
# PREDICCIONES
# ==========================================

pred_xgb = modelo_xgb.predict(X_test)

pred_rf = modelo_rf.predict(X_test)

pred_lr = modelo_lr.predict(X_test)


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


# ==========================================
# EVALUAR MODELOS
# ==========================================

evaluar_modelo(
    "XGBOOST",
    Y_test,
    pred_xgb
)

evaluar_modelo(
    "RANDOM FOREST",
    Y_test,
    pred_rf
)

evaluar_modelo(
    "REGRESION LOGISTICA",
    Y_test,
    pred_lr
)