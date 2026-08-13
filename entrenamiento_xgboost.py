import pandas as pd
from xgboost import XGBClassifier

X_train_smote=pd.read_csv("data/X_train_smote.csv")
Y_train_smote=pd.read_csv("data/Y_train_smote.csv").squeeze()

X_test=pd.read_csv("data/X_test.csv")
Y_test=pd.read_csv("data/Y_test.csv").squeeze()

print("=== DATOS DE ENTRENAMIENTO ===")

print("X_train: ", X_train_smote.shape)
print("Y_train: ", Y_train_smote.shape)

print("\n=== DATOS DE PRUEBA ===")

print("X_test: ", X_test.shape)
print("Y_test: ", Y_test.shape)

print("\n=== DISTRIBUCION DE ENTRENAMIENTO ===")
print(Y_train_smote.value_counts().sort_index())

print("\n=== DISTRIBUCION DE PRUEBA ===")
print(Y_test.value_counts().sort_index())   

modelo=XGBClassifier(
    random_state=42,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    objective="multi:softmax",
    eval_metric="mlogloss",
    num_class=3
)

print("\n=== ENTRENANDO XGBOOST ===")
modelo.fit(X_train_smote,Y_train_smote)

print("Entrenamiento terminado correctamente")

y_pred=modelo.predict(X_test)
print("\n=== PREDICCIONES REALIZADAS===")
print(y_pred[:20])

