import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix, 
                             ConfusionMatrixDisplay)

from xgboost import XGBClassifier

X_train_smote=pd.read_csv("data/X_train_smote.csv")
Y_train_smote=pd.read_csv("data/Y_train_smote.csv").squeeze()

X_test=pd.read_csv("data/X_test.csv")
Y_test=pd.read_csv("data/Y_test.csv").squeeze()

modelo=XGBClassifier(
    random_state=42,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    objective="multi:softprob",
    eval_metric="mlogloss",
    num_class=3
)   

print("\n=== ENTRENANDO XGBOOST ===")
modelo.fit(X_train_smote,Y_train_smote)
print("Entrenamiento terminado correctamente")

y_pred=modelo.predict(X_test)

print("\n=== METRICAS DE EVALUACION ===")
accuracy=accuracy_score(Y_test, y_pred)
print("Accuracy: ", accuracy)

print("\n=== MATRIZ DE CONFUSION ===")
matriz=confusion_matrix(Y_test, y_pred)
print(matriz)

print("\n=== REPORTE DE CLASIFICACION ===")
print(classification_report(Y_test, y_pred, zero_division=0))

disp=ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=[0,1,2])
disp.plot()

plt.title("Matriz de Confusión")
plt.show()

