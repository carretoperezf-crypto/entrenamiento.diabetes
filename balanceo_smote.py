import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

archivo = "data/Diabetes_Mexico_DATASET_codificado.xlsx"

df = pd.read_excel(archivo)

print("Dataset codificado cargado correctamente")
print("\nDistribución original:")
print(df["riesgo_diabetes_cat"].value_counts())

X = df.drop("riesgo_diabetes_cat", axis=1)
Y = df["riesgo_diabetes_cat"]
print("\n ===DATOS SEPARADOS=== ")
print("X: ", X.shape)
print("X: ", X.shape)


X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42,
    stratify=Y
)

print("DIVISION TRAIN / TEST")
print("X_train: ", X_train.shape)
print("Y_train: ", Y_train.shape)

print("X_test: ", X_test.shape)
print("Y_test: ", Y_test.shape)


print("\n=== DISTRIBUCIÓN ANTES DE SMOTE ===")
print("TRAIN:")
print(Y_train.value_counts())

print("\nTEST:")
print(Y_test.value_counts())

smote = SMOTE(
    random_state=42
)

X_train_smote, Y_train_smote = smote.fit_resample(
    X_train,
    Y_train
)

print("\n=== DISTRIBUCIÓN DESPUÉS DE SMOTE ===")
print(Y_train_smote.value_counts())

print("\nTamaño original de TRAIN:")
print(X_train.shape)

print("\nTamaño de TRAIN después de SMOTE:")
print(X_train_smote.shape)

print("\n=== TIPOS DE DATOS ===")
print(df.dtypes)

#GUARDAR TEST SIN SMOTE
X_test.to_csv("data/X_test.csv", index=False)
Y_test.to_csv("data/Y_test.csv", index=False)
X_train_smote.to_csv("data/X_train_SMOTE.csv", index=False)
Y_train_smote.to_csv("data/Y_train_SMOTE.csv", index=False)


df_train_smote = X_train_smote.copy()
df_train_smote["riesgo_diabetes_cat"] = Y_train_smote
df_train_smote.to_excel(
    "data/Diabetes_Mexico_DATASET_SMOTE.xlsx",
    index=False
)

print("SMOTE TERMINADO")
print("Archivo guardado:")
print("data/Diabetes_Mexico_DATASET_SMOTE.xlsx")