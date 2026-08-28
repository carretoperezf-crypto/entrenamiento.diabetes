import pandas as pd
from imblearn.over_sampling import SMOTE

archivo = "data/Diabetes_Mexico_DATASET_codificado.xlsx"

df = pd.read_excel(archivo)

print("Dataset codificado cargado correctamente")
print("\nDistribución original:")
print(df["riesgo_diabetes_cat"].value_counts())

X = df.drop("riesgo_diabetes_cat", axis=1)
Y = df["riesgo_diabetes_cat"]
print("\n ===DATOS SEPARADOS=== ")
print("X: ", X.shape)
print("Y: ", Y.shape)

smote = SMOTE(
    random_state=42
)

X_smote, Y_smote = smote.fit_resample(
    X,
    Y
)

print("\n=== DISTRIBUCIÓN DESPUÉS DE SMOTE ===")
print(Y_smote.value_counts())

print("\n ===TAMAÑOS=== ")
print("X original: ", X.shape)
print("X despues de SMOTE: ", X_smote.shape)


df_smote=X_smote.copy()
df_smote["riesgo_diabetes_cat"]=Y_smote

df_smote.to_excel(
    "data/Diabetes_Mexico_DATASET_SMOTE.xlsx",
    index=False
)

print("SMOTE TERMINADO")
print("Archivo guardado:")
print("data/Diabetes_Mexico_DATASET_SMOTE.xlsx")