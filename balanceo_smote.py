import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

ruta="data/Diabetes_Mexico_DATASET.xlsx"

df_limpio=pd.read_excel(ruta)
print("Dataset limpio cargado correctamente")
print("Dimensiones: ", df_limpio.shape)

df_limpio= pd.get_dummies(df_limpio, columns=["Ciudad"], dtype=int)
print("\nCiudad convertida a variables numericas")


X=df_limpio.drop("riesgo_diabetes_cat", axis=1)
Y=df_limpio["riesgo_diabetes_cat"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

print("\n ===DISTRIBUCION ORIGINAL===")
print(Y.value_counts().sort_index())

print("\n ===DISTRIBUCION TRAIN===")
print(Y_train.value_counts().sort_index())

print("\n ===DISTRIBUCION TEST===")
print(Y_test.value_counts().sort_index())

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, Y_train_smote = smote.fit_resample(X_train, Y_train)

print("\n ===DISTRIBUCION DESPUES DE SMOTE===")
print(Y_train_smote.value_counts().sort_index())

X_train_smote.to_csv("data/X_train_smote.csv", index=False)
Y_train_smote.to_csv("data/Y_train_smote.csv", index=False)

X_test.to_csv("data/X_test.csv", index=False)
Y_test.to_csv("data/Y_test.csv", index=False)

print("\n==== ARCHIVOS GUARDADOS CORRECTAMENTE =====")
print("X_train_smote.csv")
print("Y_train_smote.csv")
print("X_test.csv")
print("Y_test.csv")
 