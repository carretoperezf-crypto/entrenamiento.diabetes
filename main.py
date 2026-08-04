import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, 
                             classification_report, 
                             confusion_matrix)

ruta="data/Diabetes_Mexico_DATASET.xlsx"

df=pd.read_excel(ruta)

print("Primeras 5 filas del dataset: ")
print(df.head())

#dimensiones 
print("\n=== Dimensiones del dataset ===")
print(df.shape)

#info general 
print("\n=== Informacion del dataset ===")
print(df.info())

print("\n=== Estadisticas descriptivas ===")
print(df.describe())

# Valores nulos
print("\n=== Valores nulos ===")
print(df.isnull().sum())

# Duplicados
print("\n=== Duplicados ===")
print(df.duplicated().sum())

print("\n === Distribucion de la variable objetivo ===")
print(df["riesgo_diabetes_cat"].value_counts())

print("\n === Distribucion porcentual ===")
print(df["riesgo_diabetes_cat"].value_counts(normalize=True)*100)

df["riesgo_diabetes_cat"].value_counts().sort_index().plot(
    kind="bar",
    figsize=(6,4)
)

plt.title("Distribucion de la variable objetivo")
plt.xlabel("Categoria de riesgo")
plt.ylabel("Numero de pacientes")
plt.show()

df[["Peso","Estatura (cm)", "IMC"]].isnull().sum()
df[df["Peso"].isnull()][["Peso","Estatura (cm)", "IMC"]].head(20)

print("========== VARIABLES DEL DATASET ==========\n")
for i, columna in enumerate(df.columns):
    print(f"{i+1}. {columna}")

print("\n========== TIPO DE DATOS ==========\n")
print(df.dtypes)

#copia del dataset
df_limpio=df.copy()
print("Dataset copiado correctamente")

print("llegue hasta aqui")

df_limpio=pd.get_dummies(
    df_limpio,
    columns=["Ciudad"],
    dtype=int
)

print("termine aqui")

print("\n===== DATASET DESPUÉS DE LA CODIFICACIÓN =====\n")
print(df_limpio.head())

print("\n===== NUEVAS DIMENSIONES =====")
print(df_limpio.shape)

df_limpio["Peso"]=df_limpio["Peso"].fillna(df_limpio["Peso"].median())
df_limpio["Estatura (cm)"]=df_limpio["Estatura (cm)"].fillna(df_limpio["Estatura (cm)"].median())
df_limpio["IMC"]=df_limpio["IMC"].fillna(df_limpio["IMC"].median())
df_limpio["Insulina"]=df_limpio["Insulina"].fillna(df_limpio["Insulina"].median())
df_limpio["Trigliceridos"]=df_limpio["Trigliceridos"].fillna(df_limpio["Trigliceridos"].median())
df_limpio["HbA1c"]=df_limpio["HbA1c"].fillna(df_limpio["HbA1c"].median())

plt.boxplot(df_limpio["IMC"])
plt.title("Boxplot de IMC")
plt.show()

X=df_limpio.drop("riesgo_diabetes_cat", axis=1)
Y=df_limpio["riesgo_diabetes_cat"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

