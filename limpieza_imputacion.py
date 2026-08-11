import pandas as pd

ruta="data/Diabetes_Mexico_DATASET.xlsx"
df=pd.read_excel(ruta)

print("Dataset copiado correctamente")
print("Dimensiones: ", df.shape)

df_limpio = df.copy()
print("\nCopia del dataset creada correctamnete")

print("\n==== VALORES FALTANTES ====\n")
print(df_limpio.isnull().sum())

df_limpio["Peso"]=df_limpio["Peso"].fillna(df_limpio["Peso"].median())
df_limpio["Estatura (cm)"]=df_limpio["Estatura (cm)"].fillna(df_limpio["Estatura (cm)"].median())
df_limpio["IMC"]=df_limpio["IMC"].fillna(df_limpio["IMC"].median())
df_limpio["ponde_hemo"]=df_limpio["ponde_hemo"].fillna(df_limpio["ponde_hemo"].median())
df_limpio["edad"]=df_limpio["edad"].fillna(df_limpio["edad"].median())
df_limpio["ponde_venosa"]=df_limpio["ponde_venosa"].fillna(df_limpio["ponde_venosa"].median())

print ("Imputacion terminada correctamente")

print("\n=== VALORES FALTANTES DESPUES DE LA IMPUTACION===\n")
print(df_limpio.isnull().sum())

salida="data/Diabetes_Mexico_DATASET.xlsx"
df_limpio.to_excel(salida, index=False)

print("\n Dataset limpio guardado en: ")
print(salida)

print("\n==== DISTRIBUCION DE LA VARIABLE OBJETIVO =====")
print(df_limpio["riesgo_diabetes_cat"].value_counts().sort_index())



