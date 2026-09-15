import pandas as pd 

archivo="data/Diabetes_Mexico_DATASET_Imputado.xlsx"

df=pd.read_excel(archivo)

print("Dataset cargado correctamente")
print("Filas: ", df.shape[0])
print("Columnas: ", df.shape[1])

print("\n=== CIUDADES ORIGINALES ===\n")
print(df["Ciudad"].unique())
print("\nNumero de ciudades originales: ", df["Ciudad"].nunique())
df["Ciudad"]=pd.factorize(df["Ciudad"])[0]

print("\n=== ANTECENDENTES DIABETES FAMILIARES ORIGINALES ===\n")
print(df["Antecedente diabetes familiar"].unique())
print("\nNumero de antecedentes originales: ", df["Antecedente diabetes familiar"].nunique())
df["Antecedente diabetes familiar"]=df["Antecedente diabetes familiar"].map({"No":0, "Sí":1, "No sabe":2})

print("\n=== ANTECENDENTES HIPERTENSION FAMILIARES ORIGINALES ===\n")
print(df["Antecedente hipertension familiar"].unique())
print("\nNumero de antecedentes originales: ", df["Antecedente hipertension familiar"].nunique())
df["Antecedente hipertension familiar"]=df["Antecedente hipertension familiar"].map({"No":0, "Sí":1, "No sabe":2})

print("\n=== HIPERTENSION ORIGINALES ===\n")
print(df["Hipertension"].unique())
print("\nNumero de hipertensiones originales: ", df["Hipertension"].nunique())
moda_hipertension=df["Hipertension"].mode()[0]
#print("Moda de Hipertension: ", moda_hipertension)
df["Hipertension"]=df["Hipertension"].fillna(moda_hipertension)
#print("VALORES FALTANTES DESPUES DE IMPUTAR:" , df["Hipertension"].isnull().sum())
df["Hipertension"]=df["Hipertension"].map({"No":0, "Sí":1, "Sí, durante el embarazo":2})

print("\n=== ACTIVIDAD FISICA ORIGINALES ===\n")
print(df["Actividad fisica"].unique())
print("\nNumero de actividades originales: ", df["Actividad fisica"].nunique())
df["Actividad fisica"]=df["Actividad fisica"].map({"Baja":0, "Moderada":1, "Alta":2})

print("\n=== TABAQUISMO ORIGINALES ===\n")
print(df["Tabaquismo"].unique())
print("\nNumero de tabaquismos originales: ", df["Tabaquismo"].nunique())
df["Tabaquismo"]=df["Tabaquismo"].map({"No fuma":0, "Algunos días":1, "Diario":2})

print("\n=== DATOS CODIFICADOS ===\n")
print(df["Ciudad"].head(20))
print(df["Antecedente diabetes familiar"].head(20))
print(df["Antecedente hipertension familiar"].head(20))
print(df["Hipertension"].head(20))
print(df["Actividad fisica"].head(20))
print(df["Tabaquismo"].head(20))

print("\nNumero de columnas antes: ")
print(df.shape[1])

print("\n TIPO DE DATOS)")
print(df["Ciudad"].dtypes)
print(df["Antecedente diabetes familiar"].dtypes)
print(df["Antecedente hipertension familiar"].dtypes)
print(df["Hipertension"].dtypes)
print(df["Actividad fisica"].dtypes)
print(df["Tabaquismo"].dtypes)

archivo_salida="data/Diabetes_Mexico_DATASET_codificado.xlsx"

df.to_excel(archivo_salida, index=False)

print("\nDataset codificado guardado en: ", archivo_salida)

