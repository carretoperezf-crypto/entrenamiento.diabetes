import pandas as pd 

archivo="data/Diabetes_Mexico_DATASET.xlsx"

df=pd.read_excel(archivo)

print("Dataset cargado correctamente")
print("Filas: ", df.shape[0])
print("Columnas: ", df.shape[1])

print("\n=== CIUDADES ORIGINALES ===\n")
print(df["Ciudad"].unique())

print("\nNumero de ciudades originales: ", df["Ciudad"].nunique())

df["Ciudad"]=pd.factorize(df["Ciudad"])[0]

print("\n=== CIUDAD CODIFICADA ===\n")
print(df["Ciudad"].head(20))

print("\nNumero de columnas antes: ")
print(df.shape[1])

print("\n TIPO DE DATOS)")
print(df["Ciudad"].dtypes)

archivo_salida="data/Diabetes_Mexico_DATASET_codificado.xlsx"

df.to_excel(archivo_salida, index=False)

print("\nDataset codificado guardado en: ", archivo_salida)

