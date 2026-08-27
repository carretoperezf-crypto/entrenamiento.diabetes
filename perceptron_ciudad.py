import pandas as pd

print("=" * 50)
print("PASO 1 - DATASET CON VARIABLE CATEGÓRICA")
print("=" * 50)

# ============================================================
# 1. CARGAR DATASET
# ============================================================

archivo = "Diabetes_Mexico_IMPUTADO.xlsx"

df = pd.read_excel(archivo)

print("\n==========================================")
print("DATASET CARGADO")
print("==========================================")

print("Archivo:", archivo)
print("Filas:", df.shape[0])
print("Columnas:", df.shape[1])

# ============================================================
# 2. MOSTRAR COLUMNAS
# ============================================================

print("\n==========================================")
print("COLUMNAS")
print("==========================================")

for columna in df.columns:
    print("-", columna)

# ============================================================
# 3. COMPROBAR CIUDAD
# ============================================================

print("\n==========================================")
print("COMPROBANDO VARIABLE CATEGÓRICA")
print("==========================================")

if "Ciudad" in df.columns:

    print("✓ La columna 'Ciudad' existe.")

else:

    print("ERROR: No existe la columna 'Ciudad'.")
    exit()

# ============================================================
# 4. TIPO DE DATO
# ============================================================

print("\n==========================================")
print("TIPO DE DATO DE CIUDAD")
print("==========================================")

print(df["Ciudad"].dtype)

# ============================================================
# 5. CANTIDAD DE CATEGORÍAS
# ============================================================

print("\n==========================================")
print("CATEGORÍAS DE CIUDAD")
print("==========================================")

numero_ciudades = df["Ciudad"].nunique()

print("Número de categorías:")
print(numero_ciudades)

# ============================================================
# 6. MOSTRAR CATEGORÍAS
# ============================================================

print("\n==========================================")
print("LISTA DE CATEGORÍAS")
print("==========================================")

ciudades = df["Ciudad"].dropna().unique()

for ciudad in sorted(ciudades):
    print("-", ciudad)

# ============================================================
# 7. DISTRIBUCIÓN
# ============================================================

print("\n==========================================")
print("DISTRIBUCIÓN POR CATEGORÍA")
print("==========================================")

print(
    df["Ciudad"].value_counts()
)

# ============================================================
# 8. PORCENTAJES
# ============================================================

print("\n==========================================")
print("PORCENTAJE POR CATEGORÍA")
print("==========================================")

porcentajes = (
    df["Ciudad"]
    .value_counts(normalize=True)
    * 100
)

print(porcentajes.round(2))

# ============================================================
# 9. VALORES FALTANTES
# ============================================================

print("\n==========================================")
print("VALORES FALTANTES")
print("==========================================")

print(df.isnull().sum())

print("\nTotal de valores faltantes:")
print(df.isnull().sum().sum())

# ============================================================
# 10. VARIABLE OBJETIVO
# ============================================================

print("\n==========================================")
print("VARIABLE OBJETIVO")
print("==========================================")

print("riesgo_diabetes_cat")

print("\nDistribución:")

print(
    df["riesgo_diabetes_cat"].value_counts()
)

print("\nPorcentajes:")

print(
    (df["riesgo_diabetes_cat"]
     .value_counts(normalize=True) * 100)
    .round(2)
)

# ============================================================
# 11. RESUMEN
# ============================================================

print("\n==========================================")
print("RESUMEN DEL PASO 1")
print("==========================================")

print("Registros:", len(df))
print("Columnas:", len(df.columns))
print("Variable categórica:", "Ciudad")
print("Número de categorías:", numero_ciudades)
print("Variable objetivo:", "riesgo_diabetes_cat")
print(
    "Valores faltantes:",
    df.isnull().sum().sum()
)

print("\n==========================================")
print("PASO 1 TERMINADO")
print("==========================================")

print("Todavía NO se ha modificado el dataset.")
print("Todavía NO se ha aplicado One-Hot Encoding.")
print("Todavía NO se ha dividido Train/Test.")
print("Todavía NO se ha estandarizado.")
print("Todavía NO se ha entrenado el Perceptrón.")