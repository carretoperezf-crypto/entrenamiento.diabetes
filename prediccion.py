import pandas as pd
import joblib

archivo_modelo = "modelos/modelo_xgboost.pkl"
modelo = joblib.load(archivo_modelo)

archivo_dataset = ("data/Diabetes_Mexico_DATASET_SMOTE.xlsx")
df = pd.read_excel( archivo_dataset)


columnas= [
    "sexo",
    "edad",
    "Peso",
    "Estatura (cm)",
    "IMC",
    "muestra_suero",
    "insulina",
    "glu_suero",
    "creat",
    "colest",
    "trig"]


X = df[columnas]
print("\n ===COLUMNAS UTILIZADAS=== ")
for columna in X.columns:
    print("-", columna)

paciente = X.iloc[[0]]
print("\n ===DATOS DEL PACIENTE=== ")
print(paciente.to_string(index=False))

prediccion = modelo.predict(paciente)
resultado = int(prediccion[0])

if resultado == 0:
    diagnostico = "Riesgo bajo"
elif resultado == 1:
    diagnostico = "Riesgo medio"
elif resultado == 2:
    diagnostico = "Riesgo alto"
else:
    diagnostico = "Riesgo desconocido"

print("\n ===RESULTADO DE LA PREDICCIÓN=== ")
print("Prediccion:", resultado)
print("Resultado:", diagnostico)