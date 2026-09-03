import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix, 
                             ConfusionMatrixDisplay)

archivo_resultados=("data/comparacion_modelos_resultados.csv")
archivo_predicciones=("data/comparacion_modelos_predicciones.csv")
archivo_dataset=("data/Diabetes_Mexico_DATASET.xlsx")

df_resultados=pd.read_csv(archivo_resultados)
df_predicciones=pd.read_csv(archivo_predicciones)
df_dataset=pd.read_excel(archivo_dataset)

#MATRICES DE CONFUSIÓN
print("\n=== MATRICES DE CONFUSIÓN ===")
clases=[0,1,2]
nombres_clases=["Riesgo bajo", "Riesgo medio", "Riesgo alto"]
columnas_modelos=["XGBoost", "Random Forest", "Perceptron", "Árbol de Decisión", "Red Neuronal"]

for modelo in columnas_modelos:
    matriz=confusion_matrix(
        df_predicciones["Real"],
        df_predicciones[modelo],
        labels=clases
    )
    print("Modelo: ", modelo)

    fig,ax=plt.subplots(figsize=(6,5))
    imagen=ax.imshow(matriz, cmap="Blues")
    ax.set_title("Matriz de Confusión " + modelo)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_xticks(range(len(nombres_clases)))
    ax.set_yticks(range(len(nombres_clases)))
    ax.set_xticklabels(nombres_clases,rotation=30,ha="right")
    ax.set_yticklabels(nombres_clases)

    for i in range(len(clases)):
        for j in range(len(clases)):
            ax.text(j,i,matriz[i,j],ha="center",va="center",
                    color="white" if matriz[i,j]>matriz.max()/2 else "black",
                    fontsize=12, fontweight="bold")

    fig.colorbar(imagen, ax=ax, label="Numero de casos")

    plt.tight_layout()
    plt.show()

#GRAFICA DE COMPARACION 
print("\n=== GRAFICA DE COMPARACIÓN DE MODELOS ===")

df_grafica=df_resultados.set_index("Modelo")
df_grafica[["Accuracy","Precision","Recall","F1"]].plot(kind="bar",figsize=(12,6))
plt.title("Comparación de Modelos")
plt.xlabel("Modelo")
plt.ylabel("Puntaje")
plt.ylim(0,1)
plt.xticks(rotation=45)
plt.legend(title="Metricas")
plt.tight_layout()
plt.show()

#GRAFICA DE CIUDADES 
tabla_ciudades=pd.crosstab(df_dataset["Ciudad"],df_dataset["riesgo_diabetes_cat"])
tabla_ciudades.columns=["Riesgo bajo", "Riesgo medio", "Riesgo alto"]

tabla_ciudades["Total"]=tabla_ciudades.sum(axis=1)
tabla_ciudades=tabla_ciudades.sort_values(by="Total",ascending=False)
top_ciudades=tabla_ciudades.head(15)

top_ciudades_grafica=top_ciudades.drop(columns="Total")
print("\n=== 15 CIUDADES CON MAYOR RIESGO ===")
print(top_ciudades_grafica)

ax=top_ciudades_grafica.plot(kind="bar",figsize=(12,8))
plt.title("Distribución de Riesgo de Diabetes en las 15 ciudades con mayor riesgo")
plt.xlabel("Numero de casos")
plt.ylabel("Ciudad")

plt.legend(title="Categoría de Riesgo")
plt.tight_layout()
plt.show()


#GUARDAR RESULTADOS EN CSV
df_resultados.to_csv("data/evaluacion_comparacion_modelos.csv", index=False)
tabla_ciudades.to_csv("data/distribucion_riesgo_ciudades.csv")

print("\nArchivos generados:")
print("data/evaluacion_comparacion_modelos.csv")
print("data/distribucion_riesgo_ciudades.csv")