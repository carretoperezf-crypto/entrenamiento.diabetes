import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix, 
                             ConfusionMatrixDisplay,
                             roc_curve,
                             auc)
from sklearn.model_selection import (train_test_split,
                                     learning_curve,
                                     StratifiedKFold)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

archivo_resultados=("data/comparacion_modelos_resultados.csv")
archivo_predicciones=("data/comparacion_modelos_predicciones.csv")
archivo_dataset=("data/Diabetes_Mexico_DATASET_SMOTE.xlsx")

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

columna_objetivo="riesgo_diabetes_cat"
columnas_excluir=[]   #columnas que NO se usaron como variables al entrenar

X=df_dataset.drop(columns=[columna_objetivo]+columnas_excluir)
X=pd.get_dummies(X).to_numpy(dtype=float)   #convierte texto (ej. Ciudad) a numeros
y=df_dataset[columna_objetivo].astype(int).to_numpy()

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)

modelos={
    "XGBoost": XGBClassifier(eval_metric="mlogloss",random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Perceptron": make_pipeline(StandardScaler(),Perceptron(random_state=42)),
    "Árbol de Decisión": DecisionTreeClassifier(random_state=42),
    "Red Neuronal": make_pipeline(StandardScaler(),
                                  MLPClassifier(max_iter=500,random_state=42)),
}

colores_modelos={
    "XGBoost": "#2a78d6",
    "Random Forest": "#eb6834",
    "Perceptron": "#1baf7a",
    "Árbol de Decisión": "#eda100",
    "Red Neuronal": "#4a3aa7",
}

print("\n=== CURVA ROC GENERAL ===")
y_test_bin=label_binarize(y_test,classes=clases)
fpr_comun=np.linspace(0,1,200)
tabla_auc={}

fig,ax=plt.subplots(figsize=(8,7))

for nombre,modelo in modelos.items():
    modelo.fit(X_train,y_train)

    #El Perceptron no tiene predict_proba, se usa su funcion de decision
    if hasattr(modelo,"predict_proba"):
        puntajes=modelo.predict_proba(X_test)
    else:
        puntajes=modelo.decision_function(X_test)

    tpr_promedio=np.zeros_like(fpr_comun)
    tabla_auc[nombre]={}

    for i,nombre_clase in enumerate(nombres_clases):
        fpr,tpr,_=roc_curve(y_test_bin[:,i],puntajes[:,i])
        tabla_auc[nombre][nombre_clase]=auc(fpr,tpr)
        tpr_promedio+=np.interp(fpr_comun,fpr,tpr)

    tpr_promedio/=len(clases)
    tpr_promedio[0]=0.0
    auc_promedio=auc(fpr_comun,tpr_promedio)
    tabla_auc[nombre]["Promedio macro"]=auc_promedio

    ax.plot(fpr_comun,tpr_promedio,linewidth=2,color=colores_modelos[nombre],
            label=f"{nombre} (AUC = {auc_promedio:.3f})")

ax.plot([0,1],[0,1],linestyle=":",color="gray",linewidth=1.5,label="Azar (AUC = 0.500)")
ax.set_title("Curva ROC de todos los modelos (promedio de riesgo bajo, medio y alto)")
ax.set_xlabel("Tasa de falsos positivos")
ax.set_ylabel("Tasa de verdaderos positivos")
ax.set_xlim(0,1)
ax.set_ylim(0,1.02)
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("data/curva_roc_modelos.png",dpi=150)
plt.show()

df_auc=pd.DataFrame(tabla_auc).T
df_auc.index.name="Modelo"
print(df_auc.round(4))

#CURVA DE APRENDIZAJE GENERAL (todos los modelos en una sola grafica)
#Linea continua = validacion cruzada, linea punteada = entrenamiento,
#mismo color por modelo.
print("\n=== CURVA DE APRENDIZAJE GENERAL ===")
validacion_cruzada=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

fig,ax=plt.subplots(figsize=(10,7))

for nombre,modelo in modelos.items():
    print("Modelo: ", nombre)
    tamanos,puntajes_train,puntajes_val=learning_curve(
        modelo,X,y,
        cv=validacion_cruzada,
        train_sizes=np.linspace(0.1,1.0,8),
        scoring="accuracy",
        n_jobs=-1
    )

    media_train=puntajes_train.mean(axis=1)
    media_val=puntajes_val.mean(axis=1)
    std_val=puntajes_val.std(axis=1)
    color=colores_modelos[nombre]

    ax.fill_between(tamanos,media_val-std_val,media_val+std_val,color=color,alpha=0.12)
    ax.plot(tamanos,media_val,marker="o",markersize=6,linewidth=2,color=color,
            label=f"{nombre} - validación")
    ax.plot(tamanos,media_train,linestyle="--",linewidth=1.5,color=color,
            label=f"{nombre} - entrenamiento")

ax.set_title("Curva de Aprendizaje de los modelos")
ax.set_xlabel("Numero de ejemplos de entrenamiento")
ax.set_ylabel("Accuracy")
ax.set_ylim(0,1.02)
ax.legend(loc="center left",bbox_to_anchor=(1.01,0.5),fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("data/curva_aprendizaje_modelos.png",dpi=150,bbox_inches="tight")
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
df_auc.to_csv("data/auc_modelos.csv")

print("\nArchivos generados:")
print("data/evaluacion_comparacion_modelos.csv")
print("data/distribucion_riesgo_ciudades.csv")
print("data/auc_modelos.csv")
print("data/curva_roc_modelos.png")
print("data/curva_aprendizaje_modelos.png")