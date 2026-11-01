import os
from datetime import date

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score, recall_score
from xgboost import XGBClassifier

archivo = "data/Diabetes_Mexico_DATASET_codificado.xlsx"
df = pd.read_excel(archivo)

print("\n === DISTRIBUCION DE CLASES ====")
print(df["riesgo_diabetes_cat"].value_counts())

columnas = [
    "sexo",
    "edad",
    "Peso",
    "Estatura(cm)",
    "IMC",
    "Antecedente diabetes familiar",
    "Antecedente hipertension familiar",
    "Hipertension",
    "Actividad fisica",
    "Tabaquismo"
]

#VARIABLES CATEGÓRICAS (para SMOTENC)
categoricas = [
    "sexo", "Antecedente diabetes familiar", "Antecedente hipertension familiar",
    "Hipertension", "Actividad fisica", "Tabaquismo"
]
indices_categoricas = [columnas.index(c) for c in categoricas]

X = df[columnas]
Y = df["riesgo_diabetes_cat"]

print("Variables de entrada:", X.shape[1])
print("Registros:", X.shape[0])

#COLUMNAS UTILIZADAS
for columna in X.columns:
    print("-", columna)

# ==========================================
# XGBOOST
# ==========================================
modelo_xgb = XGBClassifier(
    random_state=42,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    objective="multi:softprob",
    eval_metric="mlogloss",
    num_class=3
)

pipe=Pipeline([
    ("smote", SMOTENC(categorical_features=indices_categoricas, random_state=42)),
    ("xgb", modelo_xgb)
])  

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
puntajes = cross_val_score(pipe, X, Y, cv=cv, scoring="f1_macro")

print("\n=== PROMEDIO (VALIDACIÓN CRUZADA, 5 FOLDS) ===")
print("F1 de cada entrenamiento:", np.round(puntajes, 4))
print("F1 promedio:", round(puntajes.mean(), 4), "±", round(puntajes.std(), 4))

# Probabilidades "honestas": cada paciente es predicho por un modelo que no lo vio
proba_cv = cross_val_predict(pipe, X, Y, cv=cv, method="predict_proba")

def clasificar_con_umbrales(proba, umbral_medio, umbral_alto):
    pred = np.zeros(len(proba), dtype=int)                  # por defecto: bajo
    pred[(proba[:, 1] + proba[:, 2]) >= umbral_medio] = 1   # al menos medio
    pred[proba[:, 2] >= umbral_alto] = 2                    # alto
    return pred

# Referencia: cómo le va a argmax
pred_argmax = proba_cv.argmax(axis=1)
print("\n=== ARGMAX ===")
print("F1 macro:", round(f1_score(Y, pred_argmax, average="macro"), 4))
print("Recall riesgo alto:", round(recall_score(Y, pred_argmax, labels=[2], average=None)[0], 4))

# Búsqueda de umbrales
RECALL_MINIMO_ALTO = 0.85   # detectar al menos 85% de los casos de riesgo alto
mejor = None

for u_medio in np.arange(0.20, 0.81, 0.05):
    for u_alto in np.arange(0.15, 0.81, 0.05):
        pred = clasificar_con_umbrales(proba_cv, u_medio, u_alto)
        recall_alto = recall_score(Y, pred, labels=[2], average=None)[0]
        if recall_alto < RECALL_MINIMO_ALTO:
            continue
        f1 = f1_score(Y, pred, average="macro")
        if mejor is None or f1 > mejor["f1"]:
            mejor = {"f1": f1, "medio": round(u_medio, 2),
                     "alto": round(u_alto, 2), "recall_alto": recall_alto}

if mejor is None:
    print("\nNingún par de umbrales alcanzó el recall mínimo; se usará argmax.")
    umbrales = None
else:
    umbrales = {"medio": mejor["medio"], "alto": mejor["alto"]}
    print("\n=== UMBRALES ELEGIDOS ===")
    print("Umbral medio:", umbrales["medio"], "| Umbral alto:", umbrales["alto"])
    print("F1 macro:", round(mejor["f1"], 4))
    print("Recall riesgo alto:", round(mejor["recall_alto"], 4))

# ENTRENAMIENTO FINAL CON TODOS LOS DATOS
#print("\n=== ENTRENANDO XGBOOST ===")
#modelo_xgb.fit(X, Y)
#print("XGBoost terminado")
pipe.fit(X, Y)
modelo_final = pipe.named_steps["xgb"]

# Variables que más influyen en el modelo
importancias = pd.Series(
    modelo_final.feature_importances_,
    index=columnas
).sort_values(ascending=False)

print("\n=== IMPORTANCIA DE LAS VARIABLES ===")
print(importancias.round(4).to_string())


# GUARDADO DEL PAQUETE
os.makedirs("modelos", exist_ok=True)   # crea la carpeta si no existe
paquete = {
    "modelo": modelo_final,
    "columnas": columnas,
    "clases": {0: "Riesgo bajo", 1: "Riesgo medio", 2: "Riesgo alto"},
    "umbrales": umbrales,
    "f1_promedio": puntajes.mean(),
    "fecha": str(date.today())
}

archivo_modelo = "modelos/modelo_xgboost.pkl"
joblib.dump(paquete, archivo_modelo)
print("\nArchivo del modelo guardado en:", archivo_modelo)


# PRUEBA DE CARGA (así se carga en la app)
paquete_cargado = joblib.load(archivo_modelo)
modelo_cargado = paquete_cargado["modelo"]

print("\n=== MODELO XGBOOST CARGADO ===")
print("Fecha de entrenamiento:", paquete_cargado["fecha"])
print("F1 promedio:", round(paquete_cargado["f1_promedio"], 4))
print("Columnas esperadas:", len(paquete_cargado["columnas"]))

# Prueba rápida: predecir el primer registro del dataset
ejemplo = X.iloc[[0]][paquete_cargado["columnas"]]
probabilidades = modelo_cargado.predict_proba(ejemplo)[0]

umbrales_cargados = paquete_cargado["umbrales"]
if umbrales_cargados:
    clase = int(clasificar_con_umbrales(
        probabilidades.reshape(1, -1),
        umbrales_cargados["medio"], umbrales_cargados["alto"]
    )[0])
else:
    clase = int(np.argmax(probabilidades))

print("\n=== PRUEBA DE PREDICCIÓN ===")
print("Resultado:", paquete_cargado["clases"][clase])
for numero, nombre in paquete_cargado["clases"].items():
    print(f"  {nombre}: {probabilidades[numero] * 100:.1f}%")

# Cuántos datos usó el entrenamiento final
X_bal, Y_bal = SMOTENC(categorical_features=indices_categoricas,
                       random_state=42).fit_resample(X, Y)

print("\n=== DATOS DEL ENTRENAMIENTO FINAL ===")
print("Registros originales:", len(X))
print("Registros después de SMOTENC:", len(X_bal))
print("Registros sintéticos creados:", len(X_bal) - len(X))
print("Distribución balanceada:")
print(Y_bal.value_counts().sort_index())

#==============================================================================
#METRICAS DE ENTRENAMIENTO FINAL
#==============================================================================
#Todo se calcula con las probabilidades de validación cruzada (proba_cv),
# es decir, cada paciente evaluado por un modelo que no lo vio al entrenar.
import matplotlib
matplotlib.use("Agg")   # genera imágenes sin abrir ventanas
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score,
    cohen_kappa_score, matthews_corrcoef, log_loss, classification_report
)
 
NOMBRES_CLASES = ["Riesgo bajo", "Riesgo medio", "Riesgo alto"]
COLORES = ["#2E8B57", "#E8A33D", "#C0392B"]   # verde, ámbar, rojo
etiquetas = [0, 1, 2]
 
# Predicción final = la que usará la app (umbrales si existen, si no argmax)
if umbrales:
    pred_final = clasificar_con_umbrales(proba_cv, umbrales["medio"], umbrales["alto"])
    nombre_regla = f"umbrales (medio={umbrales['medio']}, alto={umbrales['alto']})"
else:
    pred_final = pred_argmax
    nombre_regla = "argmax"
 
Y_bin = label_binarize(Y, classes=etiquetas)
 
def metricas_globales(y_real, y_pred):
    return {
        "Exactitud (accuracy)": accuracy_score(y_real, y_pred),
        "Exactitud balanceada": balanced_accuracy_score(y_real, y_pred),
        "Precisión macro": precision_score(y_real, y_pred, average="macro", zero_division=0),
        "Recall macro": recall_score(y_real, y_pred, average="macro", zero_division=0),
        "F1 macro": f1_score(y_real, y_pred, average="macro", zero_division=0),
        "F1 ponderado": f1_score(y_real, y_pred, average="weighted", zero_division=0),
        "Kappa de Cohen": cohen_kappa_score(y_real, y_pred),
        "MCC": matthews_corrcoef(y_real, y_pred),
    }
 
met_argmax = metricas_globales(Y, pred_argmax)
met_final = metricas_globales(Y, pred_final)
auc_macro = roc_auc_score(Y, proba_cv, multi_class="ovr", average="macro")
auc_ponderado = roc_auc_score(Y, proba_cv, multi_class="ovr", average="weighted")
perdida_log = log_loss(Y, proba_cv, labels=etiquetas)
 
# Métricas por clase (con la regla final)
por_clase = pd.DataFrame({
    "Precisión": precision_score(Y, pred_final, labels=etiquetas, average=None, zero_division=0),
    "Recall": recall_score(Y, pred_final, labels=etiquetas, average=None, zero_division=0),
    "F1": f1_score(Y, pred_final, labels=etiquetas, average=None, zero_division=0),
    "AUC ROC": [roc_auc_score(Y_bin[:, i], proba_cv[:, i]) for i in etiquetas],
    "AUC PR": [average_precision_score(Y_bin[:, i], proba_cv[:, i]) for i in etiquetas],
    "Soporte": Y.value_counts().reindex(etiquetas, fill_value=0).values,
}, index=NOMBRES_CLASES)
 
# Especificidad por clase (uno contra el resto)
cm_final = confusion_matrix(Y, pred_final, labels=etiquetas)
especificidad = []
for i in etiquetas:
    vn = cm_final.sum() - cm_final[i, :].sum() - cm_final[:, i].sum() + cm_final[i, i]
    fp = cm_final[:, i].sum() - cm_final[i, i]
    especificidad.append(vn / (vn + fp) if (vn + fp) > 0 else 0)
por_clase.insert(2, "Especificidad", especificidad)
 
print("\n=== REPORTE DE CLASIFICACIÓN (regla final:", nombre_regla, ") ===")
print(classification_report(Y, pred_final, labels=etiquetas,
                            target_names=NOMBRES_CLASES, digits=4, zero_division=0))
print("AUC ROC macro (OvR):", round(auc_macro, 4))
 
def tabla_en_eje(ax, df_tabla, titulo, ancho_col=None):
    ax.axis("off")
    ax.set_title(titulo, fontsize=12, fontweight="bold", loc="left")
    df_tabla = df_tabla.copy()
    df_tabla.insert(0, "", df_tabla.index)   # nombres de fila dentro de la tabla
    t = ax.table(cellText=df_tabla.values, colLabels=df_tabla.columns,
                 loc="upper center", cellLoc="center", colWidths=ancho_col)
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.5)
    for (fila, _), celda in t.get_celld().items():
        if fila == 0:
            celda.set_facecolor("#E8E8E8")
            celda.set_text_props(fontweight="bold")
    return t
 
os.makedirs("reportes", exist_ok=True)
archivo_reporte = "reportes/metricas_modelo_xgboost.pdf"
 
with PdfPages(archivo_reporte) as pdf:
 
    # ---------- PÁGINA 1: RESUMEN ----------
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Reporte de métricas – XGBoost (riesgo de diabetes)",
                 fontsize=16, fontweight="bold", x=0.05, ha="left")
    fig.text(0.05, 0.915,
             f"Fecha: {date.today()}  |  Registros: {len(X)}  |  Variables: {X.shape[1]}  |  "
             f"Validación cruzada estratificada de 5 folds con SMOTENC dentro de cada fold",
             fontsize=9, color="#555555")
 
    ax1 = fig.add_axes([0.05, 0.50, 0.42, 0.36])
    resumen = pd.DataFrame({
        "Argmax": [f"{v:.4f}" for v in met_argmax.values()],
        "Regla final": [f"{v:.4f}" for v in met_final.values()],
    }, index=list(met_argmax.keys()))
    tabla_en_eje(ax1, resumen, "Métricas globales", ancho_col=[0.44, 0.28, 0.28])
 
    ax2 = fig.add_axes([0.53, 0.50, 0.42, 0.36])
    extra = pd.DataFrame({"Valor": [
        f"{auc_macro:.4f}", f"{auc_ponderado:.4f}", f"{perdida_log:.4f}",
        f"{puntajes.mean():.4f} ± {puntajes.std():.4f}",
        "— (se usa argmax)" if umbrales is None else str(umbrales["medio"]),
        "— (se usa argmax)" if umbrales is None else str(umbrales["alto"]),
    ]}, index=["AUC ROC macro (OvR)", "AUC ROC ponderado", "Log loss",
               "F1 macro por fold (media ± DE)", "Umbral medio", "Umbral alto"])
    tabla_en_eje(ax2, extra, "Probabilidades y validación", ancho_col=[0.55, 0.45])
 
    ax3 = fig.add_axes([0.05, 0.08, 0.90, 0.33])
    tabla_por_clase = por_clase.copy()
    for c in tabla_por_clase.columns:
        if c != "Soporte":
            tabla_por_clase[c] = tabla_por_clase[c].map(lambda v: f"{v:.4f}")
    tabla_en_eje(ax3, tabla_por_clase, f"Métricas por clase (regla final: {nombre_regla})",
                 ancho_col=[0.16] + [0.12] * 7)
 
    pdf.savefig(fig)
    plt.close(fig)
 
    # ---------- PÁGINA 2: CURVAS ROC ----------
    fig, ax = plt.subplots(figsize=(11, 8.5))
    for i, (nombre, color) in enumerate(zip(NOMBRES_CLASES, COLORES)):
        fpr, tpr, _ = roc_curve(Y_bin[:, i], proba_cv[:, i])
        ax.plot(fpr, tpr, color=color, lw=2.2,
                label=f"{nombre} (AUC = {auc(fpr, tpr):.3f})")
 
    # Promedio micro: todas las decisiones clase-vs-resto juntas
    fpr_micro, tpr_micro, _ = roc_curve(Y_bin.ravel(), proba_cv.ravel())
    ax.plot(fpr_micro, tpr_micro, color="#333333", lw=1.6, ls="--",
            label=f"Promedio micro (AUC = {auc(fpr_micro, tpr_micro):.3f})")
 
    # Promedio macro: media de las tres curvas
    malla = np.linspace(0, 1, 500)
    tpr_medio = np.zeros_like(malla)
    for i in etiquetas:
        fpr, tpr, _ = roc_curve(Y_bin[:, i], proba_cv[:, i])
        tpr_medio += np.interp(malla, fpr, tpr)
    tpr_medio /= 3
    ax.plot(malla, tpr_medio, color="#3B5BA9", lw=1.6, ls=":",
            label=f"Promedio macro (AUC = {auc(malla, tpr_medio):.3f})")
 
    ax.plot([0, 1], [0, 1], color="#AAAAAA", lw=1, ls="--", label="Azar (AUC = 0.500)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Tasa de falsos positivos (1 − especificidad)", fontsize=11)
    ax.set_ylabel("Tasa de verdaderos positivos (sensibilidad / recall)", fontsize=11)
    ax.set_title("Curvas ROC – uno contra el resto (validación cruzada)",
                 fontsize=14, fontweight="bold", loc="left")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=10, frameon=False)
    pdf.savefig(fig)
    plt.close(fig)
 
    # ---------- PÁGINA 3: CURVAS PRECISIÓN-RECALL ----------
    fig, ax = plt.subplots(figsize=(11, 8.5))
    for i, (nombre, color) in enumerate(zip(NOMBRES_CLASES, COLORES)):
        prec, rec, _ = precision_recall_curve(Y_bin[:, i], proba_cv[:, i])
        ap = average_precision_score(Y_bin[:, i], proba_cv[:, i])
        prevalencia = Y_bin[:, i].mean()
        ax.plot(rec, prec, color=color, lw=2.2, label=f"{nombre} (AP = {ap:.3f})")
        ax.axhline(prevalencia, color=color, lw=0.8, ls=":", alpha=0.7)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precisión", fontsize=11)
    ax.set_title("Curvas Precisión-Recall por clase (líneas punteadas = prevalencia)",
                 fontsize=14, fontweight="bold", loc="left")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=10, frameon=False)
    pdf.savefig(fig)
    plt.close(fig)
 
    # ---------- PÁGINA 4: MATRICES DE CONFUSIÓN ----------
    fig, ejes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle("Matrices de confusión (filas = clase real, columnas = predicha)",
                 fontsize=14, fontweight="bold", x=0.05, ha="left")
    for col, (pred, titulo) in enumerate([(pred_argmax, "Argmax"),
                                          (pred_final, f"Regla final: {nombre_regla}")]):
        cm = confusion_matrix(Y, pred, labels=etiquetas)
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        for fila, (matriz, sufijo, fmt) in enumerate([(cm, "conteos", "d"),
                                                      (cm_norm, "% por fila", ".1%")]):
            ax = ejes[fila, col]
            ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
            for a in range(3):
                for b in range(3):
                    valor = format(matriz[a, b], fmt)
                    ax.text(b, a, valor, ha="center", va="center", fontsize=11,
                            color="white" if cm_norm[a, b] > 0.6 else "black")
            ax.set_xticks(etiquetas, NOMBRES_CLASES, fontsize=9)
            ax.set_yticks(etiquetas, NOMBRES_CLASES, fontsize=9)
            ax.set_title(f"{titulo} ({sufijo})", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close(fig)
 
    # ---------- PÁGINA 5: MÉTRICAS POR CLASE Y POR FOLD ----------
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 8.5),
                                     gridspec_kw={"width_ratios": [2, 1]})
    metricas_barras = ["Precisión", "Recall", "Especificidad", "F1"]
    x = np.arange(len(metricas_barras))
    ancho = 0.26
    for i, (nombre, color) in enumerate(zip(NOMBRES_CLASES, COLORES)):
        valores = por_clase.loc[nombre, metricas_barras].values.astype(float)
        barras = ax_a.bar(x + (i - 1) * ancho, valores, ancho, color=color, label=nombre)
        ax_a.bar_label(barras, fmt="%.2f", fontsize=8, padding=2)
    ax_a.axhline(RECALL_MINIMO_ALTO, color="#C0392B", lw=0.8, ls="--")
    ax_a.text(len(metricas_barras) - 0.5, RECALL_MINIMO_ALTO + 0.01,
              f"recall mínimo alto = {RECALL_MINIMO_ALTO}", ha="right", fontsize=8, color="#C0392B")
    ax_a.set_xticks(x, metricas_barras)
    ax_a.set_ylim(0, 1.08)
    ax_a.set_title("Métricas por clase (regla final)", fontsize=12, fontweight="bold", loc="left")
    ax_a.legend(frameon=False, fontsize=9, loc="lower right")
    ax_a.grid(axis="y", alpha=0.25)
 
    folds = [f"Fold {k+1}" for k in range(len(puntajes))]
    barras = ax_b.bar(folds, puntajes, color="#3B5BA9")
    ax_b.bar_label(barras, fmt="%.3f", fontsize=8, padding=2)
    ax_b.axhline(puntajes.mean(), color="#333333", ls="--", lw=1)
    ax_b.set_ylim(0, 1.08)
    ax_b.set_title(f"F1 macro por fold\n(media = {puntajes.mean():.3f})",
                   fontsize=12, fontweight="bold", loc="left")
    ax_b.tick_params(axis="x", rotation=45)
    ax_b.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
 
    # ---------- PÁGINA 6: IMPORTANCIA DE VARIABLES ----------
    fig, ax = plt.subplots(figsize=(11, 8.5))
    imp = importancias.sort_values()
    barras = ax.barh(imp.index, imp.values, color="#3B5BA9")
    ax.bar_label(barras, fmt="%.3f", fontsize=9, padding=3)
    ax.set_xlabel("Importancia (gain promedio, XGBoost)")
    ax.set_title("Importancia de las variables (modelo final)",
                 fontsize=14, fontweight="bold", loc="left")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
 
# También en Excel, por si quieres copiar los números a otro documento
archivo_metricas_xlsx = "reportes/metricas_modelo_xgboost.xlsx"
with pd.ExcelWriter(archivo_metricas_xlsx) as escritor:
    pd.DataFrame({"Argmax": met_argmax, "Regla final": met_final}).to_excel(
        escritor, sheet_name="Globales")
    por_clase.to_excel(escritor, sheet_name="Por clase")
    pd.DataFrame(confusion_matrix(Y, pred_final, labels=etiquetas),
                 index=[f"Real: {n}" for n in NOMBRES_CLASES],
                 columns=[f"Pred: {n}" for n in NOMBRES_CLASES]).to_excel(
        escritor, sheet_name="Matriz confusion")
    pd.DataFrame({"F1 macro": puntajes}, index=[f"Fold {k+1}" for k in range(len(puntajes))]
                 ).to_excel(escritor, sheet_name="Por fold")
    importancias.to_frame("Importancia").to_excel(escritor, sheet_name="Importancias")
 
print("\nReporte de métricas guardado en:", archivo_reporte)
print("Tablas de métricas guardadas en:", archivo_metricas_xlsx)
 
 
# GUARDADO DEL PAQUETE
os.makedirs("modelos", exist_ok=True)   # crea la carpeta si no existe
paquete = {
    "modelo": modelo_final,
    "columnas": columnas,
    "clases": {0: "Riesgo bajo", 1: "Riesgo medio", 2: "Riesgo alto"},
    "umbrales": umbrales,
    "f1_promedio": puntajes.mean(),
    "fecha": str(date.today())
}
 
archivo_modelo = "modelos/modelo_xgboost.pkl"
joblib.dump(paquete, archivo_modelo)
print("\nArchivo del modelo guardado en:", archivo_modelo)
 
 
# PRUEBA DE CARGA (así se carga en la app)
paquete_cargado = joblib.load(archivo_modelo)
modelo_cargado = paquete_cargado["modelo"]
 
print("\n=== MODELO XGBOOST CARGADO ===")
print("Fecha de entrenamiento:", paquete_cargado["fecha"])
print("F1 promedio:", round(paquete_cargado["f1_promedio"], 4))
print("Columnas esperadas:", len(paquete_cargado["columnas"]))
 
# Prueba rápida: predecir el primer registro del dataset
ejemplo = X.iloc[[0]][paquete_cargado["columnas"]]
probabilidades = modelo_cargado.predict_proba(ejemplo)[0]
 
umbrales_cargados = paquete_cargado["umbrales"]
if umbrales_cargados:
    clase = int(clasificar_con_umbrales(
        probabilidades.reshape(1, -1),
        umbrales_cargados["medio"], umbrales_cargados["alto"]
    )[0])
else:
    clase = int(np.argmax(probabilidades))
 
print("\n=== PRUEBA DE PREDICCIÓN ===")
print("Resultado:", paquete_cargado["clases"][clase])
for numero, nombre in paquete_cargado["clases"].items():
    print(f"  {nombre}: {probabilidades[numero] * 100:.1f}%")
 
# Cuántos datos usó el entrenamiento final
X_bal, Y_bal = SMOTENC(categorical_features=indices_categoricas,
                       random_state=42).fit_resample(X, Y)
 
print("\n=== DATOS DEL ENTRENAMIENTO FINAL ===")
print("Registros originales:", len(X))
print("Registros después de SMOTENC:", len(X_bal))
print("Registros sintéticos creados:", len(X_bal) - len(X))
print("Distribución balanceada:")
print(Y_bal.value_counts().sort_index())

