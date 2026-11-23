from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import math
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, "Frontend"),
    static_url_path=""
)
CORS(app)


try:
    app.json.ensure_ascii = False
except AttributeError:
    app.config["JSON_AS_ASCII"] = False

paquete = joblib.load(os.path.join(BASE_DIR, "modelos", "modelo_xgboost.pkl"))

modelo = paquete["modelo"]
COLUMNAS = paquete["columnas"]
CLASES = paquete["clases"]

print("Modelo cargado. Entrenado el:", paquete.get("fecha"),
      "| F1 promedio:", round(paquete.get("f1_promedio", 0), 4))

RANGOS = {
    "edad": (20, 95),          # años
    "peso": (30, 250),         # kg
    "estatura": (120, 230)     # cm
}

CODIGOS_PERMITIDOS = {
    "sexo": {0, 1},
    "antecedente diabetes familiar": {0, 1, 2},
    "antecedente hipertension familiar": {0, 1, 2},
    "hipertension": {0, 1, 2},
    "actividad fisica": {0, 1, 2},
    "tabaquismo": {0, 1, 2}
}

AVISO = (
    "Este resultado es una estimación de riesgo, no un diagnóstico. "
    "Solo un estudio de laboratorio y un médico pueden confirmar si tienes diabetes."
)

RECOMENDACIONES = {
    0: "Mantén tus hábitos saludables y revisa tu glucosa en tus chequeos de rutina.",
    1: "Te recomendamos hacerte un estudio de glucosa en ayunas o de hemoglobina "
       "glicosilada (HbA1c) y consultar a tu médico.",
    2: "Te recomendamos acudir pronto con un médico y hacerte un estudio de glucosa "
       "en ayunas o de hemoglobina glicosilada (HbA1c)."
}

UMBRALES = paquete.get("umbrales")   # None si no se guardaron

def decidir_riesgo(probabilidades):
    if not UMBRALES:
        return int(probabilidades.argmax())
    p_bajo, p_medio, p_alto = probabilidades
    if p_alto >= UMBRALES["alto"]:
        return 2
    if p_medio + p_alto >= UMBRALES["medio"]:
        return 1
    return 0

def leer_numero(datos, campo):
    """Convierte un campo a número o lanza un error claro."""
    if campo not in datos or datos[campo] in (None, ""):
        raise ValueError(f"Falta el campo '{campo}'.")
    try:
        return float(datos[campo])
    except (TypeError, ValueError):
        raise ValueError(f"El campo '{campo}' debe ser un número.")


@app.route("/")
def inicio():
    return app.send_static_file("inicio.html")


@app.route("/predict", methods=["POST"])
def predecir():
    try:
        datos = request.get_json(silent=True)
        if not datos:
            return jsonify({"error": "No se recibieron datos."}), 400

        # Valores numéricos con rango
        valores = {}
        for campo, (minimo, maximo) in RANGOS.items():
            valor = leer_numero(datos, campo)
            if not (minimo <= valor <= maximo):
                return jsonify({
                    "error": f"El valor de '{campo}' debe estar entre {minimo} y {maximo}."
                }), 400
            valores[campo] = valor

        # Valores codificados (respuestas de menús)
        for campo, permitidos in CODIGOS_PERMITIDOS.items():
            valor = leer_numero(datos, campo)
            if valor not in permitidos:
                return jsonify({
                    "error": f"Valor no válido para '{campo}'. "
                             f"Opciones permitidas: {sorted(permitidos)}."
                }), 400
            valores[campo] = int(valor)

        # El IMC se calcula aquí para que siempre coincida con peso y estatura
        imc = valores["peso"] / (valores["estatura"] / 100) ** 2

        paciente = pd.DataFrame([{
            "sexo": valores["sexo"],
            "edad": valores["edad"],
            "Peso": valores["peso"],
            "Estatura(cm)": valores["estatura"],
            "IMC": imc,
            "Antecedente diabetes familiar": valores["antecedente diabetes familiar"],
            "Antecedente hipertension familiar": valores["antecedente hipertension familiar"],
            "Hipertension": valores["hipertension"],
            "Actividad fisica": valores["actividad fisica"],
            "Tabaquismo": valores["tabaquismo"]
        }])[COLUMNAS]   # mismo orden de columnas que en el entrenamiento

        # Probabilidad de cada nivel de riesgo
        probabilidades = modelo.predict_proba(paciente)[0]
        prediccion = decidir_riesgo(probabilidades)

        return jsonify({
            "prediccion": prediccion,
            "clasificacion": CLASES.get(prediccion, "Riesgo desconocido"),
            "probabilidades": {
                CLASES[i]: round(float(p) * 100, 1)
                for i, p in enumerate(probabilidades)
            },
            "imc": round(imc, 2),
            "recomendacion": RECOMENDACIONES.get(prediccion, ""),
            "aviso": AVISO
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Error al procesar la predicción.",
                        "detalle": str(e)}), 500

# HOSPITALES CERCANOS
# Lista de hospitales guardada en el proyecto. Se genera en la computadora con
# descargar_hospitales.py (a partir del mapa de Geofabrik) y se sube al repositorio.
# Así la búsqueda es instantánea y no depende de servidores externos.
RUTA_HOSPITALES_LOCALES = os.path.join(BASE_DIR, "datos", "hospitales_mx.json")
# Se busca primero cerca; si hay pocos resultados (zonas rurales), se amplía
RADIOS_BUSQUEDA_KM = (10, 30, 60)


def cargar_hospitales_locales():
    try:
        with open(RUTA_HOSPITALES_LOCALES, encoding="utf-8") as archivo:
            lista = json.load(archivo)
        print(f"Hospitales locales cargados: {len(lista)}")
        return lista
    except FileNotFoundError:
        print("AVISO: no se encontró datos/hospitales_mx.json; "
              "la búsqueda de hospitales no funcionará.")
    except (OSError, ValueError) as e:
        print(f"No se pudo leer el archivo de hospitales locales: {e}")
    return []


HOSPITALES_LOCALES = cargar_hospitales_locales()


def calcular_distancia(lat1, lon1, lat2, lon2):
    """Calcula la distancia aproximada en km usando Haversine."""
    R = 6371.0

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    diferencia_lat = math.radians(lat2 - lat1)
    diferencia_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(diferencia_lat / 2) ** 2
        + math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(diferencia_lon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


@app.route("/hospitales", methods=["GET"])
def hospitales():
    try:
        lat = request.args.get("lat", type=float)
        lng = request.args.get("lng", type=float)
        limite = request.args.get("limite", default=5, type=int)

        if lat is None or lng is None:
            return jsonify({
                "error": "No se recibió la ubicación del dispositivo."
            }), 400

        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            return jsonify({
                "error": "Las coordenadas de ubicación no son válidas."
            }), 400

        limite = max(1, min(limite, 10))

        if not HOSPITALES_LOCALES:
            return jsonify({
                "error": "La lista de hospitales no está disponible."
            }), 503

        # 1. Buscar cerca y ampliar el radio si hay pocos resultados
        radio_maximo = RADIOS_BUSQUEDA_KM[-1]
        margen = radio_maximo / 111 + 0.01   # grados aprox., filtro rápido
        candidatos = []
        for hospital in HOSPITALES_LOCALES:
            if abs(hospital["latitud"] - lat) > margen:
                continue
            distancia = calcular_distancia(
                lat, lng, hospital["latitud"], hospital["longitud"]
            )
            if distancia <= radio_maximo:
                candidatos.append({**hospital, "distancia_km": distancia})

        hospitales_encontrados = []
        for radio in RADIOS_BUSQUEDA_KM:
            hospitales_encontrados = [
                h for h in candidatos if h["distancia_km"] <= radio
            ]
            if len(hospitales_encontrados) >= limite:
                break

        # 2. Si no hay nada en 60 km, mostrar los más cercanos de toda la lista
        if not hospitales_encontrados:
            hospitales_encontrados = [
                {**h, "distancia_km": calcular_distancia(
                    lat, lng, h["latitud"], h["longitud"])}
                for h in HOSPITALES_LOCALES
            ]

        hospitales_encontrados.sort(key=lambda x: x["distancia_km"])

        hospitales_finales = []
        nombres = set()

        for hospital in hospitales_encontrados:
            nombre_normalizado = hospital["nombre"].strip().lower()
            if nombre_normalizado in nombres:
                continue

            nombres.add(nombre_normalizado)
            hospitales_finales.append(hospital)

            if len(hospitales_finales) >= limite:
                break

        return jsonify({
            "hospitales": hospitales_finales,
            "ubicacion": {"lat": lat, "lng": lng}
        })

    except Exception as e:
        print(f"Error inesperado en /hospitales: {e!r}")
        return jsonify({"error": str(e)}), 500

@app.route("/sw.js")
def service_worker():
    respuesta=app.send_static_file("sw.js")
    respuesta.headers["Cache-Control"]="no-cache"
    respuesta.headers["Service-Worker-Allowed"]="/"
    return respuesta

if __name__ == "__main__":
    puerto=int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=puerto, debug=False)