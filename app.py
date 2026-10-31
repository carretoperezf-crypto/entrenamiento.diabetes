from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import requests
import math
import os

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
        prediccion = int(probabilidades.argmax())

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
SERVIDORES_OVERPASS = [
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

ENCABEZADOS_OVERPASS = {
    "User-Agent": "PrediccionDiabetesMX/1.0 (proyecto escolar)",
}


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

        consulta = f"""
        [out:json][timeout:15];
        (
          node["amenity"="hospital"](around:10000,{lat},{lng});
          way["amenity"="hospital"](around:10000,{lat},{lng});
          relation["amenity"="hospital"](around:10000,{lat},{lng});
        );
        out center tags;
        """

        datos = None
        errores = []

        for servidor in SERVIDORES_OVERPASS:
            try:
                respuesta = requests.post(
                    servidor,
                    data={"data": consulta},
                    headers=ENCABEZADOS_OVERPASS,
                    timeout=20
                )
                respuesta.raise_for_status()
                datos = respuesta.json()
                print(f"Overpass OK: {servidor}")
                break
            except (requests.RequestException, ValueError) as e:
                errores.append(f"{servidor} -> {e}")
                print(f"Overpass falló: {servidor} -> {e}")

        if datos is None:
            raise requests.RequestException(" | ".join(errores))

        hospitales_encontrados = []

        for elemento in datos.get("elements", []):
            tags = elemento.get("tags", {})
            nombre = tags.get("name")

            if not nombre:
                continue

            if elemento.get("type") == "node":
                lat_hospital = elemento.get("lat")
                lng_hospital = elemento.get("lon")
            else:
                centro = elemento.get("center", {})
                lat_hospital = centro.get("lat")
                lng_hospital = centro.get("lon")

            if lat_hospital is None or lng_hospital is None:
                continue

            distancia = calcular_distancia(
                lat, lng, lat_hospital, lng_hospital
            )

            calle = tags.get("addr:street")
            numero = tags.get("addr:housenumber")
            colonia = tags.get("addr:suburb")

            partes_direccion = []
            if calle:
                partes_direccion.append(calle + (f" #{numero}" if numero else ""))
            if colonia:
                partes_direccion.append(colonia)

            direccion = ", ".join(partes_direccion) or "Dirección no disponible"

            telefono = tags.get("phone") or tags.get("contact:phone")
            institucion = tags.get("operator") or "Hospital"

            hospitales_encontrados.append({
                "nombre": nombre,
                "institucion": institucion,
                "direccion": direccion,
                "telefono": telefono,
                "latitud": lat_hospital,
                "longitud": lng_hospital,
                "distancia_km": distancia
            })

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

    except requests.RequestException as e:
        return jsonify({
            "error": "No se pudo consultar el servicio de hospitales.",
            "detalle": str(e)
        }), 502

    except Exception as e:
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
    