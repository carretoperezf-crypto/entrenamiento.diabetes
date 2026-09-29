from flask import Flask,request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import requests
import math

app = Flask(__name__)
CORS(app)
modelo = joblib.load("modelos/modelo_xgboost.pkl")

@app.route("/")
def inicio():
    return "API de predicción de riesgo de diabetes"    

@app.route("/predict", methods=["POST"])
def predecir():
    try:
        datos = request.get_json()

        paciente = pd.DataFrame([{
            "sexo": datos["sexo"],
            "edad": datos["edad"],
            "Peso": datos["peso"],
            "Estatura(cm)": datos["estatura"],
            "IMC": datos["imc"],
            "Antecedente diabetes familiar": datos["antecedente diabetes familiar"],
            "Antecedente hipertension familiar": datos["antecedente hipertension familiar"],
            "Hipertension": datos["hipertension"],
            "Actividad fisica": datos["actividad fisica"],
            "Tabaquismo": datos["tabaquismo"]
        }])

        prediccion = modelo.predict(paciente)[0]
        if prediccion == 0:
            diagnostico = "Riesgo bajo"
        elif prediccion == 1:
            diagnostico = "Riesgo medio"
        elif prediccion == 2:
            diagnostico = "Riesgo alto"
        else:
            diagnostico = "Riesgo desconocido"
        return jsonify({"prediccion": int(prediccion),
                        "clasificacion": diagnostico})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ==========================================
# HOSPITALES CERCANOS
# ==========================================

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

        # OpenStreetMap / Overpass: hospitales en un radio de 10 km.
        consulta = f"""
        [out:json][timeout:20];
        (
          node["amenity"="hospital"](around:10000,{lat},{lng});
          way["amenity"="hospital"](around:10000,{lat},{lng});
          relation["amenity"="hospital"](around:10000,{lat},{lng});
        );
        out center tags;
        """

        respuesta = requests.post(
            "https://overpass-api.de/api/interpreter",
            data=consulta,
            timeout=25
        )
        respuesta.raise_for_status()
        datos = respuesta.json()

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)