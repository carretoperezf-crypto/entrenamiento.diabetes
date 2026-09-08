from flask import Flask,request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

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
            "Estatura (cm)": datos["estatura"],
            "IMC": datos["imc"],
            "muestra_suero": datos["muestra_suero"],
            "insulina": datos["insulina"],
            "glu_suero": datos["glu_suero"],
            "creat": datos["creat"],
            "colest": datos["colest"],
            "trig": datos["trig"]
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

if __name__ == "__main__":
    app.run(debug=True)


