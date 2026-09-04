from flask import Flask,request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)
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
            "Estatura": datos["estatura"],
            "IMC": datos["imc"],
            "glu_suero": datos["glu_suero"],
            "insulina": datos["insulina"],
            "hb1ac": datos["hb1ac"],
            "trig": datos["trig"],
            "col_hdl": datos["col_hdl"],
            "col_ldl": datos["col_ldl"],
            "colest": datos["colest"],
            "ac_urico": datos["ac_urico"],
            "creat": datos["creat"],
            "albu": datos["albu"]
        }])

        prediccion = modelo.predict(paciente)[0]
        return jsonify({"prediccion": int(prediccion)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)


