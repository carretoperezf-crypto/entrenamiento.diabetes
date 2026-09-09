const formulario = document.getElementById("formularioPaciente");
const resultado = document.getElementById("resultado");

function claseDeRiesgo(texto) {
    const valor = (texto || "").toString().toLowerCase();
    if (valor.includes("alto")) return "riesgo-alto";
    if (valor.includes("bajo")) return "riesgo-bajo";
    if (valor.includes("medio") || valor.includes("moderado")) return "riesgo-medio";
    return "";
}

formulario.addEventListener("submit", async function (event) {

    event.preventDefault();

    const datos = {
        sexo: Number(document.getElementById("sexo").value),
        edad: Number(document.getElementById("edad").value),
        peso: Number(document.getElementById("peso").value),
        estatura: Number(document.getElementById("estatura").value),
        imc: Number(document.getElementById("imc").value),
        muestra_suero: Number(document.getElementById("muestra_suero").value),
        insulina: Number(document.getElementById("insulina").value),
        glu_suero: Number(document.getElementById("glu_suero").value),
        creat: Number(document.getElementById("creat").value),
        colest: Number(document.getElementById("colest").value),
        trig: Number(document.getElementById("trig").value)
    };

    console.log("Datos enviados:", datos);

    resultado.className = "resultado";
    resultado.innerHTML = `<div class="resultado-tarjeta"><p>Calculando predicción…</p></div>`;

    try {

        const respuesta = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(datos)
        });

        const resultadoAPI = await respuesta.json();
        console.log("Respuesta de Flask:", resultadoAPI);

        if (respuesta.ok) {
            resultado.className = "resultado " + claseDeRiesgo(resultadoAPI.clasificacion);
            resultado.innerHTML = `
                <div class="resultado-tarjeta">
                    <h2>Resultado de la predicción</h2>
                    <div class="resultado-fila">
                        <span>Diagnóstico</span>
                        <span>${resultadoAPI.clasificacion}</span>
                    </div>
                    <div class="resultado-fila">
                        <span>Clasificación</span>
                        <span>${resultadoAPI.prediccion}</span>
                    </div>
                </div>`;
        } else {
            resultado.className = "resultado resultado-error";
            resultado.innerHTML = `
                <div class="resultado-tarjeta">
                    <h2>No se pudo completar la predicción</h2>
                    <p>${resultadoAPI.error}</p>
                </div>`;
        }

    } catch (error) {
        console.error("Error:", error);
        resultado.className = "resultado resultado-error";
        resultado.innerHTML = `
            <div class="resultado-tarjeta">
                <h2>Sin conexión con el servidor</h2>
                <p>No se pudo conectar con el servidor Flask. Verifica que esté en ejecución en 127.0.0.1:5000.</p>
            </div>`;
    }
});