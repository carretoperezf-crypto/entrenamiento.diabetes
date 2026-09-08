const formulario = document.getElementById("formularioPaciente");
const resultado = document.getElementById("resultado");

formulario.addEventListener("submit", async function(event) {

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
            resultado.innerHTML = ` <h2>Resultado</h2>
            <p>Diagnóstico: ${resultadoAPI.clasificacion}</p>
            <p>Clasificación: ${resultadoAPI.prediccion}</p> `;
        } else {
            resultado.innerHTML = ` <p>Error: ${resultadoAPI.error}</p> `;
        }

    } catch (error) {
        console.error("Error:", error);
        resultado.innerHTML = `
            <p>
                No se pudo conectar con el servidor Flask.
            </p>`;
    }
});