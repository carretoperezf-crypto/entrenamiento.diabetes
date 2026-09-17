// ==========================================
// Configuración
// ==========================================

// Dirección del servidor Flask.
// Si abres la app desde un celular, cambia 127.0.0.1 por la IP de tu
// computadora en la red local (por ejemplo "http://192.168.1.50:5000")
// y ejecuta Flask con app.run(host="0.0.0.0", port=5000).
const API_URL = "http://127.0.0.1:5000";

const formulario = document.getElementById("formularioPaciente");
const resultado = document.getElementById("resultado");
const botonEnviar = formulario.querySelector(".boton-enviar");
const avisoFormulario = document.getElementById("formularioAviso");

const campoPeso = document.getElementById("peso");
const campoEstatura = document.getElementById("estatura");
const campoIMC = document.getElementById("imc");
const panelIMC = document.getElementById("imcPanel");
const ayudaIMC = document.getElementById("imcAyuda");

const reducirMovimiento =
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;


// ==========================================
// Utilidades
// ==========================================

// Evita que texto del servidor se interprete como HTML
function esc(valor) {
    return String(valor ?? "").replace(/[&<>"']/g, (c) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    }[c]));
}

function valorOpcion(nombre) {
    return Number(formulario.elements.namedItem(nombre).value);
}

function desplazarA(elemento) {
    elemento.scrollIntoView({
        behavior: reducirMovimiento ? "auto" : "smooth",
        block: "start"
    });
}


// ==========================================
// IMC automático
// IMC = peso (kg) / estatura (m)²
// ==========================================

function categoriaIMC(imc) {
    if (imc < 18.5) return "Bajo peso";
    if (imc < 25) return "Peso normal";
    if (imc < 30) return "Sobrepeso";
    return "Obesidad";
}

function calcularIMC() {

    const peso = parseFloat(campoPeso.value);
    const estaturaCm = parseFloat(campoEstatura.value);

    const valido =
        peso > 0 && peso <= 400 &&
        estaturaCm >= 50 && estaturaCm <= 250;

    if (!valido) {
        campoIMC.value = "";
        panelIMC.dataset.estado = "vacio";
        ayudaIMC.textContent = "Se calcula con el peso y la estatura.";
        return null;
    }

    const estaturaM = estaturaCm / 100;
    const imc = Math.round((peso / (estaturaM * estaturaM)) * 100) / 100;

    campoIMC.value = imc.toFixed(1);
    panelIMC.dataset.estado = "listo";
    ayudaIMC.textContent = categoriaIMC(imc);

    return imc;
}

campoPeso.addEventListener("input", calcularIMC);
campoEstatura.addEventListener("input", calcularIMC);


// ==========================================
// Contenido por nivel de riesgo
// Las acciones pueden ser texto fijo o una función que recibe el
// perfil del paciente y devuelve texto (o null para omitirla).
// ==========================================

const accionPeso = (p) => {
    if (p.imc < 25) return null;
    const min = (p.peso * 0.05).toFixed(1);
    const max = (p.peso * 0.07).toFixed(1);
    return `Con un IMC de ${p.imc.toFixed(1)}, bajar entre 5 y 7 % del peso ` +
        `(unos ${min} a ${max} kg) reduce de forma importante el riesgo.`;
};

const accionActividad = (p) => p.activo
    ? "Mantén la actividad física: al menos 150 minutos por semana de intensidad moderada."
    : "Empieza con caminatas de 10 minutos al día y aumenta poco a poco hasta 150 minutos por semana.";

const accionTabaco = (p) => p.fuma
    ? "Dejar de fumar reduce el riesgo de diabetes y de sus complicaciones. Pide apoyo en tu centro de salud."
    : null;

const accionPresion = (p) => p.hipertension
    ? "Sigue el tratamiento para la presión arterial: presión alta y diabetes juntas elevan mucho el riesgo del corazón."
    : null;

const NIVELES = {

    bajo: {
        clase: "riesgo-bajo",
        nivel: 1,
        titulo: "Riesgo bajo",
        resumen:
            "Con los datos ingresados, el modelo no encuentra señales importantes " +
            "de riesgo. Mantener los hábitos actuales ayuda a que siga así.",
        consecuencias: [
            "Un riesgo bajo no es cero: la edad, el aumento de peso y el sedentarismo pueden elevarlo con los años.",
            (p) => p.antecedenteDiabetes
                ? "Por el antecedente familiar de diabetes, conviene vigilarlo aunque hoy sea bajo."
                : null
        ],
        acciones: [
            accionActividad,
            "Prioriza verduras, frutas, leguminosas y granos integrales. Limita bebidas azucaradas y ultraprocesados.",
            accionPeso,
            accionTabaco,
            "Hazte un análisis de glucosa al menos cada 3 años, o antes si tu médico lo indica."
        ]
    },

    medio: {
        clase: "riesgo-medio",
        nivel: 2,
        titulo: "Riesgo medio",
        resumen:
            "Algunos datos se asocian con mayor probabilidad de desarrollar diabetes " +
            "tipo 2. Es buen momento para actuar: en esta etapa los cambios de hábitos " +
            "tienen mucho efecto.",
        consecuencias: [
            "Puede existir prediabetes: glucosa por encima de lo normal que casi nunca da síntomas.",
            "Sin cambios, la prediabetes puede avanzar a diabetes tipo 2 en pocos años.",
            "También aumenta el riesgo de enfermedades del corazón y de presión alta."
        ],
        acciones: [
            "Agenda una consulta médica en los próximos meses y pide glucosa en ayunas y hemoglobina glucosilada (HbA1c).",
            accionPeso,
            accionActividad,
            "Reduce bebidas azucaradas, pan y harinas refinadas, y alimentos ultraprocesados.",
            accionTabaco,
            accionPresion
        ]
    },

    alto: {
        clase: "riesgo-alto",
        nivel: 3,
        titulo: "Riesgo alto",
        resumen:
            "Los datos se parecen a los de personas con diabetes tipo 2 o con alto " +
            "riesgo de desarrollarla. No es un diagnóstico, pero sí una razón para " +
            "acudir con un médico pronto.",
        consecuencias: [
            "La diabetes puede pasar años sin síntomas mientras daña los vasos sanguíneos.",
            "Sin control, afecta los ojos (retinopatía), los riñones (nefropatía) y los nervios, sobre todo de los pies (neuropatía).",
            "Aumenta el riesgo de infarto, embolia cerebral y problemas de circulación.",
            "Detectarla a tiempo y controlarla reduce mucho estas complicaciones."
        ],
        acciones: [
            "Acude a consulta médica en las próximas semanas para confirmar con estudios de laboratorio (glucosa en ayunas y HbA1c).",
            "Lleva este resultado y la lista de medicamentos que tomas.",
            "No empieces medicamentos, suplementos ni dietas extremas por tu cuenta.",
            accionPeso,
            accionActividad,
            accionTabaco,
            accionPresion
        ],
        alarma: true,
        hospitales: true
    }
};

function obtenerNivel(clasificacion) {
    const texto = String(clasificacion || "").toLowerCase();
    if (texto.includes("alto")) return NIVELES.alto;
    if (texto.includes("medio") || texto.includes("moderado")) return NIVELES.medio;
    if (texto.includes("bajo")) return NIVELES.bajo;
    return null;
}

function resolverLista(lista, perfil) {
    return lista
        .map((item) => (typeof item === "function" ? item(perfil) : item))
        .filter(Boolean);
}


// ==========================================
// Pintar resultados
// ==========================================

function mostrarCargando() {
    resultado.className = "resultado cargando";
    resultado.innerHTML = `
        <div class="resultado-tarjeta">
            <div class="resultado-simple">
                <h2>Calculando riesgo</h2>
                <p>Espera un momento.</p>
            </div>
        </div>
    `;
}

function mostrarError(titulo, lineas) {
    resultado.className = "resultado resultado-error";
    resultado.innerHTML = `
        <div class="resultado-tarjeta">
            <div class="resultado-simple">
                <h2 tabindex="-1">${esc(titulo)}</h2>
                ${lineas.map((l) => `<p>${esc(l)}</p>`).join("")}
            </div>
        </div>
    `;
    desplazarA(resultado);
}

function medidorHTML(nivelActual) {
    const nombres = ["Bajo", "Medio", "Alto"];
    return `
        <div class="medidor" aria-hidden="true">
            ${nombres.map((nombre, i) => {
                const n = i + 1;
                const clases = [
                    "medidor-tramo",
                    n <= nivelActual ? "activo" : "",
                    n === nivelActual ? "actual" : ""
                ].join(" ");
                return `<span class="${clases}">${nombre}</span>`;
            }).join("")}
        </div>
    `;
}

function mostrarResultado(data, perfil) {

    const nivel = obtenerNivel(data.clasificacion);

    // Clasificación desconocida: se muestra tal cual, sin recomendaciones
    if (!nivel) {
        resultado.className = "resultado";
        resultado.innerHTML = `
            <div class="resultado-tarjeta">
                <div class="resultado-simple">
                    <h2 tabindex="-1">${esc(data.clasificacion ?? "Resultado")}</h2>
                    <p>Predicción del modelo: ${esc(data.prediccion)}</p>
                </div>
            </div>
        `;
        desplazarA(resultado);
        return;
    }

    const consecuencias = resolverLista(nivel.consecuencias, perfil);
    const acciones = resolverLista(nivel.acciones, perfil);

    // Datos resumidos
    const datos = [
        ["IMC", `${perfil.imc.toFixed(1)} · ${categoriaIMC(perfil.imc)}`]
    ];

    if (typeof data.probabilidad === "number") {
        const pct = data.probabilidad <= 1
            ? data.probabilidad * 100
            : data.probabilidad;
        datos.push(["Probabilidad estimada", `${pct.toFixed(0)} %`]);
    }

    if (data.prediccion !== undefined && data.prediccion !== null) {
        datos.push(["Salida del modelo", data.prediccion]);
    }

    resultado.className = `resultado ${nivel.clase}`;

    resultado.innerHTML = `
        <article class="resultado-tarjeta">

            <header class="resultado-encabezado">
                <h2 tabindex="-1">${nivel.titulo}</h2>
                ${medidorHTML(nivel.nivel)}
                <p class="resultado-resumen">${nivel.resumen}</p>
            </header>

            <dl class="resultado-datos">
                ${datos.map(([k, v]) => `
                    <div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>
                `).join("")}
            </dl>

            ${nivel.alarma ? `
                <div class="alarma" role="note">
                    <strong>Busca atención médica de inmediato si hay:</strong>
                    mucha sed, orinar con mucha frecuencia, visión borrosa,
                    pérdida de peso sin explicación, cansancio intenso,
                    náusea o vómito. En una emergencia llama al
                    <a href="tel:911">911</a>.
                </div>
            ` : ""}

            <section class="bloque">
                <h3>Qué puede pasar</h3>
                <ul class="lista-consecuencias">
                    ${consecuencias.map((c) => `<li>${esc(c)}</li>`).join("")}
                </ul>
            </section>

            <section class="bloque">
                <h3>Qué hacer ahora</h3>
                <ol class="lista-acciones">
                    ${acciones.map((a) => `<li>${esc(a)}</li>`).join("")}
                </ol>
            </section>

            ${nivel.hospitales ? `
                <section class="bloque" id="bloqueHospitales">
                    <h3>Hospitales cercanos</h3>
                    <p class="bloque-texto">
                        Usa tu ubicación para ver los hospitales más cercanos
                        donde pueden confirmar el diagnóstico.
                    </p>
                    <button type="button" class="boton-secundario" id="botonHospitales">
                        Buscar hospitales cercanos
                    </button>
                    <div id="listaHospitales"></div>
                </section>
            ` : ""}

            <footer class="resultado-pie">
                <p class="aviso-medico">
                    Esta estimación es orientativa y no sustituye la valoración
                    de un profesional de la salud.
                </p>
                <button type="button" class="boton-secundario" id="botonNueva">
                    Nueva predicción
                </button>
            </footer>

        </article>
    `;

    document.getElementById("botonNueva")
        .addEventListener("click", reiniciar);

    const botonHospitales = document.getElementById("botonHospitales");
    if (botonHospitales) {
        botonHospitales.addEventListener("click", buscarHospitales);
    }

    desplazarA(resultado);
    resultado.querySelector("h2").focus({ preventScroll: true });
}

function reiniciar() {
    formulario.reset();
    formulario.classList.remove("intentado");
    calcularIMC();
    resultado.className = "resultado";
    resultado.innerHTML = "";
    window.scrollTo({ top: 0, behavior: reducirMovimiento ? "auto" : "smooth" });
}


// ==========================================
// Hospitales cercanos
// ==========================================

function obtenerUbicacion() {
    return new Promise((resolve) => {
        if (!("geolocation" in navigator)) {
            resolve(null);
            return;
        }
        navigator.geolocation.getCurrentPosition(
            (pos) => resolve({
                lat: pos.coords.latitude,
                lng: pos.coords.longitude
            }),
            () => resolve(null),
            { enableHighAccuracy: false, timeout: 10000, maximumAge: 300000 }
        );
    });
}

function hospitalHTML(h) {

    const destino = `${h.latitud},${h.longitud}`;
    const urlMapa =
        `https://www.google.com/maps/dir/?api=1&destination=${encodeURIComponent(destino)}`;

    const distancia = typeof h.distancia_km === "number"
        ? `<span class="hospital-distancia">${h.distancia_km.toFixed(1)} km</span>`
        : "";

    const telefono = h.telefono
        ? `<a href="tel:${esc(h.telefono.replace(/[^\d+]/g, ""))}">Llamar</a>`
        : "";

    return `
        <li class="hospital">
            <div class="hospital-cabecera">
                <span class="hospital-nombre">${esc(h.nombre)}</span>
                ${distancia}
            </div>
            <p class="hospital-institucion">${esc(h.institucion)}</p>
            <p class="hospital-direccion">${esc(h.direccion)}</p>
            <div class="hospital-acciones">
                <a class="principal" href="${urlMapa}" target="_blank" rel="noopener">
                    Cómo llegar
                </a>
                ${telefono}
            </div>
        </li>
    `;
}

async function buscarHospitales() {

    const boton = document.getElementById("botonHospitales");
    const contenedor = document.getElementById("listaHospitales");

    boton.disabled = true;
    boton.textContent = "Buscando…";
    contenedor.innerHTML = "";

    const ubicacion = await obtenerUbicacion();

    const params = new URLSearchParams({ limite: "5" });
    if (ubicacion) {
        params.set("lat", ubicacion.lat);
        params.set("lng", ubicacion.lng);
    }

    try {

        const respuesta = await fetch(`${API_URL}/hospitales?${params}`);
        const data = await respuesta.json();

        if (!respuesta.ok) {
            throw new Error(data.error || "No se pudo obtener la lista.");
        }

        const lista = data.hospitales || [];

        if (lista.length === 0) {
            contenedor.innerHTML = `
                <p class="nota">
                    No hay hospitales registrados cerca de tu ubicación.
                    Acude a tu unidad de medicina familiar o centro de salud.
                </p>
            `;
        } else {
            contenedor.innerHTML = `
                <ul class="lista-hospitales">
                    ${lista.map(hospitalHTML).join("")}
                </ul>
                ${ubicacion ? "" : `
                    <p class="nota">
                        No se pudo usar tu ubicación, así que la lista no está
                        ordenada por distancia. Activa la ubicación en tu
                        navegador para ver los más cercanos.
                    </p>
                `}
            `;
        }

        boton.hidden = true;

    } catch (error) {

        console.error("Error al buscar hospitales:", error);

        contenedor.innerHTML = `
            <p class="nota">
                No se pudo cargar la lista de hospitales. Verifica que el
                servidor esté en ejecución e inténtalo de nuevo.
            </p>
        `;
        boton.disabled = false;
        boton.textContent = "Reintentar";
    }
}


// ==========================================
// Envío del formulario
// ==========================================

formulario.addEventListener("submit", async function (event) {

    event.preventDefault();

    formulario.classList.add("intentado");
    avisoFormulario.hidden = true;

    // Validar campos
    if (!formulario.checkValidity()) {
        const faltantes = formulario.querySelectorAll(
            "input:invalid:not([type='radio']), .campo-opciones:has(input:invalid)"
        ).length;

        avisoFormulario.textContent = faltantes === 1
            ? "Falta 1 dato por completar o corregir."
            : `Faltan ${faltantes} datos por completar o corregir.`;
        avisoFormulario.hidden = false;

        formulario.reportValidity();
        return;
    }

    const imcCalculado = calcularIMC();

    if (imcCalculado === null) {
        avisoFormulario.textContent =
            "Revisa el peso y la estatura (entre 50 y 250 cm) para calcular el IMC.";
        avisoFormulario.hidden = false;
        campoEstatura.focus();
        return;
    }

    // Datos para Flask (mismos nombres que espera el modelo)
    const datos = {
        sexo: valorOpcion("sexo"),
        edad: Number(document.getElementById("edad").value),
        peso: Number(campoPeso.value),
        estatura: Number(campoEstatura.value),
        imc: imcCalculado,
        "antecedente diabetes familiar": valorOpcion("antecedente_diabetes_familiar"),
        "antecedente hipertension familiar": valorOpcion("antecedente_hipertension_familiar"),
        hipertension: valorOpcion("hipertension"),
        "actividad fisica": valorOpcion("actividad_fisica"),
        tabaquismo: valorOpcion("tabaquismo")
    };

    // Perfil usado para personalizar las recomendaciones
    const perfil = {
        imc: imcCalculado,
        peso: datos.peso,
        activo: datos["actividad fisica"] === 1,
        fuma: datos.tabaquismo === 1,
        hipertension: datos.hipertension === 1,
        antecedenteDiabetes: datos["antecedente diabetes familiar"] === 1
    };

    console.log("Datos enviados a Flask:", datos);

    botonEnviar.disabled = true;
    mostrarCargando();

    let respuesta;
    let data;

    try {

        respuesta = await fetch(`${API_URL}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(datos)
        });

    } catch (error) {

        // Error de red: el servidor no respondió
        console.error("Error de conexión:", error);
        mostrarError("No se pudo conectar con el servidor", [
            "Verifica que Flask esté en ejecución.",
            `Dirección configurada: ${API_URL}`
        ]);
        botonEnviar.disabled = false;
        return;
    }

    try {
        data = await respuesta.json();
    } catch {
        data = {};
    }

    console.log("Respuesta de Flask:", data);

    if (!respuesta.ok) {
        // El servidor respondió, pero con un error
        mostrarError("No se pudo calcular el riesgo", [
            data.error || `El servidor respondió con el código ${respuesta.status}.`
        ]);
    } else {
        mostrarResultado(data, perfil);
    }

    botonEnviar.disabled = false;
});