// ==========================================
// Menú de tres rayitas
// Abre y cierra la lista de secciones.
// ==========================================

(function () {

    const boton = document.getElementById("menuBoton");
    const panel = document.getElementById("menuPanel");

    if (!boton || !panel) {
        return;
    }

    function abrir() {
        panel.hidden = false;
        boton.setAttribute("aria-expanded", "true");
        boton.setAttribute("aria-label", "Cerrar menú");
    }

    function cerrar() {
        panel.hidden = true;
        boton.setAttribute("aria-expanded", "false");
        boton.setAttribute("aria-label", "Abrir menú");
    }

    boton.addEventListener("click", function () {
        if (panel.hidden) {
            abrir();
        } else {
            cerrar();
        }
    });

    // Cerrar al tocar fuera del menú
    document.addEventListener("click", function (event) {
        if (!panel.hidden && !event.target.closest(".menu")) {
            cerrar();
        }
    });

    // Cerrar con la tecla Escape y devolver el foco al botón
    document.addEventListener("keydown", function (event) {
        if (event.key === "Escape" && !panel.hidden) {
            cerrar();
            boton.focus();
        }
    });

})();