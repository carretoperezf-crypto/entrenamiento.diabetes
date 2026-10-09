(function () {

    const boton = document.getElementById("menuBoton");
    const panel = document.getElementById("menuPanel");

    if (!boton || !panel) {
        return;
    }

    const escritorio = window.matchMedia("(min-width: 961px)");

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

    function ajustar() {
        if (escritorio.matches) {
            panel.hidden = false;
            boton.setAttribute("aria-expanded", "false");
        } else {
            cerrar();
        }
    }

    boton.addEventListener("click", function (event) {
        event.stopPropagation();
        if (panel.hidden) {
            abrir();
        } else {
            cerrar();
        }
    });

    // Cerrar al tocar fuera del menú (solo en celular)
    document.addEventListener("click", function (event) {
        if (escritorio.matches || panel.hidden) return;
        if (!event.target.closest(".menu, #menuBoton")) {
            cerrar();
        }
    });

    // Cerrar con la tecla Escape y devolver el foco al botón
    document.addEventListener("keydown", function (event) {
        if (event.key === "Escape" && !escritorio.matches && !panel.hidden) {
            cerrar();
            boton.focus();
        }
    });

    escritorio.addEventListener("change", ajustar);
    ajustar();

})();