(function () {

    var CLAVE = "tema";
    var raiz = document.documentElement;

    function leer() {
        try {
            return localStorage.getItem(CLAVE) || "sistema";
        } catch (e) {
            return "sistema";
        }
    }

    function guardar(valor) {
        try {
            localStorage.setItem(CLAVE, valor);
        } catch (e) {
            // Sin almacenamiento (modo privado): el tema dura solo esta visita
        }
    }

    function aplicar(valor) {
        if (valor === "claro" || valor === "oscuro") {
            raiz.setAttribute("data-tema", valor);
        } else {
            raiz.removeAttribute("data-tema");
        }
    }

    function marcar(valor) {
        var opciones = document.querySelectorAll('input[name="tema"]');
        for (var i = 0; i < opciones.length; i++) {
            opciones[i].checked = opciones[i].value === valor;
        }
    }

    aplicar(leer());

    document.addEventListener("DOMContentLoaded", function () {
        marcar(leer());

        document.addEventListener("change", function (event) {
            var opcion = event.target;
            if (opcion.name !== "tema" || !opcion.checked) return;
            guardar(opcion.value);
            aplicar(opcion.value);
        });
    });

    window.addEventListener("storage", function (event) {
        if (event.key !== CLAVE) return;
        var valor = event.newValue || "sistema";
        aplicar(valor);
        marcar(valor);
    });

})();