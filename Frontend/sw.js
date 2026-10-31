// Service worker: permite instalar la app y guarda las páginas para abrir más rápido.
// La predicción y los hospitales SIEMPRE se piden al servidor (nunca se guardan).

const VERSION = "v1";
const CACHE = "riesgo-diabetes-" + VERSION;

const RUTAS_API = ["/predict", "/hospitales"];

self.addEventListener("install", (evento) => {
  evento.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll(["/", "/manifest.json"]))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (evento) => {
  evento.waitUntil(
    caches.keys().then((nombres) =>
      Promise.all(nombres.filter((n) => n !== CACHE).map((n) => caches.delete(n)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (evento) => {
  const peticion = evento.request;
  const url = new URL(peticion.url);

  // Solo GET de nuestro propio sitio; la API va directo a internet
  if (peticion.method !== "GET" || url.origin !== self.location.origin) return;
  if (RUTAS_API.some((ruta) => url.pathname.startsWith(ruta))) return;

  // Primero internet; si no hay conexión, usa la copia guardada
  evento.respondWith(
    fetch(peticion)
      .then((respuesta) => {
        if (respuesta.ok) {
          const copia = respuesta.clone();
          caches.open(CACHE).then((cache) => cache.put(peticion, copia));
        }
        return respuesta;
      })
      .catch(() => caches.match(peticion).then((r) => r || caches.match("/")))
  );
});
