import json
import os
import sys
import time

import osmium

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVO_PBF = os.path.join(BASE_DIR, "mexico-latest.osm.pbf")
RUTA_SALIDA = os.path.join(BASE_DIR, "datos", "hospitales_mx.json")


def es_hospital(tags):
    return tags.get("amenity") == "hospital" and tags.get("name")


def datos_basicos(tags):
    calle = tags.get("addr:street")
    numero = tags.get("addr:housenumber")
    colonia = tags.get("addr:suburb")
    partes = []
    if calle:
        partes.append(calle + (f" #{numero}" if numero else ""))
    if colonia:
        partes.append(colonia)
    return {
        "nombre": tags.get("name").strip(),
        "institucion": tags.get("operator") or "Hospital",
        "direccion": ", ".join(partes) or "Dirección no disponible",
        "telefono": tags.get("phone") or tags.get("contact:phone"),
    }


class Paso1(osmium.SimpleHandler):
    """Encuentra los hospitales: puntos, edificios (ways) y conjuntos (relations)."""

    def __init__(self):
        super().__init__()
        self.puntos = []           # hospitales que son un solo punto
        self.ways = {}             # id -> (datos, lista de nodos)
        self.relaciones = []       # (datos, ids de ways miembros)

    def node(self, n):
        tags = dict(n.tags)
        if es_hospital(tags) and n.location.valid():
            self.puntos.append({**datos_basicos(tags),
                                "latitud": round(n.location.lat, 6),
                                "longitud": round(n.location.lon, 6)})

    def way(self, w):
        tags = dict(w.tags)
        if es_hospital(tags):
            self.ways[w.id] = (datos_basicos(tags), [n.ref for n in w.nodes])

    def relation(self, r):
        tags = dict(r.tags)
        if es_hospital(tags):
            miembros = [m.ref for m in r.members if m.type == "w"]
            if miembros:
                self.relaciones.append((datos_basicos(tags), miembros))


class Paso2(osmium.SimpleHandler):
    """Obtiene los nodos de los ways que forman parte de una relation."""

    def __init__(self, ids_buscados):
        super().__init__()
        self.ids = ids_buscados
        self.nodos_de_way = {}

    def way(self, w):
        if w.id in self.ids:
            self.nodos_de_way[w.id] = [n.ref for n in w.nodes]


class Paso3(osmium.SimpleHandler):
    """Obtiene la ubicación de los nodos necesarios."""

    def __init__(self, ids_buscados):
        super().__init__()
        self.ids = ids_buscados
        self.ubicaciones = {}

    def node(self, n):
        if n.id in self.ids and n.location.valid():
            self.ubicaciones[n.id] = (n.location.lat, n.location.lon)


def centro(ids_nodos, ubicaciones):
    # dict.fromkeys quita el nodo repetido que cierra cada polígono
    puntos = [ubicaciones[i] for i in dict.fromkeys(ids_nodos) if i in ubicaciones]
    if not puntos:
        return None
    lat = sum(p[0] for p in puntos) / len(puntos)
    lng = sum(p[1] for p in puntos) / len(puntos)
    return round(lat, 6), round(lng, 6)


def main():
    ruta = sys.argv[1] if len(sys.argv) > 1 else ARCHIVO_PBF
    ruta = os.path.expanduser(ruta)
    if not os.path.exists(ruta):
        print(f"No encontré el archivo: {ruta}\n"
              "Descárgalo de https://download.geofabrik.de/north-america/mexico-latest.osm.pbf")
        sys.exit(1)

    inicio = time.time()

    print("Paso 1 de 3: buscando hospitales …")
    p1 = Paso1()
    p1.apply_file(ruta)
    print(f"  {len(p1.puntos)} puntos, {len(p1.ways)} edificios, "
          f"{len(p1.relaciones)} conjuntos")

    ids_ways_rel = {w for _, miembros in p1.relaciones for w in miembros}
    nodos_de_way = {}
    if ids_ways_rel:
        print("Paso 2 de 3: leyendo los conjuntos …")
        p2 = Paso2(ids_ways_rel)
        p2.apply_file(ruta)
        nodos_de_way = p2.nodos_de_way

    ids_nodos = {n for _, nodos in p1.ways.values() for n in nodos}
    for nodos in nodos_de_way.values():
        ids_nodos.update(nodos)

    print("Paso 3 de 3: obteniendo ubicaciones …")
    p3 = Paso3(ids_nodos)
    p3.apply_file(ruta)

    hospitales = list(p1.puntos)
    for datos, nodos in p1.ways.values():
        c = centro(nodos, p3.ubicaciones)
        if c:
            hospitales.append({**datos, "latitud": c[0], "longitud": c[1]})
    for datos, miembros in p1.relaciones:
        nodos = [n for w in miembros for n in nodos_de_way.get(w, [])]
        c = centro(nodos, p3.ubicaciones)
        if c:
            hospitales.append({**datos, "latitud": c[0], "longitud": c[1]})

    # Quitar duplicados (mismo nombre a menos de ~100 m)
    unicos, vistos = [], set()
    for h in hospitales:
        clave = (h["nombre"].lower(), round(h["latitud"], 3), round(h["longitud"], 3))
        if clave not in vistos:
            vistos.add(clave)
            unicos.append(h)

    if not unicos:
        print("No se encontró ningún hospital. ¿Es el archivo correcto?")
        sys.exit(1)

    os.makedirs(os.path.dirname(RUTA_SALIDA), exist_ok=True)
    with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
        json.dump(unicos, f, ensure_ascii=False, separators=(",", ":"))

    kb = os.path.getsize(RUTA_SALIDA) / 1024
    print(f"\nListo en {time.time() - inicio:.0f} s: {len(unicos)} hospitales "
          f"guardados en {RUTA_SALIDA} ({kb:.0f} KB).")


if __name__ == "__main__":
    main()