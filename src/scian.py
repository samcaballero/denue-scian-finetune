"""Catálogo SCIAN: extracción de las clases desde las estructuras oficiales del INEGI.

Lee los PDF de la estructura del SCIAN México (2023 y 2018) con pypdfium2 y saca
el código y el título en español de cada categoría. Cada PDF trae tres secciones
(español, inglés y español-inglés); los títulos se toman solo de las páginas cuyo
encabezado dice "Versión en Español". En el PDF de 2018 la sección en inglés va
primero, así que tomar la primera aparición de cada código daría títulos en inglés.

Uso: .venv/Scripts/python -m src.scian
Escribe data/scian_2023_clases.csv, data/scian_2018_clases.csv y un JSON en
results/. Imprime solo conteos y hashes.
"""

from __future__ import annotations

import csv
import datetime
import json
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import version as version_de_paquete
from pathlib import Path

import pypdfium2 as pdfium

from src.denue import REPO_RAIZ, RUTA_MANIFEST, HashNoCoincideError, sha256_archivo

VERSIONES = ("2023", "2018")
FUENTE_ESTRUCTURA = {"2023": "scian_2023_estructura", "2018": "scian_2018_estructura"}
FUENTE_TABLAS = "scian_tablas_comparativas_2023_2018"
CATALOGO = {"2023": "scian_2023_clases", "2018": "scian_2018_clases"}
NIVELES = {2: "sectores", 3: "subsectores", 4: "ramas", 5: "subramas", 6: "clases"}
# Candado: conteos oficiales que la extracción debe reproducir exactamente.
CANDADO = {
    "2023": {"clases": 1086, "subramas": 610, "ramas": 305},
    "2018": {"clases": 1084, "subramas": 615, "ramas": 306},
}
SEMILLA = 42  # la extracción no usa aleatoriedad; se registra por la regla 1

# Dígitos ASCII ([0-9]), no \d, como la regla 5 de CLAUDE.md.
RE_ESPANOL = re.compile(r"versi[oó]n en espa[nñ]ol", re.IGNORECASE)
RE_OTRO_IDIOMA = re.compile(
    r"english version|versi[oó]n en ingl[eé]s|espa[nñ]ol-ingl[eé]s", re.IGNORECASE
)
RE_NUMERO_PAGINA = re.compile(r"[0-9]{1,3}")
# Encabezado o pie de página; en el PDF de 2018 viene pegado al número de página.
RE_ENCABEZADO = re.compile(r"(?:[0-9]{1,3}\s+)?INEGI\.")
RE_ENTRADA = re.compile(r"([0-9]{2}-[0-9]{2}|[0-9]{2,6})\s+(\S.*)")
RE_CLASE_AL_INICIO = re.compile(r"^\s*([0-9]{6})(?![0-9])", re.MULTILINE)
# Marcas "T" (trilateral) al final del título, pegadas o con un espacio. Pueden venir
# repetidas: en el PDF de 2023 la subrama 23611 dice "Edificación residencialT T".
RE_MARCAS_T = re.compile(r"(?<=[a-záéíóúüñ)])(?:\s?T)+$")
RE_TABLA_COMPARATIVA = re.compile(r"TABLA COMPARATIVA (?:I|II)\.")
RE_CODIGO_6 = re.compile(r"(?<![0-9/])[0-9]{6}(?![0-9])")


class ErrorCatalogo(ValueError):
    """La extracción no pasó una validación; no se escribe ningún catálogo."""


@dataclass
class Estructura:
    """Resultado de extraer la sección en español de una estructura del SCIAN."""

    categorias: dict[str, str]  # código -> título limpio, todos los niveles
    paginas: int
    paginas_espanol: int
    lineas_antes_de_la_primera_entrada: int
    marcas_t_quitadas: dict[str, int]  # títulos con marca "T", por nivel
    codigos_con_marca_t_repetida: list[str]
    lineas_de_clase_todo_el_pdf: int
    codigos_clase_todo_el_pdf: set[str]

    @property
    def clases(self) -> dict[str, str]:
        return {c: t for c, t in self.categorias.items() if nivel(c) == 6}

    def conteos(self) -> dict[str, int]:
        cuenta = Counter(NIVELES[nivel(c)] for c in self.categorias)
        return {nombre: cuenta.get(nombre, 0) for nombre in NIVELES.values()}


def nivel(codigo: str) -> int:
    """2 para sectores (incluidos los rangos como "31-33"); si no, el número de dígitos."""
    return 2 if "-" in codigo else len(codigo)


# --- Lectura del PDF ---


def texto_de_paginas(ruta_pdf: str | Path) -> list[str]:
    """Texto de cada página, en el orden del contenido, con saltos "\\n" y Unicode NFC."""
    pdf = pdfium.PdfDocument(str(ruta_pdf))
    try:
        paginas = []
        for i in range(len(pdf)):
            pagina = pdf[i]
            texto_pagina = pagina.get_textpage()
            texto = texto_pagina.get_text_range()
            texto_pagina.close()
            pagina.close()
            texto = texto.replace("\r\n", "\n").replace("\r", "\n")
            paginas.append(unicodedata.normalize("NFC", texto))
        return paginas
    finally:
        pdf.close()


def es_pagina_en_espanol(texto: str) -> bool:
    return bool(RE_ESPANOL.search(texto)) and not RE_OTRO_IDIOMA.search(texto)


# --- Parser y limpieza ---


def limpiar_titulo(crudo: str) -> tuple[str, int]:
    """Colapsa espacios y quita todas las marcas "T" finales.

    Devuelve (título, número de marcas quitadas).
    """
    titulo = " ".join(crudo.split())
    m = RE_MARCAS_T.search(titulo)
    if not m:
        return titulo, 0
    return titulo[: m.start()], m.group(0).count("T")


def parsear_paginas(paginas: list[str]) -> tuple[list[tuple[str, str]], int]:
    """Entradas (código, título crudo) de las páginas en español, en orden.

    Ignora líneas vacías, números de página y encabezados. Una línea que no empieza
    con código continúa el título anterior, aunque haya un salto de página en medio;
    antes de la primera entrada (portada) se ignora y se cuenta.
    """
    paginas_es = [p for p in paginas if es_pagina_en_espanol(p)]
    if not paginas_es:
        raise ErrorCatalogo('No hay páginas con "Versión en Español".')
    entradas: list[list[str]] = []
    antes = 0
    for pagina in paginas_es:
        for linea in pagina.split("\n"):
            s = linea.strip()
            if not s or RE_NUMERO_PAGINA.fullmatch(s) or RE_ENCABEZADO.match(s):
                continue
            m = RE_ENTRADA.fullmatch(s)
            if m:
                entradas.append([m.group(1), m.group(2)])
            elif entradas:
                entradas[-1][1] += " " + s
            else:
                antes += 1
    return [(c, t) for c, t in entradas], antes


# --- Validaciones ---


def _caracter_latin1_imprimible(c: str) -> bool:
    return 0x20 <= ord(c) <= 0x7E or 0xA1 <= ord(c) <= 0xFF


def validar_titulos(categorias: dict[str, str]) -> None:
    """Títulos no vacíos, sin mayúscula final y solo con caracteres Latin-1 imprimibles.

    Si algo falla, el error lista código, título y motivo de cada caso.
    """
    problemas = []
    for codigo, titulo in sorted(categorias.items()):
        if not titulo:
            problemas.append((codigo, titulo, "título vacío"))
            continue
        if titulo[-1].isupper():
            problemas.append((codigo, titulo, "termina en mayúscula"))
        raros = sorted({c for c in titulo if not _caracter_latin1_imprimible(c)})
        if raros:
            nombres = ", ".join(f"U+{ord(c):04X}" for c in raros)
            problemas.append((codigo, titulo, f"caracteres fuera de Latin-1 imprimible: {nombres}"))
    if problemas:
        detalle = "\n".join(f"  {c} | {t!r} | {motivo}" for c, t, motivo in problemas)
        raise ErrorCatalogo(f"{len(problemas)} títulos no pasan la validación:\n{detalle}")


def _padre(codigo: str, codigos: set[str], sectores: list[str]) -> str | None:
    if nivel(codigo) == 3:
        dos = int(codigo[:2])
        for s in sectores:
            inicio, _, fin = s.partition("-")
            if int(inicio) <= dos <= int(fin or inicio):
                return s
        return None
    return codigo[:-1] if codigo[:-1] in codigos else None


def verificar_jerarquia(categorias: dict[str, str]) -> None:
    """Cada código de 3 a 6 dígitos tiene padre y cada uno de 2 a 5 tiene al menos un hijo."""
    codigos = set(categorias)
    sectores = [c for c in codigos if nivel(c) == 2]
    padres = {c: _padre(c, codigos, sectores) for c in codigos if nivel(c) >= 3}
    sin_padre = sorted(c for c, p in padres.items() if p is None)
    con_hijo = {p for p in padres.values() if p is not None}
    sin_hijo = sorted(c for c in codigos if nivel(c) <= 5 and c not in con_hijo)
    if sin_padre or sin_hijo:
        raise ErrorCatalogo(
            f"Jerarquía rota. Sin padre: {sin_padre}. Sin hijos: {sin_hijo}."
        )


def verificar_candado(conteos: dict[str, int], esperados: dict[str, int]) -> None:
    distintos = {k: (conteos.get(k), v) for k, v in esperados.items() if conteos.get(k) != v}
    if distintos:
        detalle = "; ".join(f"{k}: obtenido {o}, esperado {e}" for k, (o, e) in distintos.items())
        raise ErrorCatalogo(f"El candado no coincide: {detalle}.")


def extraer_estructura(paginas: list[str], candado: dict[str, int] | None) -> Estructura:
    """Extrae y valida la sección en español. Lanza ErrorCatalogo si algo falla."""
    entradas, antes = parsear_paginas(paginas)

    categorias: dict[str, str] = {}
    repetidos: list[str] = []
    marcas: Counter[str] = Counter()
    marca_repetida: list[str] = []
    for codigo, crudo in entradas:
        titulo, n_marcas = limpiar_titulo(crudo)
        if codigo in categorias:
            repetidos.append(codigo)
        categorias[codigo] = titulo
        if n_marcas:
            marcas[NIVELES[nivel(codigo)]] += 1
        if n_marcas > 1:
            marca_repetida.append(codigo)
    if repetidos:
        raise ErrorCatalogo(f"Códigos repetidos en la sección en español: {sorted(set(repetidos))}")

    validar_titulos(categorias)
    verificar_jerarquia(categorias)

    # Códigos únicos: los de 6 dígitos al inicio de línea en todo el PDF (todas las
    # secciones) deben ser exactamente los de la sección en español.
    lineas_clase = [c for p in paginas for c in RE_CLASE_AL_INICIO.findall(p)]
    todo_el_pdf = set(lineas_clase)
    clases_es = {c for c in categorias if nivel(c) == 6}
    if todo_el_pdf != clases_es:
        raise ErrorCatalogo(
            "Los códigos de clase de todo el PDF no coinciden con los de la sección en español. "
            f"Solo en otras secciones: {sorted(todo_el_pdf - clases_es)}. "
            f"Solo en español: {sorted(clases_es - todo_el_pdf)}."
        )

    estructura = Estructura(
        categorias=categorias,
        paginas=len(paginas),
        paginas_espanol=sum(es_pagina_en_espanol(p) for p in paginas),
        lineas_antes_de_la_primera_entrada=antes,
        marcas_t_quitadas={n: marcas.get(n, 0) for n in NIVELES.values()},
        codigos_con_marca_t_repetida=sorted(set(marca_repetida)),
        lineas_de_clase_todo_el_pdf=len(lineas_clase),
        codigos_clase_todo_el_pdf=todo_el_pdf,
    )
    if candado is not None:
        verificar_candado(estructura.conteos(), candado)
    return estructura


def codigos_en_tablas(paginas: list[str]) -> set[str]:
    """Códigos de 6 dígitos en las páginas de las tablas comparativas I y II."""
    return {c for p in paginas if RE_TABLA_COMPARATIVA.search(p) for c in RE_CODIGO_6.findall(p)}


# --- Catálogo en CSV ---


def escribir_catalogo(clases: dict[str, str], ruta: str | Path) -> None:
    """`codigo,titulo`; UTF-8 sin BOM; saltos "\\n"; ordenado por código."""
    with open(ruta, "w", encoding="utf-8", newline="") as f:
        escritor = csv.writer(f, lineterminator="\n")
        escritor.writerow(["codigo", "titulo"])
        for codigo in sorted(clases):
            escritor.writerow([codigo, clases[codigo]])


def leer_catalogo(ruta: str | Path) -> dict[str, str]:
    with open(ruta, encoding="utf-8", newline="") as f:
        filas = list(csv.reader(f))
    if not filas or filas[0] != ["codigo", "titulo"]:
        raise ErrorCatalogo(f"{ruta}: el encabezado debe ser codigo,titulo.")
    clases: dict[str, str] = {}
    for fila in filas[1:]:
        if len(fila) != 2 or not re.fullmatch(r"[0-9]{6}", fila[0]) or fila[0] in clases:
            raise ErrorCatalogo(f"{ruta}: fila inválida o repetida: {fila[:1]}")
        clases[fila[0]] = fila[1]
    return clases


# --- Manifest ---


def leer_manifest(ruta_manifest: str | Path = RUTA_MANIFEST) -> dict:
    with open(ruta_manifest, encoding="utf-8") as f:
        return json.load(f)


def escribir_manifest(manifest: dict, ruta_manifest: str | Path = RUTA_MANIFEST) -> None:
    """Escribe el manifest con el formato del repo: indentación 2, UTF-8 sin escapar, "\\n" final.

    Escribe primero un archivo temporal y luego lo cambia de nombre, para no dejar un
    manifest a medias si algo falla.
    """
    ruta = Path(ruta_manifest)
    texto = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    temporal = ruta.with_name(ruta.name + ".tmp")
    temporal.write_bytes(texto.encode("utf-8"))
    temporal.replace(ruta)


def archivo_verificado(entrada: dict) -> Path:
    """Ruta (relativa a la raíz del repo) de una entrada del manifest, con su SHA-256 verificado."""
    ruta = REPO_RAIZ / entrada["ruta"]
    sha = sha256_archivo(ruta)
    if sha != entrada["sha256"]:
        raise HashNoCoincideError(
            f"El SHA-256 de {ruta} no coincide con el manifest: "
            f"calculado {sha}, registrado {entrada['sha256']}."
        )
    return ruta


def _git(*args: str) -> str:
    resultado = subprocess.run(
        ["git", *args], cwd=REPO_RAIZ, capture_output=True, text=True, check=True
    )
    return resultado.stdout.strip()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    fecha = datetime.date.today().isoformat()
    commit = _git("rev-parse", "HEAD")
    arbol_limpio = _git("status", "--porcelain", "--untracked-files=no") == ""
    fuentes = leer_manifest()["fuentes"]

    # 1. Extraer y validar las dos versiones antes de escribir nada.
    estructuras: dict[str, Estructura] = {}
    for v in VERSIONES:
        ruta_pdf = archivo_verificado(fuentes[FUENTE_ESTRUCTURA[v]])
        estructuras[v] = extraer_estructura(texto_de_paginas(ruta_pdf), CANDADO[v])

    # 2. Escribir los catálogos.
    catalogos = {}
    for v in VERSIONES:
        ruta = REPO_RAIZ / "data" / f"{CATALOGO[v]}.csv"
        escribir_catalogo(estructuras[v].clases, ruta)
        catalogos[v] = {
            "ruta": ruta.relative_to(REPO_RAIZ).as_posix(),
            "sha256": sha256_archivo(ruta),
            "tamano_bytes": ruta.stat().st_size,
        }

    resultado = {
        "fecha": fecha,
        "experimento": "fase1_paso2b_catalogo_scian",
        "commit": commit,
        "arbol_limpio": arbol_limpio,
        "modelo": None,
        "configuracion": {
            "script": "src/scian.py",
            "libreria_pdf": f"pypdfium2 {version_de_paquete('pypdfium2')}",
            "seccion_usada": 'páginas con "Versión en Español" (sin "Versión en Inglés" ni "Español-Inglés")',
            "candado": CANDADO,
        },
        "semilla": SEMILLA,
        "split": "ninguno: no lee el DENUE",
        "hash_split": None,
        "hash_split_origen": "no aplica: este paso no lee el DENUE ni splits",
        "hashes_fuentes": {FUENTE_ESTRUCTURA[v]: fuentes[FUENTE_ESTRUCTURA[v]]["sha256"] for v in VERSIONES},
        "tamano_muestra": {f"clases_{v}": len(estructuras[v].clases) for v in VERSIONES},
        "horas_gpu": None,
        "versiones": {
            v: {
                "paginas_pdf": e.paginas,
                "paginas_espanol": e.paginas_espanol,
                "conteos_por_nivel": e.conteos(),
                "candado_cumplido": True,
                "lineas_de_clase_todo_el_pdf": e.lineas_de_clase_todo_el_pdf,
                "codigos_clase_unicos_todo_el_pdf": len(e.codigos_clase_todo_el_pdf),
                "codigos_clase_seccion_espanol": len(e.clases),
                "lineas_ignoradas_antes_de_la_primera_entrada": e.lineas_antes_de_la_primera_entrada,
                "marcas_t_quitadas_por_nivel": e.marcas_t_quitadas,
                "titulos_con_marca_t_repetida": {
                    "total": len(e.codigos_con_marca_t_repetida),
                    "codigos": e.codigos_con_marca_t_repetida,
                },
                "catalogo": catalogos[v],
            }
            for v, e in estructuras.items()
        },
    }
    salida = REPO_RAIZ / "results" / f"{fecha}_fase1_catalogo-scian.json"
    with open(salida, "w", encoding="utf-8", newline="\n") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"commit {commit[:7]}  |  árbol limpio: {arbol_limpio}")
    for v, e in estructuras.items():
        print(
            f"SCIAN {v}: {e.paginas} páginas ({e.paginas_espanol} en español); "
            f"conteos {e.conteos()}; códigos de clase únicos en todo el PDF "
            f"{len(e.codigos_clase_todo_el_pdf)} ({e.lineas_de_clase_todo_el_pdf} líneas); "
            f"portada ignorada {e.lineas_antes_de_la_primera_entrada} líneas; "
            f"marcas T quitadas {e.marcas_t_quitadas}; "
            f"marca T repetida en {e.codigos_con_marca_t_repetida}"
        )
        print(f"  {catalogos[v]['ruta']}: {catalogos[v]['tamano_bytes']:,} bytes, SHA-256 {catalogos[v]['sha256']}")
    print(f"Guardado en {salida.relative_to(REPO_RAIZ).as_posix()}")


if __name__ == "__main__":
    main()
