"""Marcas de subgrupo del clasificador (CLAUDE.md, sección c, regla 8).

Única definición de las tres marcas que se reportan aparte en validación y test:

- `sin_nombre`: la frase "SIN NOMBRE" como palabras completas en la llave.
- `generico`: todas las palabras de la llave, sin la frase, están en el vocabulario
  de los títulos del SCIAN 2023 (`data/scian_2023_vocabulario.csv`).
- `igual_titulo`: la llave es igual al título normalizado de una clase del catálogo
  o a `nombre_act` normalizado de la misma fila.

Las marcas solo sirven para reportar: nunca son entrada ni variable de un modelo
(la de igual a título usa `nombre_act`, que describe la etiqueta).

Uso: .venv/Scripts/python -m src.subgrupos --vocabulario
Extrae el vocabulario de la estructura oficial del SCIAN 2023, escribe
data/scian_2023_vocabulario.csv y registra su entrada en data/manifest.json.
Imprime solo conteos y hashes.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

from src.denue import REPO_RAIZ, RUTA_MANIFEST, sha256_archivo
from src.normalizacion import normalizar_nombre

VERSION = "2023"
VOCABULARIO = "scian_2023_vocabulario"  # llave en manifest["catalogos"]
RUTA_VOCABULARIO = REPO_RAIZ / "data" / f"{VOCABULARIO}.csv"
ENCABEZADO = "palabra"
PALABRAS_ESPERADAS = 1453  # paso 4: results/2026-11-20_fase1_exploracion-clasificador.json

# Frase con la que el DENUE marca a los negocios sin nombre comercial. No es ninguna
# llave (ninguna llave es solo la frase), así que puede ir en el código y en results/.
MARCADOR = "SIN NOMBRE"
RE_MARCADOR = re.compile(r"\bSIN NOMBRE\b")
RE_PALABRA = re.compile(r"[^\s,\"]+")

CAMPOS_MARCAS = ("sin_nombre", "generico", "igual_titulo")
DEFINICIONES = {
    "sin_nombre": 'La frase "SIN NOMBRE" como palabras completas en cualquier parte de la llave.',
    "generico": (
        'Llave no vacía cuyas palabras, sin la frase "SIN NOMBRE", aparecen todas en el vocabulario '
        "de los títulos normalizados del SCIAN 2023 de todos los niveles (se acepta la palabra con o "
        "sin S o ES final), con al menos una palabra de 4 letras o más."
    ),
    "igual_titulo": (
        "Llave no vacía igual al título normalizado de una clase del catálogo SCIAN 2023 o a "
        "nombre_act normalizado de la misma fila."
    ),
}


class ErrorVocabulario(ValueError):
    """El archivo del vocabulario no tiene el formato esperado."""


# --- Marca "SIN NOMBRE" ---


def tiene_marcador(llave: str) -> bool:
    return bool(RE_MARCADOR.search(llave))


def quitar_marcador(llave: str) -> str:
    """La llave sin la frase "SIN NOMBRE" (el resto), con espacios colapsados."""
    return " ".join(RE_MARCADOR.sub(" ", llave).split())


# --- Marca de genérico ---


def vocabulario(titulos: Iterable[str]) -> frozenset[str]:
    """Palabras de los títulos normalizados con `normalizar_nombre()`."""
    return frozenset(p for t in titulos for p in normalizar_nombre(t).split())


def en_vocabulario(palabra: str, vocab: frozenset[str]) -> bool:
    """La palabra, o la palabra con o sin S o ES final, está en el vocabulario."""
    candidatas = {palabra, palabra + "S", palabra + "ES"}
    if palabra.endswith("S"):
        candidatas.add(palabra[:-1])
    if palabra.endswith("ES"):
        candidatas.add(palabra[:-2])
    return bool(candidatas & vocab)


def es_generico(llave: str, vocab: frozenset[str]) -> bool:
    palabras = quitar_marcador(llave).split()
    return (
        bool(palabras)
        and all(en_vocabulario(p, vocab) for p in palabras)
        and any(len(p) >= 4 for p in palabras)
    )


# --- Marca de igual a título ---


def titulos_normalizados(catalogo: Mapping[str, str]) -> frozenset[str]:
    """Títulos de las clases del catálogo, normalizados con `normalizar_nombre()`."""
    return frozenset(normalizar_nombre(t) for t in catalogo.values())


def es_igual_a_titulo(llave: str, nombre_act: str, titulos: frozenset[str]) -> bool:
    """Llave no vacía igual a un título de clase o a `nombre_act` de la fila, ambos normalizados."""
    return bool(llave) and (llave in titulos or llave == normalizar_nombre(nombre_act))


# --- Las tres marcas ---


def marcas(llave: str, nombre_act: str, vocab: frozenset[str], titulos: frozenset[str]) -> dict[str, bool]:
    """Las tres marcas de una fila, con las claves en el orden de `CAMPOS_MARCAS`."""
    return {
        "sin_nombre": tiene_marcador(llave),
        "generico": es_generico(llave, vocab),
        "igual_titulo": es_igual_a_titulo(llave, nombre_act, titulos),
    }


# --- Archivo del vocabulario ---


def escribir_vocabulario(vocab: Iterable[str], ruta: str | Path) -> None:
    """CSV con encabezado `palabra`; una palabra por línea, ordenadas; UTF-8 sin BOM; saltos "\\n"."""
    palabras = sorted(set(vocab))
    malas = [i for i, p in enumerate(palabras) if not RE_PALABRA.fullmatch(p)]
    if malas:
        raise ErrorVocabulario(f"Palabras vacías o con espacios, comas o comillas en las posiciones {malas}.")
    with open(ruta, "w", encoding="utf-8", newline="") as f:
        escritor = csv.writer(f, lineterminator="\n")
        escritor.writerow([ENCABEZADO])
        escritor.writerows([p] for p in palabras)


def leer_vocabulario(ruta: str | Path) -> frozenset[str]:
    """Lee el CSV del vocabulario; rechaza encabezado distinto, filas mal formadas, repetidos y desorden."""
    with open(ruta, encoding="utf-8", newline="") as f:
        filas = list(csv.reader(f))
    if not filas or filas[0] != [ENCABEZADO]:
        raise ErrorVocabulario(f"{ruta}: el encabezado debe ser {ENCABEZADO}.")
    palabras = []
    for i, fila in enumerate(filas[1:], 2):
        if len(fila) != 1 or not RE_PALABRA.fullmatch(fila[0]):
            raise ErrorVocabulario(f"{ruta}: fila {i} inválida.")
        if palabras and fila[0] <= palabras[-1]:
            raise ErrorVocabulario(f"{ruta}: fila {i} repetida o fuera de orden.")
        palabras.append(fila[0])
    return frozenset(palabras)


# --- Carga verificada contra el manifest ---


def cargar_vocabulario(ruta_manifest: str | Path = RUTA_MANIFEST) -> frozenset[str]:
    """Vocabulario de `data/scian_2023_vocabulario.csv`, con su SHA-256 y su conteo verificados."""
    from src.scian import archivo_verificado, leer_manifest

    entrada = leer_manifest(ruta_manifest)["catalogos"][VOCABULARIO]
    vocab = leer_vocabulario(archivo_verificado(entrada))
    if len(vocab) != entrada["palabras"]:
        raise ErrorVocabulario(f"El vocabulario tiene {len(vocab)} palabras y el manifest registra {entrada['palabras']}.")
    return vocab


def cargar_titulos(ruta_manifest: str | Path = RUTA_MANIFEST) -> frozenset[str]:
    """Títulos normalizados de las clases del catálogo de la versión fijada, con su SHA-256 verificado."""
    from src.scian import CATALOGO, archivo_verificado, leer_catalogo, leer_manifest

    manifest = leer_manifest(ruta_manifest)
    catalogo = leer_catalogo(archivo_verificado(manifest["catalogos"][CATALOGO[manifest["scian_version"]]]))
    titulos = titulos_normalizados(catalogo)
    if len(titulos) != len(catalogo):
        raise ErrorVocabulario("Hay títulos de clase repetidos tras normalizar.")
    return titulos


# --- Programa principal: extraer y registrar el vocabulario ---


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--vocabulario",
        action="store_true",
        help="extrae el vocabulario de la estructura oficial y lo registra en data/manifest.json",
    )
    args = parser.parse_args(argv)
    if not args.vocabulario:
        parser.print_help()
        return

    from src.scian import (
        CANDADO,
        FUENTE_ESTRUCTURA,
        archivo_verificado,
        escribir_manifest,
        extraer_estructura,
        leer_manifest,
        texto_de_paginas,
    )

    manifest = leer_manifest()
    if manifest["scian_version"] != VERSION:
        sys.exit(f"La versión del manifest es {manifest['scian_version']}, no {VERSION}.")
    if VOCABULARIO in manifest["catalogos"] or RUTA_VOCABULARIO.exists():
        sys.exit("El vocabulario ya existe o ya está en el manifest; no se sobrescribe.")

    fuente = FUENTE_ESTRUCTURA[VERSION]
    ruta_pdf = archivo_verificado(manifest["fuentes"][fuente])
    estructura = extraer_estructura(texto_de_paginas(ruta_pdf), CANDADO[VERSION])
    vocab = vocabulario(estructura.categorias.values())
    if len(vocab) != PALABRAS_ESPERADAS:
        sys.exit(f"El vocabulario tiene {len(vocab)} palabras; el paso 4 tenía {PALABRAS_ESPERADAS}.")

    escribir_vocabulario(vocab, RUTA_VOCABULARIO)
    if leer_vocabulario(RUTA_VOCABULARIO) != vocab:
        sys.exit("El vocabulario releído no es igual al extraído.")
    datos = RUTA_VOCABULARIO.read_bytes()
    if datos.startswith(b"\xef\xbb\xbf") or b"\r" in datos:
        sys.exit("El vocabulario tiene BOM o retornos de carro.")

    entrada = {
        "ruta": RUTA_VOCABULARIO.relative_to(REPO_RAIZ).as_posix(),
        "origen": f"fuentes.{fuente}",
        "extractor": "src/subgrupos.py",
        "descripcion": (
            "Palabras de los títulos de todos los niveles (sector a clase) de la estructura oficial, "
            "normalizados con normalizar_nombre(); vocabulario de la marca de genérico (regla 8)."
        ),
        "palabras": len(vocab),
        "codificacion": "utf-8",
        "fin_de_linea": "LF",
        "sha256": sha256_archivo(RUTA_VOCABULARIO),
        "tamano_bytes": RUTA_VOCABULARIO.stat().st_size,
        "fecha_registro": datetime.date.today().isoformat(),
    }
    manifest["catalogos"][VOCABULARIO] = entrada
    escribir_manifest(manifest)
    if cargar_vocabulario() != vocab:
        sys.exit("La carga verificada contra el manifest no da el mismo vocabulario.")

    print(f"Vocabulario: {entrada['palabras']:,} palabras, {entrada['tamano_bytes']:,} bytes")
    print(f"SHA-256: {entrada['sha256']}")
    print(f"Escrito en {entrada['ruta']} y registrado en data/manifest.json (catalogos.{VOCABULARIO})")


if __name__ == "__main__":
    main()
