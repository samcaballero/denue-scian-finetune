"""Cargador único del DENUE (INEGI, CDMX, edición 05/2026).

Toda lectura de columnas del CSV original pasa por `cargar_denue()`. La ruta, el
SHA-256, el número de filas y la codificación del archivo salen de
`data/manifest.json`, que es la única fuente de esos datos.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

REPO_RAIZ = Path(__file__).resolve().parents[1]
RUTA_MANIFEST = REPO_RAIZ / "data" / "manifest.json"
FUENTE_DENUE = "denue_cdmx"

# Columnas sensibles (CLAUDE.md, sección d): nunca se cargan.
COLUMNAS_PROHIBIDAS = frozenset({"telefono", "correoelec", "www", "raz_social"})

# Las 38 columnas no sensibles del CSV, en el orden del archivo.
COLUMNAS_PERMITIDAS = (
    "id",
    "clee",
    "nom_estab",
    "codigo_act",
    "nombre_act",
    "per_ocu",
    "tipo_vial",
    "nom_vial",
    "tipo_v_e_1",
    "nom_v_e_1",
    "tipo_v_e_2",
    "nom_v_e_2",
    "tipo_v_e_3",
    "nom_v_e_3",
    "numero_ext",
    "letra_ext",
    "edificio",
    "edificio_e",
    "numero_int",
    "letra_int",
    "tipo_asent",
    "nomb_asent",
    "tipoCenCom",
    "nom_CenCom",
    "num_local",
    "cod_postal",
    "cve_ent",
    "entidad",
    "cve_mun",
    "municipio",
    "cve_loc",
    "localidad",
    "ageb",
    "manzana",
    "tipoUniEco",
    "latitud",
    "longitud",
    "fecha_alta",
)


class ColumnaProhibidaError(ValueError):
    """Se pidió una columna sensible (datos de contacto o razón social)."""


class HashNoCoincideError(ValueError):
    """El SHA-256 del archivo no coincide con el registrado en el manifest."""


def sha256_archivo(ruta: str | Path, bloque: int = 1 << 20) -> str:
    """SHA-256 de un archivo, leído en bloques de 1 MiB."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        while trozo := f.read(bloque):
            h.update(trozo)
    return h.hexdigest()


def entrada_manifest(
    fuente: str = FUENTE_DENUE, ruta_manifest: str | Path = RUTA_MANIFEST
) -> dict:
    """Datos de una fuente en el manifest, con la ruta ya resuelta.

    Las rutas relativas del manifest son relativas a la raíz del repo.
    """
    with open(ruta_manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    try:
        entrada = manifest["fuentes"][fuente]
    except KeyError as e:
        raise KeyError(f"La fuente {fuente!r} no está en {ruta_manifest}") from e
    faltan = {"ruta", "sha256", "filas", "codificacion"} - entrada.keys()
    if faltan:
        raise KeyError(f"A la fuente {fuente!r} del manifest le faltan: {sorted(faltan)}")
    return {
        "ruta": REPO_RAIZ / entrada["ruta"],  # si la ruta es absoluta, se queda igual
        "sha256": entrada["sha256"],
        "filas": int(entrada["filas"]),
        "codificacion": entrada["codificacion"],
    }


def _validar_columnas(columnas: Sequence[str]) -> list[str]:
    if isinstance(columnas, str):
        raise ValueError("`columnas` debe ser una lista de nombres, no un solo texto.")
    columnas = list(columnas)
    if not columnas:
        raise ValueError("Hay que pedir al menos una columna.")
    if len(set(columnas)) != len(columnas):
        raise ValueError("La lista de columnas tiene repetidos.")
    prohibidas = [c for c in columnas if str(c).strip().lower() in COLUMNAS_PROHIBIDAS]
    if prohibidas:
        raise ColumnaProhibidaError(
            f"Columnas sensibles que el cargador no lee: {prohibidas} (CLAUDE.md, sección d)."
        )
    desconocidas = [c for c in columnas if c not in COLUMNAS_PERMITIDAS]
    if desconocidas:
        raise ValueError(f"Columnas que no están en COLUMNAS_PERMITIDAS: {desconocidas}")
    return columnas


def cargar_denue(
    columnas: Sequence[str],
    *,
    ruta_manifest: str | Path = RUTA_MANIFEST,
    fuente: str = FUENTE_DENUE,
    ruta_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Lee columnas del CSV del DENUE, todas como texto.

    1. Valida las columnas antes de abrir cualquier archivo.
    2. Verifica el SHA-256 del CSV contra el manifest.
    3. Lee con `dtype=str` y `keep_default_na=False`: "NA", "NULL" o "None" se
       conservan como texto y un campo vacío queda como "".
    4. Verifica que el número de filas coincida con el del manifest.

    `ruta_csv=None` usa la ruta registrada en el manifest.
    """
    columnas = _validar_columnas(columnas)
    entrada = entrada_manifest(fuente, ruta_manifest)
    ruta = Path(ruta_csv) if ruta_csv is not None else entrada["ruta"]

    sha = sha256_archivo(ruta)
    if sha != entrada["sha256"]:
        raise HashNoCoincideError(
            f"El SHA-256 de {ruta} no coincide con el manifest: "
            f"calculado {sha}, registrado {entrada['sha256']}."
        )

    df = pd.read_csv(
        ruta,
        usecols=columnas,
        dtype=str,
        keep_default_na=False,
        encoding=entrada["codificacion"],
    )
    if len(df) != entrada["filas"]:
        raise ValueError(
            f"Se leyeron {len(df)} filas y el manifest registra {entrada['filas']}."
        )
    return df[columnas]
