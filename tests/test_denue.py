"""Pruebas del cargador único del DENUE.

Solo usan un CSV sintético que crea cada prueba en una carpeta temporal; nunca
leen el CSV real ni el `data/manifest.json` del repo.
"""

import json

import pandas as pd
import pytest

from src.denue import (
    COLUMNAS_PERMITIDAS,
    COLUMNAS_PROHIBIDAS,
    ColumnaProhibidaError,
    HashNoCoincideError,
    cargar_denue,
    sha256_archivo,
)

ENCABEZADO = '"id","clee","nom_estab","codigo_act","telefono"'
# Valores inventados. La última fila deja nom_estab vacío sin comillas.
FILAS = [
    '"0000000001","SINTETICO01","NA","461110","SINTETICO"',
    '"0000000002","SINTETICO02","NULL","461110","SINTETICO"',
    '"0000000003","SINTETICO03","None","722511","SINTETICO"',
    '"0000000004","SINTETICO04","N/A","722511","SINTETICO"',
    '"0000000005","SINTETICO05","","812110","SINTETICO"',
    '"0000000006","SINTETICO06","CAFÉ ÑANDÚ","722515","SINTETICO"',
    '"0000000007","SINTETICO07",,"812110","SINTETICO"',
]
NOMBRES_ESPERADOS = ["NA", "NULL", "None", "N/A", "", "CAFÉ ÑANDÚ", ""]


def _escribir_manifest(ruta_manifest, ruta_csv, *, sha256=None, filas=None):
    entrada = {
        "ruta": str(ruta_csv),  # absoluta: se usa tal cual
        "sha256": sha256 if sha256 is not None else sha256_archivo(ruta_csv),
        "filas": filas if filas is not None else len(FILAS),
        "codificacion": "latin-1",
    }
    contenido = {"version": 1, "fuentes": {"denue_cdmx": entrada}, "splits": {}}
    ruta_manifest.write_text(json.dumps(contenido), encoding="utf-8")


@pytest.fixture
def fuente(tmp_path):
    """CSV sintético en latin-1 (con LF, como el real) y su manifest."""
    csv = tmp_path / "denue_sintetico.csv"
    csv.write_bytes(("\n".join([ENCABEZADO, *FILAS]) + "\n").encode("latin-1"))
    manifest = tmp_path / "manifest.json"
    _escribir_manifest(manifest, csv)
    return csv, manifest


# --- Listas de columnas ---


def test_listas_de_columnas_no_se_cruzan():
    assert len(COLUMNAS_PERMITIDAS) == 38
    assert len(set(COLUMNAS_PERMITIDAS)) == 38
    assert not COLUMNAS_PROHIBIDAS & {c.lower() for c in COLUMNAS_PERMITIDAS}


# --- Columnas prohibidas y desconocidas ---


@pytest.mark.parametrize(
    "columna", ["telefono", "correoelec", "www", "raz_social", "TELEFONO", " Www "]
)
def test_columna_prohibida_lanza_error(fuente, columna):
    _, manifest = fuente
    with pytest.raises(ColumnaProhibidaError):
        cargar_denue([columna], ruta_manifest=manifest)


def test_prohibida_junto_a_permitidas_lanza_error(fuente):
    _, manifest = fuente
    with pytest.raises(ColumnaProhibidaError):
        cargar_denue(["nom_estab", "codigo_act", "telefono"], ruta_manifest=manifest)


def test_prohibida_se_rechaza_antes_de_abrir_archivos(tmp_path):
    no_existe = tmp_path / "no_existe.json"
    with pytest.raises(ColumnaProhibidaError):
        cargar_denue(["nom_estab", "correoelec"], ruta_manifest=no_existe)


def test_columna_desconocida_lanza_error(fuente):
    _, manifest = fuente
    with pytest.raises(ValueError):
        cargar_denue(["nom_estab", "columna_que_no_existe"], ruta_manifest=manifest)


@pytest.mark.parametrize("columnas", [[], "nom_estab", ["nom_estab", "nom_estab"]])
def test_lista_invalida_lanza_error(fuente, columnas):
    _, manifest = fuente
    with pytest.raises(ValueError):
        cargar_denue(columnas, ruta_manifest=manifest)


# --- Textos tipo "NA" ---


def test_textos_tipo_na_se_preservan(fuente):
    _, manifest = fuente
    df = cargar_denue(["nom_estab", "codigo_act"], ruta_manifest=manifest)
    assert df["nom_estab"].tolist() == NOMBRES_ESPERADOS
    assert int(df.isna().sum().sum()) == 0


# --- Hash ---


def test_hash_falso_en_manifest_lanza_error(fuente):
    csv, manifest = fuente
    _escribir_manifest(manifest, csv, sha256="0" * 64)
    with pytest.raises(HashNoCoincideError):
        cargar_denue(["nom_estab"], ruta_manifest=manifest)


def test_csv_modificado_despues_de_registrar_lanza_error(fuente):
    csv, manifest = fuente
    datos = bytearray(csv.read_bytes())
    i = datos.index(b"SINTETICO01")
    datos[i] = ord("X")  # cambia un solo byte
    csv.write_bytes(bytes(datos))
    with pytest.raises(HashNoCoincideError):
        cargar_denue(["nom_estab"], ruta_manifest=manifest)


def test_ruta_csv_explicita_tambien_verifica_hash(fuente, tmp_path):
    csv, manifest = fuente
    otro = tmp_path / "otro.csv"
    otro.write_bytes(csv.read_bytes().replace(b"461110", b"461111"))
    with pytest.raises(HashNoCoincideError):
        cargar_denue(["nom_estab"], ruta_manifest=manifest, ruta_csv=otro)


# --- Filas ---


def test_filas_distintas_al_manifest_lanza_error(fuente):
    csv, manifest = fuente
    _escribir_manifest(manifest, csv, filas=len(FILAS) + 1)
    with pytest.raises(ValueError, match="filas"):
        cargar_denue(["nom_estab"], ruta_manifest=manifest)


# --- Carga correcta ---


def test_carga_correcta(fuente):
    _, manifest = fuente
    df = cargar_denue(["codigo_act", "id", "nom_estab"], ruta_manifest=manifest)
    assert list(df.columns) == ["codigo_act", "id", "nom_estab"]  # orden pedido
    assert len(df) == len(FILAS)
    assert df["id"].iloc[0] == "0000000001"  # ceros a la izquierda
    assert df["nom_estab"].iloc[5] == "CAFÉ ÑANDÚ"  # acentos de latin-1
    assert df["codigo_act"].tolist()[:2] == ["461110", "461110"]
    for columna in df.columns:
        assert pd.api.types.is_string_dtype(df[columna])
