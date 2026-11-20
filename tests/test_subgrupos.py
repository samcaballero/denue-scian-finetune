"""Pruebas de las marcas de subgrupo (regla 8) y del archivo del vocabulario.

Solo usan nombres, títulos y palabras sintéticos inventados; nunca leen el DENUE.
"""

import json

import pytest

from src.scian import escribir_manifest
from src.subgrupos import (
    CAMPOS_MARCAS,
    DEFINICIONES,
    ErrorVocabulario,
    en_vocabulario,
    es_generico,
    es_igual_a_titulo,
    escribir_vocabulario,
    leer_vocabulario,
    marcas,
    quitar_marcador,
    tiene_marcador,
    titulos_normalizados,
    vocabulario,
)

VOCAB = frozenset({"TIENDAS", "DE", "ABARROTES", "FLOR", "Y", "REPARACION"})
TITULOS = titulos_normalizados({"811111": "Reparación de prueba", "999999": "Tiendas de flor"})


# --- "SIN NOMBRE" ---


def test_marcador_solo_como_palabras_completas():
    assert tiene_marcador("TALLER PATITO SIN NOMBRE")
    assert tiene_marcador("SIN NOMBRE TALLER")
    assert tiene_marcador("A SIN NOMBRE B")
    assert not tiene_marcador("TALLER SINNOMBRE")
    assert not tiene_marcador("TALLER SIN NOMBRES")
    assert not tiene_marcador("PASIN NOMBRE")
    assert not tiene_marcador("")


def test_quitar_marcador():
    assert quitar_marcador("TALLER PATITO SIN NOMBRE") == "TALLER PATITO"
    assert quitar_marcador("A SIN NOMBRE B") == "A B"
    assert quitar_marcador("SIN NOMBRE") == ""
    assert quitar_marcador("TALLER PATITO") == "TALLER PATITO"


# --- Genérico ---


def test_vocabulario_normaliza_los_titulos():
    assert vocabulario(["Reparación de flores", "Tiendas, abarrotes"]) == frozenset(
        {"REPARACION", "DE", "FLORES", "TIENDAS", "ABARROTES"}
    )


def test_plurales_del_vocabulario():
    assert en_vocabulario("TIENDA", VOCAB)  # el vocabulario la tiene con S
    assert en_vocabulario("FLORES", VOCAB)  # la palabra trae ES
    assert en_vocabulario("ABARROTE", VOCAB)
    assert en_vocabulario("DE", VOCAB)
    assert not en_vocabulario("PATITO", VOCAB)


def test_generico():
    assert es_generico("TIENDA DE ABARROTES", VOCAB)
    assert es_generico("ABARROTES SIN NOMBRE", VOCAB)  # la frase no cuenta
    assert not es_generico("ABARROTES PATITO", VOCAB)
    assert not es_generico("DE Y", VOCAB)  # sin palabras de 4 letras o más
    assert not es_generico("SIN NOMBRE", VOCAB)  # sin palabras tras quitar la frase
    assert not es_generico("", VOCAB)


# --- Igual a título ---


def test_igual_a_titulo_por_el_catalogo():
    assert es_igual_a_titulo("REPARACION DE PRUEBA", "Otra cosa", TITULOS)
    assert es_igual_a_titulo("TIENDAS DE FLOR", "", TITULOS)
    assert not es_igual_a_titulo("REPARACION DE PRUEBA SIN NOMBRE", "Otra cosa", TITULOS)


def test_igual_a_titulo_por_nombre_act_normalizado():
    # nombre_act se normaliza antes de comparar: acentos, mayúsculas y punto final.
    assert es_igual_a_titulo("CAFE DE PRUEBA", "Café de prueba.", TITULOS)
    assert not es_igual_a_titulo("CAFE DE PRUEBA", "Café de otra prueba", TITULOS)


def test_llave_vacia_sin_ninguna_marca():
    # Aunque nombre_act normalizado también sea vacío.
    assert marcas("", "", VOCAB, TITULOS) == {"sin_nombre": False, "generico": False, "igual_titulo": False}
    assert marcas("", "...", VOCAB, TITULOS)["igual_titulo"] is False


def test_marcas_claves_en_orden():
    m = marcas("TIENDAS DE FLOR SIN NOMBRE", "Algo", VOCAB, TITULOS)
    assert tuple(m) == CAMPOS_MARCAS == ("sin_nombre", "generico", "igual_titulo")
    assert m == {"sin_nombre": True, "generico": True, "igual_titulo": False}
    assert all(isinstance(v, bool) for v in m.values())
    assert set(DEFINICIONES) == set(CAMPOS_MARCAS)


# --- Archivo del vocabulario ---


def test_vocabulario_ida_y_vuelta(tmp_path):
    ruta = tmp_path / "vocab.csv"
    escribir_vocabulario({"FLOR", "ABARROTES", "DE"}, ruta)
    datos = ruta.read_bytes()
    assert datos == b"palabra\nABARROTES\nDE\nFLOR\n"  # sin BOM, ordenado, con \n
    assert leer_vocabulario(ruta) == frozenset({"FLOR", "ABARROTES", "DE"})


@pytest.mark.parametrize(
    "contenido",
    [
        b"palabras\nDE\n",  # encabezado distinto
        b"palabra\nFLOR\nDE\n",  # fuera de orden
        b"palabra\nDE\nDE\n",  # repetida
        b"palabra\nDE,FLOR\n",  # dos columnas
        b"palabra\n\"DE FLOR\"\n",  # con espacio
    ],
)
def test_leer_vocabulario_rechaza_errores(tmp_path, contenido):
    ruta = tmp_path / "vocab.csv"
    ruta.write_bytes(contenido)
    with pytest.raises(ErrorVocabulario):
        leer_vocabulario(ruta)


def test_escribir_vocabulario_rechaza_palabras_con_espacio(tmp_path):
    with pytest.raises(ErrorVocabulario):
        escribir_vocabulario({"DE FLOR"}, tmp_path / "vocab.csv")


# --- Manifest ---


def test_escribir_manifest_formato(tmp_path):
    ruta = tmp_path / "manifest.json"
    manifest = {"version": 1, "nota": "contraseña", "splits": {}}
    escribir_manifest(manifest, ruta)
    datos = ruta.read_bytes()
    assert datos == (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    assert "contraseña".encode("utf-8") in datos and b"\r" not in datos
    assert not (tmp_path / "manifest.json.tmp").exists()
