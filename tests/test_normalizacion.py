"""Pruebas de la normalización de nombres (regla 2).

Solo usan texto sintético inventado; nunca leen el DENUE.
"""

import pytest

from src.normalizacion import VARIANTE, VARIANTES, _normalizar, normalizar_nombre

CASOS = [
    # Acentos, diéresis y Ñ
    ("ÁÉÍÓÚ áéíóú", "AEIOU AEIOU"),
    ("PEÑA", "PENA"),
    ("peña", "PENA"),
    ("GÜERO", "GUERO"),
    ("pingüino", "PINGUINO"),
    # NFC (acento integrado) y NFD (acento separado)
    ("José", "JOSE"),
    ("José", "JOSE"),
    # Mayúsculas, sin depender del locale
    ("Tacos el Primo", "TACOS EL PRIMO"),
    ("tienda iris", "TIENDA IRIS"),
    # Puntuación, variante C: punto y apóstrofo se borran; el resto, espacio
    ("S.A. DE C.V.", "SA DE CV"),
    ("D'LUPITA", "DLUPITA"),
    ("D’LUPITA", "DLUPITA"),
    ("TORTI-LANDIA", "TORTI LANDIA"),
    ("PAN&CAFÉ", "PAN CAFE"),
    ("PAN & CAFÉ", "PAN CAFE"),
    ("(SUC. 2)", "SUC 2"),
    ("A/B", "A B"),
    ("A_B", "A B"),
    ("A+B", "A B"),
    ("A#B", "A B"),
    ("A@B", "A B"),
    ("A,B", "A B"),
    ("«A»B", "A B"),
    ("¿QUÉ?", "QUE"),
    ("50°C", "50 C"),
    ("DR.ROJAS", "DRROJAS"),
    ("DR. ROJAS", "DR ROJAS"),
    # El acento suelto ´ no es apóstrofo: NFKD lo vuelve espacio + acento combinante
    ("D´LUPITA", "D LUPITA"),
    # Espacios múltiples, tabuladores, saltos de línea y no separables
    ("  TACOS   EL\tPRIMO \n", "TACOS EL PRIMO"),
    ("TACOS EL PRIMO", "TACOS EL PRIMO"),
    ("TACOS EL PRIMO", "TACOS EL PRIMO"),
    # º y ª (NFKD los vuelve o y a)
    ("1º", "1O"),
    ("Nº 5", "NO 5"),
    ("2ª SECCIÓN", "2A SECCION"),
    # Dígitos
    ("TACOS 24 HRS", "TACOS 24 HRS"),
    ("LOCAL 3²", "LOCAL 32"),
    # Vacío y pura puntuación
    ("", ""),
    ("   ", ""),
    ("...", ""),
    ("- - -", ""),
    ("¡¿?!", ""),
    ("'", ""),
]


@pytest.mark.parametrize("entrada, esperado", CASOS)
def test_normalizar_nombre(entrada, esperado):
    assert normalizar_nombre(entrada) == esperado


@pytest.mark.parametrize("entrada, _", CASOS)
def test_idempotente_en_los_casos(entrada, _):
    llave = normalizar_nombre(entrada)
    assert normalizar_nombre(llave) == llave


def test_idempotente_en_latin1_caracteres_y_pares():
    latin1 = [chr(i) for i in range(256)]
    for a in latin1:
        llave = normalizar_nombre(a)
        assert normalizar_nombre(llave) == llave, f"U+{ord(a):04X}"
        for b in latin1:
            llave = normalizar_nombre(a + b)
            assert normalizar_nombre(llave) == llave, f"U+{ord(a):04X} U+{ord(b):04X}"


def test_variantes_de_escritura_dan_la_misma_llave():
    variantes = ["Tacos El Güero", "TACOS EL GUERO", "tacos  el güero.", " Tacos-El-Güero ", "TACOS EL GÜERO"]
    assert {normalizar_nombre(v) for v in variantes} == {"TACOS EL GUERO"}


@pytest.mark.parametrize(
    "entrada, a, b, c",
    [
        ("S.A. DE C.V.", "S A DE C V", "SA DE CV", "SA DE CV"),
        ("TORTI-LANDIA", "TORTI LANDIA", "TORTILANDIA", "TORTI LANDIA"),
        ("PAN&CAFÉ", "PAN CAFE", "PANCAFE", "PAN CAFE"),
        ("D'LUPITA", "D LUPITA", "DLUPITA", "DLUPITA"),
        ("DR.ROJAS", "DR ROJAS", "DRROJAS", "DRROJAS"),
        ("¡¿...?!", "", "", ""),
    ],
)
def test_tres_variantes_de_puntuacion(entrada, a, b, c):
    assert _normalizar(entrada, VARIANTES["A"]) == a
    assert _normalizar(entrada, VARIANTES["B"]) == b
    assert _normalizar(entrada, VARIANTES["C"]) == c


def test_la_variante_elegida_es_c():
    assert VARIANTE == "C"
    assert normalizar_nombre("S.A.-X") == _normalizar("S.A.-X", VARIANTES["C"]) == "SA X"


@pytest.mark.parametrize("entrada", [None, float("nan"), 5, b"TACOS"])
def test_rechaza_lo_que_no_es_texto(entrada):
    with pytest.raises(TypeError):
        normalizar_nombre(entrada)
