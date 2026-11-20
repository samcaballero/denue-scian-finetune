"""Pruebas de los conteos del paso 3 (llaves y grupos por nombre normalizado).

Solo usan nombres sintéticos inventados; nunca leen el DENUE.
"""

from src.grupos_nombres import (
    es_signo,
    fusiones,
    grupos_por_tamano,
    inventario_de_caracteres,
    llaves,
    perfil_nom_estab,
    resumen_llaves,
)
from src.normalizacion import VARIANTES

FILAS = {"A.B": 1, "AB": 2, "A B": 3, "A-B": 4}


def test_fusiones_en_ambas_direcciones():
    a = llaves(FILAS, VARIANTES["A"])  # A B, AB, A B, A B
    c = llaves(FILAS, VARIANTES["C"])  # AB, AB, A B, A B
    # C junta "A B" (de A.B) con "AB": 2 llaves de A en 1 de C, con 3 filas.
    assert fusiones(a, c, FILAS) == {"llaves_de_origen": 2, "llaves_resultantes": 1, "filas": 3}
    # A junta "AB" (de A.B) con "A B": 2 llaves de C en 1 de A, con 8 filas.
    assert fusiones(c, a, FILAS) == {"llaves_de_origen": 2, "llaves_resultantes": 1, "filas": 8}


def test_fusiones_sin_cambios():
    a = llaves(FILAS, VARIANTES["A"])
    assert fusiones(a, a, FILAS) == {"llaves_de_origen": 0, "llaves_resultantes": 0, "filas": 0}


def test_resumen_llaves_con_vacia():
    filas = {"": 5, "...": 2, "X": 1, "x.": 3}
    assert resumen_llaves(llaves(filas, VARIANTES["C"]), filas) == {
        "llaves_distintas": 2,
        "filas_llave_vacia": 7,
        "nombres_distintos_llave_vacia": 2,
    }


def test_grupos_por_tamano_excluye_la_llave_vacia():
    res = grupos_por_tamano({"": 500, "X": 1, "Y": 2, "Z": 9, "W": 10, "V": 99, "U": 100, "T": 150})
    assert res == {
        "grupos": {
            "1": {"grupos": 1, "filas": 1},
            "2-9": {"grupos": 2, "filas": 11},
            "10-99": {"grupos": 2, "filas": 109},
            "100+": {"grupos": 2, "filas": 250},
        },
        "grupo_mas_grande_filas": 150,
        "filas_llave_vacia": 500,
    }


def test_inventario_de_caracteres():
    inv = inventario_de_caracteres({"PEÑA S.A.": 2, "ÑU": 1, "ABC 12": 5, "": 3})
    assert list(inv) == ["U+00D1", "U+002E"]  # de más a menos filas
    assert inv["U+00D1"]["nombres"] == 2 and inv["U+00D1"]["filas"] == 3
    assert inv["U+002E"]["nombres"] == 1 and inv["U+002E"]["filas"] == 2
    assert inv["U+00D1"]["categoria"] == "Lu"


def test_es_signo():
    assert es_signo("-") and es_signo("+") and es_signo("°") and es_signo("\x01")
    assert not (es_signo("Ñ") or es_signo("7") or es_signo(" ") or es_signo("\t"))


def test_perfil_nom_estab():
    filas = {"abc": 1, "A  B": 2, " X": 3, "Y ": 4, "Z": 5}
    assert perfil_nom_estab(filas) == {
        "con_minusculas_ascii": {"nombres": 1, "filas": 1},
        "con_espacios_dobles": {"nombres": 1, "filas": 2},
        "con_espacios_al_inicio_o_al_final": {"nombres": 2, "filas": 7},
    }
