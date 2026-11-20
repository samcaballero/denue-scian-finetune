"""Pruebas de la regla de decisión de la versión del SCIAN y de sus clasificaciones.

Solo usan conjuntos de códigos sintéticos; nunca leen el DENUE ni los catálogos reales.
"""

import pytest

from src.version_scian import (
    clasificar_codigo,
    clave_titulo,
    decidir_version,
    nombre_act_contra_titulos,
    resumen_por_categoria,
)

CAT_2023 = {"111110", "513110", "722511"}  # 513110 solo existe en 2023
CAT_2018 = {"111110", "511111", "722511"}  # 511111 solo existe en 2018


@pytest.mark.parametrize(
    "codigos, esperado",
    [
        ({"111110", "722511", "513110"}, "2023"),  # todos en 2023, uno falta en 2018
        ({"111110", "722511", "511111"}, "2018"),  # todos en 2018, uno falta en 2023
        ({"111110", "722511"}, None),  # todos existen en ambas
        ({"111110", "999999"}, None),  # uno no existe en ninguna
        ({"513110", "511111"}, None),  # ausentes en las dos
    ],
)
def test_decidir_version(codigos, esperado):
    assert decidir_version(codigos, CAT_2023, CAT_2018) == esperado


@pytest.mark.parametrize(
    "codigo, esperado",
    [("111110", "ambas"), ("513110", "solo_2023"), ("511111", "solo_2018"), ("999999", "ninguna")],
)
def test_clasificar_codigo(codigo, esperado):
    assert clasificar_codigo(codigo, CAT_2023, CAT_2018) == esperado


def test_resumen_por_categoria_suma_codigos_y_filas():
    filas = {"111110": 10, "722511": 5, "513110": 3, "511111": 2, "999999": 1, "888888": 4}
    assert resumen_por_categoria(filas, CAT_2023, CAT_2018) == {
        "solo_2018": {"codigos": 1, "filas": 2},
        "solo_2023": {"codigos": 1, "filas": 3},
        "ambas": {"codigos": 2, "filas": 15},
        "ninguna": {"codigos": 2, "filas": 5},
    }


def test_clave_titulo_laxa():
    assert clave_titulo("  Comercio  al por MENOR ") == clave_titulo("comercio al por menor")
    assert clave_titulo("Café") == clave_titulo("Café")  # NFD contra NFC
    assert clave_titulo("Café") != clave_titulo("Cafe")  # los acentos sí cuentan


def test_nombre_act_contra_titulos():
    catalogos = {
        "2023": {"111110": "Clase uno", "513110": "Clase nueva"},
        "2018": {"111110": "Clase Uno", "511111": "Clase vieja"},
    }
    pares = {
        ("111110", "Clase uno"): 4,  # exacto en 2023, laxo en 2018
        ("513110", "Clase nueva"): 2,  # exacto en 2023, inexistente en 2018
        ("511111", "Otra cosa"): 1,  # inexistente en 2023, distinto en 2018
    }
    res = nombre_act_contra_titulos(pares, catalogos)
    assert res["por_version"]["2023"] == {
        "exacto": {"pares": 2, "filas": 6},
        "solo_laxo": {"pares": 0, "filas": 0},
        "distinto": {"pares": 0, "filas": 0},
        "codigo_inexistente": {"pares": 1, "filas": 1},
    }
    assert res["por_version"]["2018"] == {
        "exacto": {"pares": 0, "filas": 0},
        "solo_laxo": {"pares": 1, "filas": 4},
        "distinto": {"pares": 1, "filas": 1},
        "codigo_inexistente": {"pares": 1, "filas": 2},
    }
    assert res["coincidencia_laxa_por_par"] == {
        "solo_2018": {"pares": 0, "filas": 0},
        "solo_2023": {"pares": 1, "filas": 2},
        "ambas": {"pares": 1, "filas": 4},
        "ninguna": {"pares": 1, "filas": 1},
    }
