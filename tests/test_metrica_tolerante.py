"""Pruebas de la métrica tolerante (src/metrica_tolerante.py), con datos inventados."""

import pytest

from src.metrica_tolerante import aciertos_tolerantes, categoria_tolerante, evaluar_tolerante, extraer
from src.metricas_clasificador import ErrorMetricas

CODIGOS = frozenset({"461110", "722511", "311110"})


@pytest.mark.parametrize(
    ("respuesta", "esperado"),
    [
        ("461110", "461110"),
        ("  461110\n", "461110"),
        ("Clase 461110.", "461110"),
        ("El código es 461110 (comercio)", "461110"),
        ("4611101", None),  # 7 dígitos: no hay 6 aislados
        ("12345", None),
        ("461110 o 722511", "461110"),  # la primera
        ("1234567 y 722511", "722511"),  # la primera de exactamente 6
        ("٣461110", "461110"),  # dígito arábigo-índico al lado: cuenta como no-dígito
        ("461110٣", "461110"),
        ("４６１１１０", None),  # dígitos de ancho completo: no son ASCII
        ("sin código", None),
        ("", None),
        ("<think>\n\n</think>\n\n461110", "461110"),
    ],
)
def test_extraer(respuesta, esperado):
    assert extraer(respuesta) == esperado


def test_extraer_rechaza_no_texto():
    with pytest.raises(ErrorMetricas):
        extraer(461110)


def test_categorias():
    assert categoria_tolerante("Clase 461110", CODIGOS) == "codigo_existente"
    assert categoria_tolerante("Clase 999999", CODIGOS) == "codigo_inexistente"
    assert categoria_tolerante("nada", CODIGOS) == "formato_invalido"


def _filas():
    return [
        {"id": "30", "codigo_act": "461110"},
        {"id": "4", "codigo_act": "722511"},
        {"id": "100", "codigo_act": "311110"},
        {"id": "7", "codigo_act": "461110"},
    ]


def test_evaluar_tolerante_cuenta_y_compara_con_la_regla5():
    pred = {"30": "461110", "4": "Respuesta: 722511", "100": "999999", "7": "no sé"}
    r = evaluar_tolerante(pred, _filas(), CODIGOS)
    assert r["total"] == 4
    assert r["conteos"] == {"formato_invalido": 1, "codigo_inexistente": 1, "codigo_existente": 2}
    assert sum(r["conteos"].values()) == r["total"]
    assert r["aciertos"] == 2 and r["accuracy"] == 0.5
    assert r["aciertos_regla5"] == 1
    assert r["filas_aciertan_solo_con_la_tolerante"] == 1
    assert r["ic95"] is not None
    # Solo números y categorías: ningún texto de respuesta en la salida.
    assert "Respuesta" not in repr(r) and "no sé" not in repr(r)


def test_aciertos_en_el_orden_de_las_filas():
    pred = {"30": "461110", "4": "x", "100": "311110", "7": "461110 461110"}
    assert aciertos_tolerantes(pred, _filas(), CODIGOS) == [True, False, True, True]


def test_ids_distintos_fallan():
    with pytest.raises(ErrorMetricas):
        evaluar_tolerante({"30": "461110"}, _filas(), CODIGOS)
