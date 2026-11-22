"""Pruebas de las métricas de la regla 5 (src/metricas_clasificador.py).

Solo usan ids, respuestas y un catálogo inventados; nunca leen datos reales.
"""

import json

import pytest

from src.metricas_clasificador import (
    CATEGORIAS,
    CODIGO_EXISTENTE,
    CODIGO_INEXISTENTE,
    FORMATO_INVALIDO,
    ErrorMetricas,
    categoria,
    evaluar,
    leer_predicciones,
    limpiar,
    verificar_ids,
)

CATALOGO = frozenset({"467111", "722511", "461110"})


@pytest.mark.parametrize(
    "respuesta",
    ["467111", " 467111", "467111 ", "\n467111\n", "\r\n467111\r\n", "  \n 467111 \n  "],
)
def test_codigo_existente_con_orillas_permitidas(respuesta):
    assert categoria(respuesta, CATALOGO) == CODIGO_EXISTENTE


@pytest.mark.parametrize(
    "respuesta",
    [
        "\t467111",  # tabulador
        "467111\t",
        " 467111",  # espacio no separable
        "467111 ",
        " 467111",  # espacio Unicode (em space)
        "467111 ",  # separador de línea Unicode
        "467111\x0b",  # tabulador vertical
        "４６７１１１",  # dígitos de ancho completo
        "٤٦٧١١١",  # dígitos arábigo-índicos
        "46711",
        "4671110",
        "",
        "   ",
        "467111\n467111",
        "467111<|im_end|>",
        "4671 11",
        "Código: 467111",
        "+467111",
        "467111.",
    ],
)
def test_formato_invalido(respuesta):
    assert categoria(respuesta, CATALOGO) == FORMATO_INVALIDO


@pytest.mark.parametrize("respuesta", ["999999", " 000000\n", "467112"])
def test_codigo_inexistente(respuesta):
    assert categoria(respuesta, CATALOGO) == CODIGO_INEXISTENTE


def test_limpiar_solo_quita_espacio_ascii_cr_y_lf():
    assert limpiar(" \r\n 123 \r\n ") == "123"
    assert limpiar("\t123\t") == "\t123\t"
    assert limpiar(" 123") == " 123"
    with pytest.raises(ErrorMetricas):
        limpiar(None)


def _pred(pares):
    return "".join(json.dumps({"id": i, "respuesta": r}, ensure_ascii=False) + "\n" for i, r in pares).encode("utf-8")


def test_leer_predicciones_valido():
    datos = _pred([("1", "467111"), ("20", "texto raro"), ("3", "")])
    assert leer_predicciones(datos) == {"1": "467111", "20": "texto raro", "3": ""}


@pytest.mark.parametrize(
    "datos",
    [
        b"\xef\xbb\xbf" + _pred([("1", "467111")]),  # BOM
        _pred([("1", "467111")])[:-1],  # sin salto final
        b"",
        _pred([("1", "467111"), ("1", "722511")]),  # id repetido
        _pred([("x1", "467111")]),  # id no numérico
        _pred([("١", "467111")]),  # id con dígito no ASCII
        b'{"respuesta": "467111", "id": "1"}\n',  # orden de claves
        b'{"id": "1", "respuesta": "467111", "extra": 1}\n',
        b'{"id": 1, "respuesta": "467111"}\n',  # id numérico, no texto
        b'{"id": "1", "respuesta": 467111}\n',
        b"no es json\n",
        b"[1, 2]\n",
    ],
)
def test_leer_predicciones_rechaza(datos):
    with pytest.raises(ErrorMetricas):
        leer_predicciones(datos)


def test_verificar_ids():
    verificar_ids(["1", "2"], ["2", "1"])
    with pytest.raises(ErrorMetricas, match="faltan 1 y sobran 0"):
        verificar_ids(["1"], ["1", "2"])
    with pytest.raises(ErrorMetricas, match="faltan 0 y sobran 1"):
        verificar_ids(["1", "2", "3"], ["1", "2"])


def test_evaluar_cuenta_sobre_el_total():
    referencias = {"1": "467111", "2": "722511", "3": "461110", "4": "467111", "5": "722511", "6": "461110", "7": "467111", "8": "467111"}
    predicciones = {
        "1": "467111",  # acierto
        "2": " 722511\n",  # acierto con orillas permitidas
        "3": "467111",  # existente pero equivocado
        "4": "999999",  # inexistente: error
        "5": "\t722511",  # formato inválido: error aunque el código sea el correcto
        "6": "４６１１１０",  # formato inválido: dígitos de ancho completo
        "7": "",  # formato inválido
        "8": "467111<|im_end|>",  # formato inválido
    }
    r = evaluar(predicciones, referencias, CATALOGO)
    assert r["total"] == 8
    assert r["conteos"] == {FORMATO_INVALIDO: 4, CODIGO_INEXISTENTE: 1, CODIGO_EXISTENTE: 3}
    assert sum(r["conteos"].values()) == r["total"]
    assert tuple(r["conteos"]) == CATEGORIAS
    assert r["porcentajes"] == {FORMATO_INVALIDO: 50.0, CODIGO_INEXISTENTE: 12.5, CODIGO_EXISTENTE: 37.5}
    assert sum(r["porcentajes"].values()) == pytest.approx(100.0)
    assert r["pct_formato_valido"] == 50.0
    assert r["pct_codigos_inexistentes"] == 12.5
    assert r["aciertos"] == 2 and r["errores"] == 6
    assert r["accuracy"] == 0.25 and r["pct_accuracy"] == 25.0


def test_evaluar_todo_correcto_y_todo_invalido():
    ref = {"1": "467111", "2": "722511"}
    assert evaluar({"1": "467111", "2": "722511"}, ref, CATALOGO)["accuracy"] == 1.0
    r = evaluar({"1": "abc", "2": "12"}, ref, CATALOGO)
    assert r["accuracy"] == 0.0 and r["conteos"][FORMATO_INVALIDO] == 2 and r["pct_formato_valido"] == 0.0


def test_evaluar_redondeo():
    ref = {str(i): "467111" for i in range(3)}
    r = evaluar({"0": "467111", "1": "x", "2": "x"}, ref, CATALOGO)
    assert r["pct_accuracy"] == 33.33 and r["accuracy"] == 0.333333


@pytest.mark.parametrize(
    "predicciones, referencias",
    [
        ({"1": "467111"}, {"1": "467111", "2": "722511"}),  # falta un id
        ({"1": "467111", "3": "467111"}, {"1": "467111"}),  # sobra un id
        ({}, {}),  # vacío
    ],
)
def test_evaluar_ids_distintos_o_vacio(predicciones, referencias):
    with pytest.raises(ErrorMetricas):
        evaluar(predicciones, referencias, CATALOGO)


@pytest.mark.parametrize("referencia", ["999999", "46711", "４６７１１１", None])
def test_evaluar_referencia_fuera_del_catalogo(referencia):
    with pytest.raises(ErrorMetricas, match="referencias"):
        evaluar({"1": "467111"}, {"1": referencia}, CATALOGO)


def test_errores_no_muestran_respuestas_ni_ids():
    secreto = "NEGOCIO INVENTADO SECRETO"
    with pytest.raises(ErrorMetricas) as e:
        leer_predicciones(_pred([("777", secreto), ("777", secreto)]))
    assert secreto not in str(e.value) and "777" not in str(e.value)
    with pytest.raises(ErrorMetricas) as e:
        evaluar({"98765": secreto}, {"12345": "467111"}, CATALOGO)
    assert secreto not in str(e.value) and "98765" not in str(e.value) and "12345" not in str(e.value)
