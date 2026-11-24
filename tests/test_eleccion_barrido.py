"""Pruebas de la regla de elección del barrido (src/eleccion_barrido.py), con resultados inventados."""

import json
from pathlib import Path

import pytest

from src import eleccion_barrido as eb

RAIZ = Path(__file__).resolve().parents[1]


def _c(nombre, aciertos, rank=16, lr=2e-4, inicial=False, pasos=5855, filas=1000):
    return {
        "nombre": nombre,
        "es_inicial": inicial,
        "rank": rank,
        "learning_rate": lr,
        "pasos_hechos": pasos,
        "filas_inferidas": filas,
        "aciertos_clase": aciertos,
        "n": filas,
    }


def _cinco(inicial=678, a=670, b=690, c=665, d=680):
    return [
        _c("06b-inicial", inicial, 16, 2e-4, inicial=True),
        _c("06b-lr1e-4-r8", a, 8, 1e-4),
        _c("06b-lr1e-4-r32", b, 32, 1e-4),
        _c("06b-lr2e-4-r8", c, 8, 2e-4),
        _c("06b-lr2e-4-r32", d, 32, 2e-4),
    ]


def test_gana_una():
    r = eb.elegir(_cinco())
    assert r["ganadora"] == "06b-lr1e-4-r32"
    assert r["clausula"] == "mayor_accuracy"
    assert r["empatadas"] == ["06b-lr1e-4-r32"]
    assert r["accuracy_maxima"] == 0.69
    assert [o["nombre"] for o in r["orden"]] == [
        "06b-lr1e-4-r32", "06b-lr2e-4-r32", "06b-inicial", "06b-lr1e-4-r8", "06b-lr2e-4-r8",
    ]
    assert r["excluidas_incompletas"] == []


def test_gana_la_inicial_si_es_la_mejor():
    r = eb.elegir(_cinco(inicial=700))
    assert (r["ganadora"], r["clausula"], r["ganadora_es_la_inicial"]) == ("06b-inicial", "mayor_accuracy", True)


def test_empate_con_la_inicial_gana_la_inicial():
    r = eb.elegir(_cinco(inicial=690, b=690))
    assert r["ganadora"] == "06b-inicial"
    assert r["clausula"] == "empate_con_la_inicial"
    assert r["empatadas"] == ["06b-inicial", "06b-lr1e-4-r32"]
    assert r["orden"][0]["nombre"] == "06b-inicial"


def test_empate_sin_la_inicial_gana_el_menor_rank():
    r = eb.elegir(_cinco(a=690, b=690))
    assert r["ganadora"] == "06b-lr1e-4-r8"
    assert r["clausula"] == "empate_por_rank"
    assert r["empatadas"] == ["06b-lr1e-4-r32", "06b-lr1e-4-r8"]


def test_empate_de_rank_lo_resuelve_la_tasa():
    r = eb.elegir(_cinco(a=690, b=650, c=690))
    assert r["ganadora"] == "06b-lr2e-4-r8"
    assert r["clausula"] == "empate_por_lr"
    # En el orden, entre las empatadas de rank 8 va primero la de lr 2e-4.
    assert [o["nombre"] for o in r["orden"][:2]] == ["06b-lr2e-4-r8", "06b-lr1e-4-r8"]


def test_empate_de_las_cuatro_del_barrido():
    r = eb.elegir(_cinco(inicial=600, a=690, b=690, c=690, d=690))
    assert (r["ganadora"], r["clausula"]) == ("06b-lr2e-4-r8", "empate_por_lr")


@pytest.mark.parametrize("incompleta", [{"pasos": 3000}, {"filas": 0}, {"filas": 999}])
def test_configuracion_incompleta_queda_excluida_aunque_tenga_mas_aciertos(incompleta):
    cands = _cinco()
    pasos, filas = incompleta.get("pasos", 5855), incompleta.get("filas", 1000)
    cands[2] = {**cands[2], "pasos_hechos": pasos, "filas_inferidas": filas, "n": filas, "aciertos_clase": min(999, filas)}
    r = eb.elegir(cands)
    assert r["ganadora"] == "06b-lr2e-4-r32"
    assert r["excluidas_incompletas"] == ["06b-lr1e-4-r32"]
    assert r["orden"][-1] == {
        "nombre": "06b-lr1e-4-r32", "es_inicial": False, "rank": 32, "learning_rate": 1e-4, "completa": False,
        "pasos_hechos": pasos, "filas_inferidas": filas,
    }


def test_incompleta_sin_aciertos():
    cands = _cinco()
    cands[4] = {**cands[4], "pasos_hechos": 1200, "filas_inferidas": None, "aciertos_clase": None}
    r = eb.elegir(cands)
    assert r["ganadora"] == "06b-lr1e-4-r32"
    assert r["excluidas_incompletas"] == ["06b-lr2e-4-r32"]


def test_errores():
    with pytest.raises(eb.ErrorEleccion, match="No hay"):
        eb.elegir([])
    with pytest.raises(eb.ErrorEleccion, match="Ninguna"):
        eb.elegir([_c("06b-inicial", None, inicial=True, pasos=10)])
    with pytest.raises(eb.ErrorEleccion, match="exactamente una configuración inicial; hay 0"):
        eb.elegir([_c("a", 1), _c("b", 2)])
    with pytest.raises(eb.ErrorEleccion, match="hay 2"):
        eb.elegir([_c("a", 1, inicial=True), _c("b", 2, inicial=True)])
    with pytest.raises(eb.ErrorEleccion, match="repetidos"):
        eb.elegir([_c("a", 1, inicial=True), _c("a", 2)])
    with pytest.raises(eb.ErrorEleccion, match="claves"):
        eb.elegir([{**_c("a", 1, inicial=True), "extra": 1}])
    with pytest.raises(eb.ErrorEleccion, match="aciertos_clase"):
        eb.elegir([_c("a", 1001, inicial=True)])
    with pytest.raises(eb.ErrorEleccion, match="aciertos_clase"):
        eb.elegir([_c("a", 0.5, inicial=True)])
    with pytest.raises(eb.ErrorEleccion, match="sin resolver"):
        eb.elegir([_c("i", 1, inicial=True), _c("a", 5, 8, 1e-4), _c("b", 5, 8, 1e-4)])


def test_candidata_desde_el_resultado_de_06b_inicial():
    res = json.loads((RAIZ / "results" / "2026-11-23_fase2_06b-inicial.json").read_bytes())
    c = eb.candidata_desde_resultado(res, es_inicial=True)
    assert c == {
        "nombre": "06b-inicial", "es_inicial": True, "rank": 16, "learning_rate": 0.0002, "pasos_hechos": 5855,
        "filas_inferidas": 1000, "aciertos_clase": 678, "n": 1000,
    }
    assert eb.completa(c)
