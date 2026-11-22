"""Pruebas del tope por clase y del train final (Fase 1, paso 8).

Solo usan nombres, llaves, ids y códigos inventados, en memoria o en carpetas temporales;
nunca leen el DENUE ni los splits reales.
"""

import builtins
import hashlib
import io
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import src.hard_set as hs
import src.tope_clase as tc
from src.exploracion_clasificador import textos
from src.hard_set import ErrorHardSet, candado_hashes
from src.muestra_humo import ErrorMuestra
from src.splits import CAMPOS
from src.splits import serializar as serializar_split
from src.tope_clase import (
    ENTRADA_FINAL,
    LIMITE_SESION_KAGGLE_H,
    POR_DECIDIR,
    RUTA_FINAL,
    SIN_TOPE,
    ErrorTope,
    agregar_entrada,
    aplicar_tope,
    armar_resultado,
    cargar,
    contexto,
    distancia_variacion_total,
    entrada_final,
    generar,
    horas_por_epoca,
    leer_tasas,
    orden_topes,
    por_clase,
    punto_de_control,
    resumen_tope,
    tope_aprobado,
    validar_tope,
)

# Clases inventadas con tamaños distintos: dos sobre el tope 10, dos bajo él.
TAMANOS = {"222220": 12, "111110": 30, "444440": 1, "333330": 5}


def _sha(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def _reg(i: int, nombre: str, codigo: str, sin_nombre: bool = False, generico: bool = False) -> dict:
    """Registro con los campos y el orden de src/splits.py."""
    return {
        "id": str(i),
        "nom_estab": nombre,
        "codigo_act": codigo,
        "llave": nombre.upper(),
        "sin_nombre": sin_nombre,
        "generico": generico,
        "igual_titulo": False,
    }


def _origen() -> list[dict]:
    """Train candidato inventado, en orden de int(id), con las clases intercaladas."""
    regs, i = [], 3
    pendientes = dict(TAMANOS)
    while any(pendientes.values()):
        for codigo in TAMANOS:
            if pendientes[codigo]:
                k = TAMANOS[codigo] - pendientes[codigo]
                # Dos filas de cada clase comparten llave, para que llaves y filas difieran.
                regs.append(_reg(i, f"Negocio inventado {codigo} {k // 2}", codigo, sin_nombre=k == 0, generico=k == 1))
                pendientes[codigo] -= 1
                i += 7
    return regs


def _val() -> list[dict]:
    codigos = ["111110"] * 5 + ["222220"] * 2 + ["333330"] * 2 + ["555550"]  # 555550: ausente de train
    return [_reg(5000 + k, f"Café inventado {k}", c) for k, c in enumerate(codigos)]


TASAS = {
    "qwen3_0.6b": {"modelo": "Qwen/Qwen3-0.6B", "lote": 32, "ejemplos_por_s": 10.0, "costo_fijo_s": 360.0},
    "qwen3_4b": {"modelo": "Qwen/Qwen3-4B", "lote": 32, "ejemplos_por_s": 0.001, "costo_fijo_s": 0.0},
}


def _resultado_humo(lote: int = 32) -> dict:
    return {
        "estimacion_horas_gpu": {
            "por_modelo": {
                "qwen3_0.6b": {"modelo": "Qwen/Qwen3-0.6B", "lote": lote, "ejemplos_por_s": 50.0,
                               "costo_fijo_s": {"instalacion": 1.0, "total": 99.0}},
                "qwen3_4b": {"modelo": "Qwen/Qwen3-4B", "lote": 32, "ejemplos_por_s": 8.0,
                             "costo_fijo_s": {"total": 150.0}},
            }
        }
    }


def _repo(tmp_path: Path, origen=None, val=None):
    """Repo de juguete con train candidato sin hard, validación, test, hard y el resultado de la prueba."""
    raiz = tmp_path / "repo"
    splits = raiz / "data" / "splits"
    splits.mkdir(parents=True)
    archivos = {
        "train_candidatos_sin_hard.jsonl": origen if origen is not None else _origen(),
        "val.jsonl": val if val is not None else _val(),
        "test.jsonl": [_reg(9000 + k, f"Prueba inventada {k}", "111110") for k in range(4)],
        "hard.jsonl": [_reg(9500 + k, f"Difícil inventado {k}", "222220") for k in range(3)],
    }
    shas = {}
    for nombre, regs in archivos.items():
        datos = serializar_split(regs)
        (splits / nombre).write_bytes(datos)
        shas[nombre] = _sha(datos)
    entradas = {
        "train_candidatos_sin_hard.jsonl": "clasificador_train_candidatos_sin_hard",
        "val.jsonl": "clasificador_validacion",
        "test.jsonl": "clasificador_test",
        "hard.jsonl": "clasificador_hard",
    }
    manifest = {
        "version": 1,
        "splits": {e: {"ruta": f"data/splits/{n}", "sha256": shas[n], "filas": len(archivos[n])}
                   for n, e in entradas.items()},
    }
    resultados = raiz / "results"
    resultados.mkdir()
    (resultados / "2026-11-21_fase2_prueba-humo.json").write_text(json.dumps(_resultado_humo()), encoding="utf-8")
    return raiz, manifest


# --- Tope ---


def test_aplicar_tope_recorta_solo_las_clases_grandes():
    origen = _origen()
    final = aplicar_tope(origen, 10)
    assert Counter(r["codigo_act"] for r in final) == {"111110": 10, "222220": 10, "333330": 5, "444440": 1}
    ids = [int(r["id"]) for r in final]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    por_id = {r["id"]: r for r in origen}
    assert all(r == por_id[r["id"]] for r in final)  # filas sin cambios
    assert all(tuple(r) == CAMPOS for r in final)


def test_el_sorteo_es_el_de_default_rng_42():
    origen = _origen()
    rng = np.random.default_rng(42)
    esperados = set()
    for codigo in sorted(TAMANOS):  # clases en orden ascendente de código
        filas = sorted((r for r in origen if r["codigo_act"] == codigo), key=lambda r: int(r["id"]))
        perm = rng.permutation(len(filas))
        esperados |= {filas[j]["id"] for j in perm[: min(len(filas), 10)]}
    assert {r["id"] for r in aplicar_tope(origen, 10)} == esperados
    assert {r["id"] for r in aplicar_tope(origen, 10, semilla=43)} != esperados


def test_no_depende_del_orden_de_entrada():
    origen = _origen()
    barajado = list(origen)
    random.Random(0).shuffle(barajado)
    assert barajado != origen
    for tope in (1, 4, 10, None):
        assert serializar_split(aplicar_tope(barajado, tope)) == serializar_split(aplicar_tope(origen, tope))


def test_los_topes_quedan_anidados_y_sin_tope_conserva_todo():
    origen = _origen()
    anterior: set[str] = set()
    for tope in (1, 2, 5, 10, 12, 30, None):
        ids = {r["id"] for r in aplicar_tope(origen, tope)}
        assert anterior <= ids
        anterior = ids
    assert aplicar_tope(origen, None) == origen
    assert aplicar_tope(origen, 30) == origen  # ninguna clase pasa de 30


@pytest.mark.parametrize("tope", [0, -1, True, 2.5, "10"])
def test_tope_invalido(tope):
    with pytest.raises(ErrorTope):
        validar_tope(tope)
    with pytest.raises(ErrorTope):
        aplicar_tope(_origen(), tope)


def test_tope_aprobado():
    with pytest.raises(ErrorTope, match="PARADA 1"):
        tope_aprobado(POR_DECIDIR)
    assert tope_aprobado(SIN_TOPE) is None
    assert tope_aprobado(200) == 200
    with pytest.raises(ErrorTope):
        tope_aprobado(0)
    assert tc.TOPE in (POR_DECIDIR, SIN_TOPE) or tope_aprobado(tc.TOPE) >= 1  # el valor del código es válido


def test_orden_topes():
    assert orden_topes([500, None, 100, 500, 200]) == [100, 200, 500, None]
    with pytest.raises(ErrorTope):
        orden_topes([0])


def test_codigo_invalido_o_ids_repetidos():
    with pytest.raises(ErrorTope):
        por_clase([_reg(1, "A", "81111")])
    with pytest.raises(ErrorTope):
        por_clase([_reg(1, "A", "٨١١١١١")])  # dígitos no ASCII
    with pytest.raises(ErrorTope):
        aplicar_tope([_reg(1, "A", "811111"), _reg(1, "B", "811112")], 5)


# --- Medidas ---


def test_distancia_variacion_total():
    assert distancia_variacion_total({"a": 3, "b": 1}, {"a": 3, "b": 1}) == 0
    assert distancia_variacion_total({"a": 2, "b": 2}, {"a": 1, "b": 1}) == 0  # proporciones iguales
    assert distancia_variacion_total({"a": 5}, {"b": 2}) == 1
    assert distancia_variacion_total({"a": 3, "b": 1}, {"a": 1, "b": 1}) == pytest.approx(0.25)
    assert distancia_variacion_total({"a": 1}, {"a": 1, "b": 3}) == pytest.approx(0.75)
    with pytest.raises(ErrorTope):
        distancia_variacion_total({}, {"a": 1})


def test_horas_por_epoca():
    tasa = {"ejemplos_por_s": 10.0, "costo_fijo_s": 360.0}
    assert horas_por_epoca(36_000, tasa) == {"entrenamiento": 1.0, "con_costo_fijo": 1.1,
                                              "pasa_del_limite_de_sesion": False}
    limite = LIMITE_SESION_KAGGLE_H * 3600 * 10  # filas que llenan justo la sesión sin el costo fijo
    assert horas_por_epoca(limite, {"ejemplos_por_s": 10.0, "costo_fijo_s": 0.0})["pasa_del_limite_de_sesion"] is False
    assert horas_por_epoca(limite, tasa)["pasa_del_limite_de_sesion"] is True  # el costo fijo la pasa


def test_leer_tasas(tmp_path):
    ruta = tmp_path / "humo.json"
    ruta.write_text(json.dumps(_resultado_humo()), encoding="utf-8")
    tasas = leer_tasas(ruta)
    assert tasas["qwen3_0.6b"] == {"modelo": "Qwen/Qwen3-0.6B", "lote": 32, "ejemplos_por_s": 50.0, "costo_fijo_s": 99.0}
    assert tasas["qwen3_4b"]["costo_fijo_s"] == 150.0
    ruta.write_text(json.dumps(_resultado_humo(lote=16)), encoding="utf-8")
    with pytest.raises(ErrorTope):
        leer_tasas(ruta)


def test_resumen_tope():
    origen, val = _origen(), _val()
    final = aplicar_tope(origen, 10)
    f = resumen_tope(final, 10, Counter(r["codigo_act"] for r in origen), Counter(r["codigo_act"] for r in val), TASAS)
    assert (f["tope"], f["filas"], f["clases"]) == (10, 26, 4)
    assert f["llaves"] == len({r["llave"] for r in final})
    assert f["clases_con_menos_de_10_filas"] == 2  # 333330 (5) y 444440 (1)
    assert f["clases_recortadas"] == 2  # 111110 y 222220
    assert f["clase_mas_grande"] == {"codigo": "111110", "filas": 10, "pct_filas": round(100 * 10 / 26, 2)}  # empate: menor código
    assert f["validacion_filas_de_clases_recortadas"] == 7
    assert f["validacion_pct_filas_de_clases_recortadas"] == 70.0
    assert f["horas_gpu_por_epoca"]["qwen3_0.6b"]["entrenamiento"] == round(26 / 10 / 3600, 2)
    assert f["horas_gpu_por_epoca"]["qwen3_4b"]["pasa_del_limite_de_sesion"] is False
    assert f["filas_por_marca"]["sin_nombre"] == sum(r["sin_nombre"] for r in final)
    natural = resumen_tope(origen, None, Counter(r["codigo_act"] for r in origen), Counter(r["codigo_act"] for r in val), TASAS)
    assert natural["tvd_contra_train_sin_tope"] == 0 and natural["clases_recortadas"] == 0
    assert natural["validacion_pct_filas_de_clases_recortadas"] == 0
    assert f["tvd_contra_train_sin_tope"] > 0


def test_punto_de_control():
    origen, val = _origen(), _val()
    tabla = punto_de_control(origen, val, [None, 10, 2, 10], TASAS)
    assert [f["tope"] for f in tabla] == [2, 10, None]
    assert [f["filas"] for f in tabla] == [7, 26, 48]
    assert all(f["clases"] == 4 for f in tabla)
    assert [f["clases_recortadas"] for f in tabla] == [3, 2, 0]
    ctx = contexto(origen, val)
    assert ctx["validacion_clases_ausentes_de_train"] == 1 and ctx["validacion_filas_de_clases_ausentes_de_train"] == 1
    assert ctx["train_candidatos_sin_hard"]["filas"] == 48


# --- Entradas ---


def test_cargar_y_generar(tmp_path):
    raiz, manifest = _repo(tmp_path)
    origen, val = cargar(manifest, raiz)
    assert origen == _origen() and val == _val()
    datos, final, _, _ = generar(manifest, 10, raiz)
    assert datos == serializar_split(aplicar_tope(_origen(), 10)) and len(final) == 26
    assert generar(manifest, 10, raiz)[0] == datos  # mismos bytes desde cero


def test_cargar_verifica_hash_filas_y_llaves(tmp_path):
    raiz, manifest = _repo(tmp_path)
    ruta = raiz / "data/splits/train_candidatos_sin_hard.jsonl"
    ruta.write_bytes(ruta.read_bytes().replace(b"111110", b"111111", 1))
    with pytest.raises(ErrorMuestra, match="SHA-256"):
        cargar(manifest, raiz)
    raiz2, manifest2 = _repo(tmp_path / "filas")
    manifest2["splits"]["clasificador_validacion"]["filas"] = 99
    with pytest.raises(ErrorTope, match="filas"):
        cargar(manifest2, raiz2)
    val = [_reg(5000, "Negocio inventado 111110 0", "111110")]  # llave que también está en train
    raiz3, manifest3 = _repo(tmp_path / "llaves", val=val)
    with pytest.raises(ErrorTope, match="llaves"):
        cargar(manifest3, raiz3)


def test_cargar_no_usa_las_rutas_de_test_ni_de_hard(tmp_path):
    raiz, manifest = _repo(tmp_path)
    for entrada, archivo in (("clasificador_test", "test.jsonl"), ("clasificador_hard", "hard.jsonl")):
        del manifest["splits"][entrada]["ruta"]
        (raiz / "data" / "splits" / archivo).unlink()
    assert len(cargar(manifest, raiz)[0]) == 48


# --- Manifest y results/ ---


def test_entrada_final_y_agregar_entrada(tmp_path):
    raiz, manifest = _repo(tmp_path)
    datos, final, _, _ = generar(manifest, 10, raiz)
    e = entrada_final(datos=datos, final=final, manifest=manifest, tope=10, commit="abc", fecha="2026-01-01")
    assert e["ruta"] == RUTA_FINAL == "data/splits/train_final.jsonl"
    assert e["sha256"] == _sha(datos) and e["tamano_bytes"] == len(datos)
    assert (e["filas"], e["clases"], e["tope"], e["semilla"], e["congelado"]) == (26, 4, 10, 42, True)
    assert e["campos"] == list(CAMPOS)
    assert e["fuentes"] == {"train_candidatos_sin_hard_sha256": manifest["splits"]["clasificador_train_candidatos_sin_hard"]["sha256"]}
    assert "default_rng(42)" in e["procedimiento"] and "Tope: 10" in e["procedimiento"]
    original = json.loads(json.dumps(manifest))
    nuevo = agregar_entrada(manifest, e)
    assert manifest == original  # no muta el original
    assert nuevo["splits"][ENTRADA_FINAL] == e
    assert {k: v for k, v in nuevo["splits"].items() if k != ENTRADA_FINAL} == original["splits"]
    with pytest.raises(ErrorTope):
        agregar_entrada(nuevo, e)
    sin_tope = entrada_final(datos=datos, final=final, manifest=manifest, tope=None, commit="abc", fecha="2026-01-01")
    assert sin_tope["tope"] is None and "Tope: sin tope" in sin_tope["procedimiento"]


def test_candado_rechaza_un_hash_del_manifest(tmp_path):
    raiz, manifest = _repo(tmp_path)
    datos, _, _, _ = generar(manifest, 10, raiz)
    candado_hashes({RUTA_FINAL: _sha(datos)}, manifest)
    for entrada in ("clasificador_test", "clasificador_hard", "clasificador_train_candidatos_sin_hard"):
        with pytest.raises(ErrorHardSet):
            candado_hashes({RUTA_FINAL: manifest["splits"][entrada]["sha256"]}, manifest)


def test_resultado_sin_nombres_llaves_ni_ids(tmp_path):
    raiz, manifest = _repo(tmp_path)
    origen, val = cargar(manifest, raiz)
    tabla = punto_de_control(origen, val, [5, 10, None], TASAS)
    datos, final, _, _ = generar(manifest, 10, raiz)
    res = armar_resultado(fecha="2026-01-01", commit="abc", arbol_limpio=True, manifest=manifest, tope=10,
                          tabla=tabla, ctx=contexto(origen, val), final=final, origen=origen,
                          hash_final=_sha(datos), tasas=TASAS)
    for campo in ("fecha", "commit", "modelo", "configuracion", "semilla", "split", "hash_split", "tamano_muestra",
                  "horas_gpu"):
        assert campo in res
    assert res["horas_gpu"] is None and res["semilla"] == 42
    assert res["train_final"]["tope"] == 10 and res["train_final"]["filas"] == 26
    assert res["train_final"]["por_clase"]["111110"] == {"antes": 30, "despues": 10}
    assert [f["tope"] for f in res["comparacion_de_topes"]] == [5, 10, None]
    prohibidos = {r[c] for r in origen + val for c in ("id", "nom_estab", "llave")}
    assert not {t for _, t in textos(res)} & prohibidos
    texto = json.dumps(res, ensure_ascii=False)
    assert "inventad" not in texto.lower()


# --- Test y hard nunca abiertos ---


def test_el_flujo_completo_nunca_abre_test_ni_hard(tmp_path, monkeypatch):
    raiz, manifest = _repo(tmp_path)
    monkeypatch.setattr(hs, "esta_ignorado", lambda r: True)
    abiertos: list[str] = []
    original = io.open

    def espia(archivo, *args, **kwargs):
        if not isinstance(archivo, int):
            abiertos.append(Path(os.fspath(archivo)).name)
        return original(archivo, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", espia)
    monkeypatch.setattr(io, "open", espia)
    tasas = leer_tasas(raiz / "results" / "2026-11-21_fase2_prueba-humo.json")
    origen, val = cargar(manifest, raiz)
    punto_de_control(origen, val, [2, 10, None], tasas)
    datos, _, _, _ = generar(manifest, 10, raiz)
    candado_hashes({RUTA_FINAL: _sha(datos)}, manifest)
    ruta = raiz / RUTA_FINAL
    hs.escribir_nuevo(datos, ruta)
    with pytest.raises(FileExistsError):
        hs.escribir_nuevo(b"otro\n", ruta)  # no se sobrescribe
    leido = ruta.read_bytes()
    monkeypatch.undo()
    assert {"train_candidatos_sin_hard.jsonl", "val.jsonl", "2026-11-21_fase2_prueba-humo.json",
            "train_final.jsonl"} <= set(abiertos)  # el espía sí ve las aperturas
    assert "test.jsonl" not in abiertos and "hard.jsonl" not in abiertos
    assert leido == datos
