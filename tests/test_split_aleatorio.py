"""Pruebas del split aleatorio del experimento de fuga (Fase 1, paso 10).

Solo usan nombres, llaves, ids y códigos inventados, en memoria o en carpetas temporales;
nunca leen el DENUE ni los splits reales.
"""

import builtins
import hashlib
import io
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from src.hard_set import ErrorHardSet, agregar_entradas, candado_hashes
from src.normalizacion import normalizar_nombre
from src.split_aleatorio import (
    ENTRADA_A_TRAIN,
    ENTRADA_A_VAL,
    N_VAL,
    RUTA_A_TRAIN,
    RUTA_A_VAL,
    ErrorSplitAleatorio,
    SplitAleatorio,
    conteos,
    entradas_manifest,
    generar,
    partir,
    unir,
    verificar,
)
from src.splits import CAMPOS
from src.splits import serializar as serializar_split
from src.tope_clase import aplicar_tope

CLASES = ("111110", "222220", "333330", "444440")


def _reg(i: int, nombre: str, codigo: str, sin_nombre: bool = False, generico: bool = False) -> dict:
    return {"id": str(i), "nom_estab": nombre, "codigo_act": codigo, "llave": normalizar_nombre(nombre),
            "sin_nombre": sin_nombre, "generico": generico, "igual_titulo": False}


def _origen(n: int = 60) -> list[dict]:
    # Cada llave tiene 3 filas: el split aleatorio puede dejar llaves de validación en train.
    return [_reg(10 + 3 * k, f"Negocio inventado {k // 3}", CLASES[k % 4], sin_nombre=k % 11 == 0) for k in range(n)]


def _val(n: int = 8) -> list[dict]:
    return [_reg(11 + 3 * k, f"Café inventado {k}", CLASES[k % 2]) for k in range(n)]


def _sha(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


# --- Construcción ---


def test_unir_ordena_por_id_y_rechaza_id_compartidos():
    p = unir(_origen(), _val())
    ids = [int(r["id"]) for r in p]
    assert ids == sorted(ids) and len(p) == 68
    with pytest.raises(ErrorSplitAleatorio, match="comparten id"):
        unir(_origen(), [_origen()[0]])


def test_partir_sigue_el_procedimiento():
    p = unir(_origen(), _val())
    s = partir(p, tope=5, n_val=10)
    perm = np.random.default_rng(42).permutation(len(p))
    assert [r["id"] for r in s.val] == [p[int(i)]["id"] for i in np.sort(perm[:10])]
    assert [r["id"] for r in s.piso] == [p[int(i)]["id"] for i in np.sort(perm[10:20])]
    ids_val = {r["id"] for r in s.val}
    assert [r["id"] for r in s.resto] == [r["id"] for r in p if r["id"] not in ids_val]
    assert s.train == aplicar_tope(s.resto, 5)
    verificar(s)
    assert all(tuple(r) == CAMPOS for r in s.val + s.train)


def test_el_sorteo_no_mira_las_llaves():
    p = unir(_origen(), _val())
    otra = [{**r, "llave": f"OTRA {i}"} for i, r in enumerate(p)]
    assert [r["id"] for r in partir(p, 5, n_val=10).val] == [r["id"] for r in partir(otra, 5, n_val=10).val]


def test_partir_rechaza_p_desordenado_o_chico():
    p = unir(_origen(), _val())
    with pytest.raises(ErrorSplitAleatorio, match="orden"):
        partir(p[::-1], 5, n_val=10)
    with pytest.raises(ErrorSplitAleatorio, match="se piden"):
        partir(p[:15], 5, n_val=10)


def test_verificar_detecta_un_train_fuera_del_resto():
    s = partir(unir(_origen(), _val()), 5, n_val=10)
    malo = SplitAleatorio(p=s.p, val=s.val, resto=s.resto, train=s.train + [s.val[0]], piso=s.piso)
    with pytest.raises(ErrorSplitAleatorio, match="fuera del resto"):
        verificar(malo)


def test_conteos_sin_nombres_ni_llaves():
    origen, val = _origen(), _val()
    s = partir(unir(origen, val), 5, n_val=10)
    manifest = {"splits": {"clasificador_train_candidatos": {"filas": 70},
                           "clasificador_train_candidatos_sin_hard": {"filas": 60}}}
    cnt = conteos(s, val, manifest, 7)
    assert cnt["p"]["filas"] == 68 and cnt["validacion_aleatoria"]["filas"] == 10
    assert cnt["validacion_por_nombre_con_llave_en_train"] == 0
    llaves_train = {r["llave"] for r in s.train}
    assert cnt["validacion_aleatoria_con_llave_en_train_aleatorio"]["filas"] == sum(r["llave"] in llaves_train for r in s.val)
    assert cnt["p_frente_a_la_poblacion_de_la_validacion_por_nombre"]["filas_de_las_llaves_del_hard_set"] == 10
    assert cnt["filas_por_marca"]["validacion_aleatoria"]["sin_nombre"] == sum(r["sin_nombre"] for r in s.val)
    texto = repr(cnt)
    assert not any(r["llave"] in texto or r["nom_estab"] in texto for r in s.p)


# --- Repo de juguete: generar, manifest y test/hard nunca abiertos ---


def _repo(tmp_path: Path):
    raiz = tmp_path / "repo"
    splits = raiz / "data" / "splits"
    splits.mkdir(parents=True)
    origen = _origen(2 * N_VAL + 40)
    archivos = {
        "train_candidatos_sin_hard.jsonl": origen,
        "val.jsonl": _val(),
        "test.jsonl": [_reg(9_000_000 + k, f"Prueba inventada {k}", "111110") for k in range(4)],
        "hard.jsonl": [_reg(9_500_000 + k, f"Difícil inventado {k}", "222220") for k in range(3)],
    }
    entradas = {"train_candidatos_sin_hard.jsonl": "clasificador_train_candidatos_sin_hard",
                "val.jsonl": "clasificador_validacion", "test.jsonl": "clasificador_test", "hard.jsonl": "clasificador_hard"}
    manifest = {"version": 1, "otra": {"x": 1}, "splits": {}}
    for nombre, regs in archivos.items():
        datos = serializar_split(regs)
        (splits / nombre).write_bytes(datos)
        manifest["splits"][entradas[nombre]] = {"ruta": f"data/splits/{nombre}", "sha256": _sha(datos), "filas": len(regs)}
    manifest["splits"]["clasificador_train_candidatos"] = {"filas": len(origen) + 5}
    return raiz, manifest


def test_generar_da_los_mismos_bytes_y_nunca_abre_test_ni_hard(tmp_path, monkeypatch):
    raiz, manifest = _repo(tmp_path)
    abiertos: list[str] = []
    original = io.open

    def espia(archivo, *args, **kwargs):
        if not isinstance(archivo, int):
            abiertos.append(Path(os.fspath(archivo)).name)
        return original(archivo, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", espia)
    monkeypatch.setattr(io, "open", espia)
    s, val_nombre, datos = generar(manifest, raiz)
    _, _, datos2 = generar(manifest, raiz)
    cnt = conteos(s, val_nombre, manifest, None)
    monkeypatch.undo()
    assert datos == datos2
    assert set(datos) == {RUTA_A_VAL, RUTA_A_TRAIN}
    assert len(s.val) == N_VAL and cnt["validacion_aleatoria"]["filas"] == N_VAL
    assert {"train_candidatos_sin_hard.jsonl", "val.jsonl"} <= set(abiertos)  # el espía sí ve las aperturas
    assert "test.jsonl" not in abiertos and "hard.jsonl" not in abiertos


def test_generar_no_usa_las_rutas_de_test_ni_de_hard(tmp_path):
    raiz, manifest = _repo(tmp_path)
    for entrada, archivo in (("clasificador_test", "test.jsonl"), ("clasificador_hard", "hard.jsonl")):
        del manifest["splits"][entrada]["ruta"]
        (raiz / "data" / "splits" / archivo).unlink()
    assert len(generar(manifest, raiz)[0].val) == N_VAL


def test_entradas_del_manifest_y_candados(tmp_path):
    raiz, manifest = _repo(tmp_path)
    s, _, datos = generar(manifest, raiz)
    entradas = entradas_manifest(datos=datos, s=s, manifest=manifest, tope=1000, commit="abc", fecha="2026-11-22")
    assert set(entradas) == {ENTRADA_A_VAL, ENTRADA_A_TRAIN}
    for e, ruta in ((ENTRADA_A_VAL, RUTA_A_VAL), (ENTRADA_A_TRAIN, RUTA_A_TRAIN)):
        assert entradas[e]["sha256"] == _sha(datos[ruta]) and entradas[e]["solo_experimento"] is True
    nuevo = agregar_entradas(manifest, entradas)
    assert {k: v for k, v in nuevo["splits"].items() if k not in entradas} == manifest["splits"]
    assert nuevo["otra"] == manifest["otra"]
    candado_hashes({r: _sha(d) for r, d in datos.items()}, manifest)
    with pytest.raises(ErrorHardSet, match="SHA-256"):
        candado_hashes({"x": manifest["splits"]["clasificador_test"]["sha256"]}, manifest)


def test_tvd_y_piso_de_ruido():
    origen, val = _origen(), _val()
    s = partir(unir(origen, val), 5, n_val=10)
    manifest = {"splits": {"clasificador_train_candidatos": {"filas": 60},
                           "clasificador_train_candidatos_sin_hard": {"filas": 60}}}
    t = conteos(s, val, manifest, None)["tvd_clases"]
    a = Counter(r["codigo_act"] for r in s.val)
    b = Counter(r["codigo_act"] for r in s.piso)
    esperado = sum(abs(a[c] / 10 - b[c] / 10) for c in set(a) | set(b)) / 2
    assert t["piso_de_ruido_validacion_aleatoria_contra_perm_1000_2000"] == pytest.approx(esperado, abs=1e-4)
