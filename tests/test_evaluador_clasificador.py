"""Pruebas del evaluador de validación (src/evaluador_clasificador.py).

Solo usan ids, códigos, respuestas y catálogos inventados; nunca leen datos reales.
"""

import builtins
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.evaluador_clasificador import (
    SECTORES_2023,
    ErrorEvaluador,
    aciertos_por_nivel,
    cargar_codigos,
    cargar_referencias,
    diferencia_bootstrap,
    diferencia_pareada_bootstrap,
    en_nivel,
    evaluar_validacion,
    intervalo_bootstrap,
    macro_f1,
    validar_referencias,
    verificar_sectores,
)
from src.splits import serializar

CATALOGO = frozenset({"111110", "311811", "322110", "331111", "431110", "461110", "481111", "491110", "722511"})


def _fila(i: int, codigo: str, sin_nombre: bool = False, generico: bool = False) -> dict:
    return {"id": str(i), "codigo_act": codigo, "sin_nombre": sin_nombre, "generico": generico}


# --- Niveles ---


@pytest.mark.parametrize(
    "codigo, sector",
    [("111110", "11"), ("311811", "31-33"), ("322110", "31-33"), ("331111", "31-33"), ("431110", "43"),
     ("461110", "46"), ("481111", "48-49"), ("491110", "48-49"), ("722511", "72")],
)
def test_sector_oficial(codigo, sector):
    assert en_nivel(codigo, 2) == sector


def test_prefijos_de_los_demas_niveles():
    assert [en_nivel("322110", n) for n in (3, 4, 5, 6)] == ["322", "3221", "32211", "322110"]


def test_sectores_2023_sin_prefijos_repetidos():
    assert len(SECTORES_2023) == 20
    verificar_sectores(CATALOGO)


def test_codigo_fuera_de_todo_sector_se_rechaza():
    with pytest.raises(ErrorEvaluador, match="ningún sector"):
        verificar_sectores(CATALOGO | {"991110"})


def test_exigir_todos_los_sectores():
    with pytest.raises(ErrorEvaluador, match="Sectores sin clases"):
        verificar_sectores(CATALOGO, exigir_todos=True)


def test_dos_sectores_con_el_mismo_prefijo_se_rechazan():
    with pytest.raises(ErrorEvaluador, match="comparten"):
        verificar_sectores(CATALOGO, sectores=("31-33", "33"))


def test_acierto_parcial_por_nivel():
    filas = [_fila(1, "311811"), _fila(2, "481111"), _fila(3, "431110"), _fila(4, "322110")]
    pred = {"1": "322110", "2": "491110", "3": "461110", "4": "322110"}
    res = aciertos_por_nivel(pred, filas, CATALOGO)
    assert res["sector"] == [True, True, False, True]  # 31-33 y 48-49 agrupados; 43 y 46 separados
    assert res["subsector"] == [False, False, False, True]
    assert res["clase"] == [False, False, False, True]


def test_invalido_e_inexistente_son_error_en_todos_los_niveles():
    filas = [_fila(1, "311811"), _fila(2, "311811"), _fila(3, "311811")]
    # 311899 tiene formato válido y la misma rama, pero no está en el catálogo
    pred = {"1": "311899", "2": "\t311811", "3": "3118"}
    res = aciertos_por_nivel(pred, filas, CATALOGO)
    assert all(v == [False, False, False] for v in res.values())


# --- Macro-F1 ---


def test_macro_f1_igual_a_sklearn():
    rng = np.random.default_rng(42)
    etiquetas = ["11", "31-33", "43", "46", "72"]
    verdaderos = [str(x) for x in rng.choice(etiquetas[:4], size=200)]
    predichos = [None if rng.random() < 0.15 else str(rng.choice(etiquetas)) for _ in verdaderos]
    propio = macro_f1(verdaderos, predichos)
    esperado = f1_score(
        verdaderos, ["(ninguno)" if p is None else p for p in predichos],
        labels=sorted(set(verdaderos)), average="macro", zero_division=0,
    )
    assert propio["valor"] == pytest.approx(esperado, abs=1e-6)
    assert propio["etiquetas"] == 4  # "72" solo aparece en predichos: no entra al promedio


def test_macro_f1_sector_sin_aciertos_vale_cero():
    res = macro_f1(["11", "43"], ["11", None])
    assert res["por_etiqueta"] == {"11": 1.0, "43": 0.0}
    assert res["valor"] == 0.5


# --- Bootstrap ---


def test_bootstrap_determinista_y_contiene_la_accuracy():
    aciertos = [True] * 70 + [False] * 30
    a, b = intervalo_bootstrap(aciertos), intervalo_bootstrap(aciertos)
    assert a == b
    assert a["inferior"] < 0.7 < a["superior"]


def test_bootstrap_extremos_y_vacio():
    assert intervalo_bootstrap([True] * 10) == {"inferior": 1.0, "superior": 1.0}
    assert intervalo_bootstrap([False] * 10) == {"inferior": 0.0, "superior": 0.0}
    assert intervalo_bootstrap([]) is None


def test_diferencia_bootstrap():
    a = [True] * 50 + [False] * 50
    d = diferencia_bootstrap(a, [True] * 80 + [False] * 20)
    assert d["diferencia"] == pytest.approx(0.3)
    assert d["ic95"]["inferior"] < 0.3 < d["ic95"]["superior"]
    assert d == diferencia_bootstrap(a, [True] * 80 + [False] * 20)
    assert diferencia_bootstrap(a, a)["diferencia"] == 0.0


def test_diferencia_pareada_iguales_es_cero():
    a = [True] * 60 + [False] * 40
    d = diferencia_pareada_bootstrap(a, list(a))
    assert d["diferencia"] == 0.0
    assert d["ic95"] == {"inferior": 0.0, "superior": 0.0}
    assert (d["n"], d["filas_solo_a_acierta"], d["filas_solo_b_acierta"]) == (100, 0, 0)


def test_diferencia_pareada_a_mano():
    """b acierta en las mismas filas que a más 10: cada remuestreo da una diferencia >= 0."""
    a = [True] * 50 + [False] * 50
    b = [True] * 60 + [False] * 40
    d = diferencia_pareada_bootstrap(a, b)
    assert d["diferencia"] == pytest.approx(0.1)
    assert (d["filas_solo_a_acierta"], d["filas_solo_b_acierta"]) == (0, 10)
    assert 0.0 <= d["ic95"]["inferior"] < 0.1 < d["ic95"]["superior"]
    rng = np.random.default_rng(42)
    idx = rng.integers(0, 100, size=(1000, 100))
    x, y = np.asarray(a, float), np.asarray(b, float)
    inf, sup = np.percentile(y[idx].mean(axis=1) - x[idx].mean(axis=1), (2.5, 97.5))
    assert d["ic95"] == {"inferior": round(float(inf), 6), "superior": round(float(sup), 6)}


def test_diferencia_pareada_signo_y_determinismo():
    a = [True] * 70 + [False] * 30
    b = [True] * 40 + [False] * 60
    d = diferencia_pareada_bootstrap(a, b)
    assert d["diferencia"] == pytest.approx(-0.3)
    assert d == diferencia_pareada_bootstrap(a, b)
    assert diferencia_pareada_bootstrap(b, a)["diferencia"] == pytest.approx(0.3)


def test_diferencia_pareada_largos_distintos_o_vacia():
    with pytest.raises(ErrorEvaluador):
        diferencia_pareada_bootstrap([True, False], [True])
    with pytest.raises(ErrorEvaluador):
        diferencia_pareada_bootstrap([], [])


def test_diferencia_pareada_mas_angosta_que_la_independiente_con_filas_correlacionadas():
    rng = np.random.default_rng(7)
    a = rng.random(1000) < 0.6
    b = a.copy()
    cambia = rng.choice(1000, size=60, replace=False)
    b[cambia] = ~b[cambia]
    p = diferencia_pareada_bootstrap(a.tolist(), b.tolist())
    i = diferencia_bootstrap(a.tolist(), b.tolist())
    assert p["diferencia"] == pytest.approx(i["diferencia"])
    assert p["ic95"]["superior"] - p["ic95"]["inferior"] < (i["ic95"]["superior"] - i["ic95"]["inferior"]) / 2


# --- Evaluación completa ---


def _caso():
    filas = [
        _fila(1003, "311811", sin_nombre=True),
        _fila(1001, "461110", generico=True),
        _fila(1002, "461110"),
        _fila(1004, "722511", sin_nombre=True, generico=True),
        _fila(1005, "431110"),
    ]
    pred = {"1001": "461110", "1002": " 461110\n", "1003": "322110", "1004": "sin código", "1005": "431199"}
    return filas, pred


def test_evaluacion_completa():
    filas, pred = _caso()
    res = evaluar_validacion(pred, filas, CATALOGO)
    r5 = res["regla5"]
    assert r5["conteos"] == {"formato_invalido": 1, "codigo_inexistente": 1, "codigo_existente": 3}
    assert r5["aciertos"] == 2 and r5["accuracy"] == 0.4
    assert res["por_nivel"]["clase"]["accuracy"] == r5["accuracy"]
    assert res["por_nivel"]["sector"]["aciertos"] == 3  # 322110 cae en 31-33, como 311811
    assert (res["subgrupos"]["sin_nombre"]["con"]["filas"], res["subgrupos"]["sin_nombre"]["con"]["aciertos"]) == (2, 0)
    assert res["subgrupos"]["sin_nombre"]["sin"]["accuracy"] == pytest.approx(2 / 3, abs=1e-6)
    assert res["subgrupos"]["generico"]["con"]["filas"] == 2
    assert res["concentracion"]["codigo_valido_mas_predicho"] == "461110"
    assert res["concentracion"]["filas_con_ese_codigo"] == 2
    assert res["macro_f1_sector"]["sectores_en_la_referencia"] == 4


def test_el_resultado_no_trae_ids_ni_respuestas():
    filas, pred = _caso()
    texto = json.dumps(evaluar_validacion(pred, filas, CATALOGO), ensure_ascii=False)
    assert not any(f["id"] in texto for f in filas)
    assert "sin código" not in texto


def test_el_orden_de_las_filas_no_cambia_el_resultado():
    filas, pred = _caso()
    assert evaluar_validacion(pred, filas, CATALOGO) == evaluar_validacion(pred, filas[::-1], CATALOGO)


def test_ids_que_no_coinciden():
    filas, pred = _caso()
    del pred["1005"]
    with pytest.raises(ErrorEvaluador, match="faltan 1"):
        evaluar_validacion(pred, filas, CATALOGO)


def test_referencias_invalidas():
    with pytest.raises(ErrorEvaluador, match="bool"):
        validar_referencias([{"id": "1", "codigo_act": "461110", "sin_nombre": 1, "generico": False}])
    with pytest.raises(ErrorEvaluador, match="repetidos"):
        validar_referencias([_fila(1, "461110"), _fila(1, "431110")])
    with pytest.raises(ErrorEvaluador, match="seis dígitos"):
        validar_referencias([_fila(1, "46111")])


# --- Entradas verificadas ---

CATALOGO_COMPLETO = [f"{s[:2]}1110" for s in SECTORES_2023]


def _reg(i: int, codigo: str) -> dict:
    return {"id": str(i), "nom_estab": f"Negocio inventado {i}", "codigo_act": codigo, "llave": f"NEGOCIO INVENTADO {i}",
            "sin_nombre": False, "generico": False, "igual_titulo": False}


def _repo(tmp_path: Path) -> tuple[Path, dict]:
    raiz = tmp_path / "repo"
    (raiz / "data" / "splits").mkdir(parents=True)
    cat = ("codigo,titulo\n" + "".join(f"{c},Título inventado {c}\n" for c in sorted(CATALOGO_COMPLETO))).encode("utf-8")
    (raiz / "data" / "cat.csv").write_bytes(cat)
    archivos = {"val.jsonl": [_reg(10 + k, CATALOGO_COMPLETO[k]) for k in range(5)],
                "test.jsonl": [_reg(90 + k, "111110") for k in range(3)],
                "hard.jsonl": [_reg(95 + k, "111110") for k in range(2)]}
    splits = {}
    for nombre, regs in archivos.items():
        datos = serializar(regs)
        (raiz / "data" / "splits" / nombre).write_bytes(datos)
        entrada = {"val.jsonl": "clasificador_validacion", "test.jsonl": "clasificador_test",
                   "hard.jsonl": "clasificador_hard"}[nombre]
        splits[entrada] = {"ruta": f"data/splits/{nombre}", "sha256": hashlib.sha256(datos).hexdigest(),
                           "filas": len(regs)}
    manifest = {"scian_version": "2023", "splits": splits,
                "catalogos": {"scian_2023_clases": {"ruta": "data/cat.csv", "sha256": hashlib.sha256(cat).hexdigest()}}}
    return raiz, manifest


def test_cargar_codigos_y_referencias_verificados(tmp_path):
    raiz, manifest = _repo(tmp_path)
    assert cargar_codigos(manifest, raiz) == frozenset(CATALOGO_COMPLETO)
    assert len(cargar_referencias(manifest, raiz=raiz)) == 5
    (raiz / "data" / "cat.csv").write_bytes(b"codigo,titulo\n111110,Otro\n")
    from src.muestra_humo import ErrorMuestra

    with pytest.raises(ErrorMuestra, match="SHA-256"):
        cargar_codigos(manifest, raiz)


def test_cargar_referencias_se_niega_con_test_y_hard(tmp_path, monkeypatch):
    raiz, manifest = _repo(tmp_path)
    abiertos: list[str] = []
    original = io.open

    def espia(archivo, *args, **kwargs):
        if not isinstance(archivo, int):
            abiertos.append(Path(os.fspath(archivo)).name)
        return original(archivo, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", espia)
    monkeypatch.setattr(io, "open", espia)
    for entrada in ("clasificador_test", "clasificador_hard"):
        with pytest.raises(ErrorEvaluador, match="Fase 4"):
            cargar_referencias(manifest, entrada, raiz)
    filas = cargar_referencias(manifest, raiz=raiz)
    codigos = cargar_codigos(manifest, raiz)
    evaluar_validacion({f["id"]: f["codigo_act"] for f in filas}, filas, codigos)
    monkeypatch.undo()
    assert "val.jsonl" in abiertos  # el espía sí ve las aperturas
    assert "test.jsonl" not in abiertos and "hard.jsonl" not in abiertos
