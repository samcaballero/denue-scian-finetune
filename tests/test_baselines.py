"""Pruebas de los baselines del clasificador (Fase 1, paso 10).

Solo usan nombres, llaves, ids y códigos inventados, en memoria o en carpetas temporales;
nunca leen el DENUE ni los splits reales.
"""

import builtins
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest

import src.baselines as bl
from src.baselines import (
    CONFIG_LR,
    CONFIG_TFIDF,
    MODELO_DICCIONARIO,
    MODELO_TFIDF,
    ErrorBaseline,
    Normalizador,
    Split,
    ajustar_diccionario,
    ajustar_lr,
    ajustar_vectorizador,
    correr,
    correr_tfidf,
    elegir_c,
    estimar,
    medir_submuestras,
    predecir_diccionario,
    verificar_llaves,
)
from src.evaluador_clasificador import SECTORES_2023
from src.exploracion_clasificador import ErrorPrivacidad
from src.normalizacion import normalizar_nombre
from src.split_aleatorio import partir, unir
from src.splits import serializar as serializar_split

A, B, C_, D = "461110", "311110", "721110", "111110"
PALABRA = {A: "Abarrotes", B: "Panadería", C_: "Hotel", D: "Rancho"}


def _reg(i: int, nombre: str, codigo: str, sin_nombre: bool = False, generico: bool = False) -> dict:
    return {"id": str(i), "nom_estab": nombre, "codigo_act": codigo, "llave": normalizar_nombre(nombre),
            "sin_nombre": sin_nombre, "generico": generico, "igual_titulo": False}


# --- Diccionario ---


def test_empate_dentro_de_la_llave_por_el_conteo_antes_del_tope():
    train = [_reg(1, "Café Luna", A), _reg(2, "Café Luna", B)]
    pre = train + [_reg(3, "Café Luna", B), _reg(4, "Café Luna", B), _reg(5, "Café Luna", A)]
    d = ajustar_diccionario(train, pre)
    assert d.por_llave["CAFE LUNA"] == B  # 3 contra 2 antes del tope
    assert d.empates["llaves_con_empate"] == 1 and d.empates["resueltas_por_el_conteo_antes_del_tope"] == 1


def test_empate_que_persiste_se_rompe_por_el_codigo_menor():
    train = [_reg(1, "Café Luna", A), _reg(2, "Café Luna", B)]
    d = ajustar_diccionario(train, train)
    assert d.por_llave["CAFE LUNA"] == B  # 311110 < 461110
    assert d.empates["resueltas_por_el_codigo_menor"] == 1


def test_la_mayoria_de_la_llave_gana_sin_mirar_el_train_antes_del_tope():
    train = [_reg(1, "Café Luna", A), _reg(2, "Café Luna", A), _reg(3, "Café Luna", B)]
    pre = train + [_reg(10 + k, "Café Luna", B) for k in range(10)]
    assert ajustar_diccionario(train, pre).por_llave["CAFE LUNA"] == A


def test_empate_global_por_el_conteo_antes_del_tope_y_luego_por_el_codigo():
    train = [_reg(1, "Uno", A), _reg(2, "Dos", A), _reg(3, "Tres", C_), _reg(4, "Cuatro", C_), _reg(5, "Cinco", B)]
    pre = train + [_reg(6, "Seis", C_)]
    d = ajustar_diccionario(train, pre)
    assert d.clase_global == C_
    assert d.empates["global"] == {"clase": C_, "filas_en_train": 2, "clases_empatadas_en_train": 2,
                                   "filas_antes_del_tope": 3, "resuelto_por": "conteo antes del tope"}
    d2 = ajustar_diccionario(train, train)
    assert d2.clase_global == A and d2.empates["global"]["resuelto_por"] == "código menor"


def test_llave_no_vista_da_la_global_y_la_llave_sale_de_nom_estab():
    train = [_reg(1, "Café Luna", B), _reg(2, "Otro", A), _reg(3, "Otro más", A)]
    d = ajustar_diccionario(train, train)
    assert predecir_diccionario(d, ["  café   LUNA.", "Nunca visto"]) == [B, A]


def test_verificar_llaves_y_train_fuera_del_pre_tope():
    with pytest.raises(ErrorBaseline, match="llave distinta"):
        verificar_llaves([{**_reg(1, "Café Luna", A), "llave": "OTRA"}], Normalizador())
    with pytest.raises(ErrorBaseline, match="fuera del train antes del tope"):
        ajustar_diccionario([_reg(1, "Café Luna", A)], [_reg(2, "Café Luna", A)])


# --- TF-IDF + LR ---


def _nombres(codigo: str, n: int, inicio: int, sufijo: str = "inventado") -> list[dict]:
    return [_reg(inicio + k, f"{PALABRA[codigo]} {sufijo} {k % 7}", codigo) for k in range(n)]


def _split_chico() -> Split:
    train = [r for i, c in enumerate((A, B, C_)) for r in _nombres(c, 20, 1000 * (i + 1))]
    val = [_reg(9000 + k, f"{PALABRA[c]} qqzz {k}", c) for k, c in enumerate((A, B, C_, A, B))]
    return Split(nombre="nombre", train=train, val=val, pre_tope=train, hash_split={}, hash_split_origen="",
                 hash_pre_tope={}, descripcion="")


def test_el_vectorizador_se_ajusta_solo_con_train():
    s = _split_chico()
    codigos = frozenset({A, B, C_, D, *(f"{x[:2]}1110" for x in SECTORES_2023)})
    _, (v, X, _) = correr_tfidf(s, 1.0, codigos)
    assert X.shape[0] == len(s.train)
    vocab = set(v.vocabulary_)
    assert "qq" not in vocab and "zz" not in vocab  # n-gramas que solo están en validación
    v_solo_train, _, _ = ajustar_vectorizador([r["nom_estab"] for r in s.train])
    assert v.vocabulary_ == v_solo_train.vocabulary_
    assert np.array_equal(v.idf_, v_solo_train.idf_)


def test_la_regresion_logistica_es_determinista():
    s = _split_chico()
    _, X, _ = ajustar_vectorizador([r["nom_estab"] for r in s.train])
    y = [r["codigo_act"] for r in s.train]
    a, b = ajustar_lr(X, y, 10.0), ajustar_lr(X, y, 10.0)
    assert np.array_equal(a.modelo.coef_, b.modelo.coef_)
    assert a.n_iter >= 1 and a.memoria_pico_mb >= 0


def test_elegir_c_gana_el_de_mas_aciertos_y_en_empate_el_menor():
    assert elegir_c([{"C": 1.0, "aciertos": 5}, {"C": 10.0, "aciertos": 7}, {"C": 100.0, "aciertos": 7}]) == 10.0
    assert elegir_c([{"C": 100.0, "aciertos": 3}, {"C": 1.0, "aciertos": 3}]) == 1.0
    with pytest.raises(ErrorBaseline):
        elegir_c([])


def test_punto_de_control_en_submuestras():
    train = [r for i, c in enumerate((A, B, C_)) for r in _nombres(c, 30, 1000 * (i + 1))]
    filas = medir_submuestras(train, submuestras=(30, 60), valores_c=(1.0, 10.0))
    assert [(f["filas"], f["C"]) for f in filas] == [(30, 1.0), (30, 10.0), (60, 1.0), (60, 10.0)]
    est = estimar(filas, filas_full=600, clases_full=3, memoria_datos_mb=100.0)
    assert est["submuestra_de_referencia"] == 60
    assert [c["C"] for c in est["por_c"]] == [1.0, 10.0]
    assert est["escala_filas_por_clases"] == pytest.approx(600 * 3 / (60 * filas[-1]["clases"]), abs=0.01)
    assert est["total_horas"]["pesimista_s"] >= est["total_horas"]["lineal_s"]
    with pytest.raises(ErrorBaseline, match="pasa del train"):
        medir_submuestras(train, submuestras=(1000,))


# --- Flujo completo en un repo de juguete ---


def _sha(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def _repo(tmp_path: Path):
    raiz = tmp_path / "repo"
    (raiz / "data" / "splits").mkdir(parents=True)
    (raiz / "results").mkdir()
    origen = sorted((r for i, c in enumerate((A, B, C_, D)) for r in _nombres(c, 25, 10_000 * (i + 1))),
                    key=lambda r: int(r["id"]))
    val = [_reg(90_000 + k, f"{PALABRA[c]} único {k}", c, sin_nombre=k == 0) for k, c in enumerate((A, B, C_, D) * 3)]
    s = partir(unir(origen, val), 1000, n_val=12)
    archivos = {
        "clasificador_train_final": ("train_final.jsonl", origen),
        "clasificador_train_candidatos_sin_hard": ("train_candidatos_sin_hard.jsonl", origen),
        "clasificador_validacion": ("val.jsonl", val),
        "experimento_aleatorio_train_final": ("aleatorio_train_final.jsonl", s.train),
        "experimento_aleatorio_validacion": ("aleatorio_val.jsonl", s.val),
        "clasificador_test": ("test.jsonl", [_reg(99_000 + k, f"Prueba inventada {k}", A) for k in range(3)]),
        "clasificador_hard": ("hard.jsonl", [_reg(99_500 + k, f"Difícil inventado {k}", B) for k in range(2)]),
    }
    manifest = {"scian_version": "2023", "splits": {}, "catalogos": {}}
    for entrada, (nombre, regs) in archivos.items():
        datos = serializar_split(regs)
        (raiz / "data" / "splits" / nombre).write_bytes(datos)
        manifest["splits"][entrada] = {"ruta": f"data/splits/{nombre}", "sha256": _sha(datos), "filas": len(regs)}
    codigos = sorted({A, B, C_, D, *(f"{x[:2]}1110" for x in SECTORES_2023)})
    cat = ("codigo,titulo\n" + "".join(f"{c},Título inventado {c}\n" for c in codigos)).encode("utf-8")
    (raiz / "data" / "cat.csv").write_bytes(cat)
    manifest["catalogos"]["scian_2023_clases"] = {"ruta": "data/cat.csv", "sha256": _sha(cat)}
    return raiz, manifest


def _correr(raiz, manifest, commit="a" * 40, **kw):
    ctx = {"fecha": "2026-11-22", "commit": commit, "arbol_limpio": True}
    return correr(manifest, raiz=raiz, carpeta_registros=raiz / "data" / "baselines", carpeta_resultados=raiz / "results",
                  ctx=ctx, valores_c=(1.0, 10.0), avance=lambda _: None, **kw)


def test_flujo_completo_escribe_los_resultados_y_nunca_abre_test_ni_hard(tmp_path, monkeypatch):
    raiz, manifest = _repo(tmp_path)
    abiertos: list[str] = []
    original = io.open

    def espia(archivo, *args, **kwargs):
        if not isinstance(archivo, int):
            abiertos.append(Path(os.fspath(archivo)).name)
        return original(archivo, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", espia)
    monkeypatch.setattr(io, "open", espia)
    escritos = _correr(raiz, manifest)
    monkeypatch.undo()
    nombres = sorted(p.name for p in escritos)
    assert nombres == sorted([
        "2026-11-22_fase1_baseline-diccionario_split-nombre.json",
        "2026-11-22_fase1_baseline-diccionario_split-aleatorio.json",
        "2026-11-22_fase1_baseline-tfidf-lr_split-nombre.json",
        "2026-11-22_fase1_baseline-tfidf-lr_split-aleatorio.json",
        "2026-11-22_fase1_fuga-split-aleatorio.json",
    ])
    assert {"train_final.jsonl", "val.jsonl", "aleatorio_val.jsonl"} <= set(abiertos)  # el espía sí ve las aperturas
    assert "test.jsonl" not in abiertos and "hard.jsonl" not in abiertos
    assert not list((raiz / "results").glob("*.tmp")) and not list((raiz / "data" / "baselines").rglob("*.tmp"))

    todos = [r for f in ("train_final.jsonl", "val.jsonl", "aleatorio_val.jsonl")
             for r in map(json.loads, (raiz / "data" / "splits" / f).read_text(encoding="utf-8").splitlines())]
    for p in escritos:
        texto = p.read_text(encoding="utf-8")
        res = json.loads(texto)
        for campo in ("fecha", "commit", "modelo", "configuracion", "semilla", "split", "hash_split", "tamano_muestra",
                      "horas_gpu"):
            assert campo in res
        assert not any(f'"{r["id"]}"' in texto or r["nom_estab"] in texto or f'"{r["llave"]}"' in texto for r in todos)

    dic = json.loads((raiz / "results" / "2026-11-22_fase1_baseline-diccionario_split-nombre.json").read_text("utf-8"))
    assert dic["modelo"] == MODELO_DICCIONARIO
    assert dic["detalle"]["validacion_con_llave_vista_en_train"]["filas"] == 0  # regla 2
    assert dic["metricas"]["concentracion"]["clases_existentes_distintas_predichas"] == 1
    tf = json.loads((raiz / "results" / "2026-11-22_fase1_baseline-tfidf-lr_split-nombre.json").read_text("utf-8"))
    assert tf["modelo"] == MODELO_TFIDF and [b["C"] for b in tf["barrido"]] == [1.0, 10.0]
    assert tf["configuracion"]["c_elegido"] == elegir_c(tf["barrido"])
    tfa = json.loads((raiz / "results" / "2026-11-22_fase1_baseline-tfidf-lr_split-aleatorio.json").read_text("utf-8"))
    assert tfa["configuracion"]["regresion_logistica"]["C"] == tf["configuracion"]["c_elegido"]
    assert tfa["hash_split"]["validacion"] == manifest["splits"]["experimento_aleatorio_validacion"]["sha256"]
    fuga = json.loads((raiz / "results" / "2026-11-22_fase1_fuga-split-aleatorio.json").read_text("utf-8"))
    d = fuga["por_modelo"][MODELO_DICCIONARIO]
    assert d["fuga_clase"]["diferencia"] == pytest.approx(d["accuracy_split_aleatorio"] - d["accuracy_split_nombre"], abs=1e-6)


def test_una_corrida_cortada_reusa_los_ajustes_terminados(tmp_path, monkeypatch):
    raiz, manifest = _repo(tmp_path)
    original = bl.correr_tfidf
    llamadas: list[str] = []
    cortar = {"si": True}

    def espia(split, C, *args, **kwargs):
        if cortar["si"] and split.nombre == "aleatorio":
            raise KeyboardInterrupt  # el corte llega en el último ajuste
        llamadas.append(split.nombre)
        return original(split, C, *args, **kwargs)

    monkeypatch.setattr(bl, "correr_tfidf", espia)
    with pytest.raises(KeyboardInterrupt):
        _correr(raiz, manifest)
    hechos = sorted(p.name for p in (raiz / "results").iterdir())
    assert len(hechos) == 3  # los dos del diccionario y el de TF-IDF por nombre, completos
    assert not list((raiz / "results").glob("*.tmp"))
    assert llamadas == ["nombre", "nombre"]
    cortar["si"] = False
    escritos = _correr(raiz, manifest)
    assert sorted(p.name for p in escritos) == [
        "2026-11-22_fase1_baseline-tfidf-lr_split-aleatorio.json", "2026-11-22_fase1_fuga-split-aleatorio.json"]
    assert llamadas == ["nombre", "nombre", "aleatorio"]  # el barrido no se repitió


def test_otro_commit_no_sobrescribe_ni_mezcla(tmp_path):
    raiz, manifest = _repo(tmp_path)
    _correr(raiz, manifest)
    with pytest.raises(ErrorBaseline, match="otro commit"):
        _correr(raiz, manifest, commit="b" * 40)


def test_la_revision_de_privacidad_detiene_un_resultado_con_una_llave(tmp_path, monkeypatch):
    raiz, manifest = _repo(tmp_path)
    llave = normalizar_nombre(f"{PALABRA[A]} inventado 3")
    monkeypatch.setattr(bl, "config_diccionario", lambda: {"nota": llave})
    with pytest.raises(ErrorPrivacidad):
        _correr(raiz, manifest)
    assert not list((raiz / "results").iterdir())


def test_cargar_se_niega_con_test_y_hard(tmp_path):
    raiz, manifest = _repo(tmp_path)
    for entrada in ("clasificador_test", "clasificador_hard"):
        with pytest.raises(ErrorBaseline, match="Fase 4"):
            bl._leer(manifest, entrada, raiz)


def test_configuracion_documentada():
    assert CONFIG_TFIDF["analyzer"] == "char_wb" and CONFIG_LR["solver"] == "saga" and CONFIG_LR["random_state"] == 42
    assert "solo con los nombres de train" in bl.PREPROCESAMIENTO_TFIDF
