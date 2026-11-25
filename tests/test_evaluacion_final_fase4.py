"""Pruebas de la evaluación final de la Fase 4 (src/evaluacion_final_fase4.py), con datos inventados.

El config es el pre-registro real (versionado y sin nombres); los nombres, ids, códigos, respuestas y
registros de validación son inventados. Nunca se lee data/splits/test.jsonl ni hard.jsonl.
"""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import src.evaluacion_final_fase4 as ef
from src.baselines_fase4 import EstadoGit
from src.denue import REPO_RAIZ
from src.evaluador_clasificador import aciertos_por_nivel, diferencia_pareada_bootstrap
from src.exploracion_clasificador import ErrorPrivacidad
from src.splits import sha256_bytes

CONFIG_REAL = json.loads((REPO_RAIZ / ef.RUTA_CONFIG).read_text(encoding="utf-8"))
CODIGOS = ("461110", "722511", "311110", "811111", "541110", "431110")
TITULOS = {c: f"Título inventado {c}" for c in CODIGOS}
LIMPIO = EstadoGit(commit="a" * 40, arbol_limpio=True, versionado=True)


def _fila(i: int, conj: str, codigo: str, **extra) -> dict:
    return {"id": str(i), "nom_estab": f"Negocio inventado {conj} {i}", "codigo_act": codigo, "llave": "",
            "sin_nombre": i % 5 == 0, "generico": i % 4 == 0, "igual_titulo": False, **extra}


def _respuesta(i: int, codigo: str, modelo: str) -> str:
    """Respuestas deterministas: aciertos, errores de clase, de sector, inválidas e inexistentes."""
    k = (i * 7 + len(modelo)) % 6
    if "base" in modelo and k == 0:
        return "La clase es " + codigo  # solo acierta con la tolerante
    if "base" in modelo and k == 1:
        return "no sé"
    if k == 2:
        return CODIGOS[(CODIGOS.index(codigo) + 1) % len(CODIGOS)]
    if k == 3 and "base" in modelo:
        return "999999"
    return codigo


def _registrado(preds_val, val, codigos):
    reg = {}
    for m, p in preds_val.items():
        reg[m] = {"metricas": ef.metricas_modelo(p, val, codigos, None, False)}
    ac = {m: aciertos_por_nivel(p, val, codigos) for m, p in preds_val.items()}
    for a, b, dueño, claves in ef.PAREADAS_VALIDACION:
        obj = reg[dueño]
        for k in claves[:-1]:
            obj = obj.setdefault(k, {})
        obj[claves[-1]] = {n: diferencia_pareada_bootstrap(ac[a][n], ac[b][n]) for n in ("clase", "sector")}
    return reg


def juguete(tmp_path: Path):
    """Entradas completas de juguete y las etiquetas de test y hard (que en la corrida real solo lee --test)."""
    config = copy.deepcopy(CONFIG_REAL)
    val = [_fila(100 + 3 * i, "val", CODIGOS[i % 6]) for i in range(24)]
    test = [_fila(500 + 5 * i, "test", CODIGOS[(i * 2) % 6]) for i in range(30)]
    hard = [_fila(900 + i, "hard", CODIGOS[(i * 5) % 6], tipo=("ambiguo", "indirecto")[i % 2]) for i in range(12)]
    etiquetas = {"test": test, "hard": hard}
    modelos = [m["id"] for m in config["modelos"]]
    preds, tope = {}, {}
    for m in modelos:
        conjs = {"test": test, "hard": hard}
        if m in ("diccionario", "tfidf-lr-C10", "06b-lr2e-4-r32", "4b-lr2e-4-r32"):
            conjs["validacion"] = val
        # Diccionarios en orden inverso a int(id): el cálculo debe alinear por id, no por orden de llegada.
        preds[m] = {c: {f["id"]: _respuesta(int(f["id"]), f["codigo_act"], m) for f in reversed(fs)} for c, fs in conjs.items()}
        if m in ef.llm(config):
            tope[m] = {c: {f["id"]: ("base" in m and int(f["id"]) % 9 == 0) for f in fs} for c, fs in conjs.items()}
    # Predicciones de validación de referencia de los afinados (las del kernel son idénticas).
    for m in ("06b-lr2e-4-r32", "4b-lr2e-4-r32"):
        ruta = Path("ref") / f"{m}.jsonl"
        datos = "".join(json.dumps({"id": f["id"], "respuesta": preds[m]["validacion"][f["id"]]}) + "\n" for f in val).encode()
        (tmp_path / ruta).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / ruta).write_bytes(datos)
        ef.modelo_config(config, m)["predicciones_validacion"] = {"ruta": ruta.as_posix(), "sha256": sha256_bytes(datos)}
    config["conjuntos"]["test"]["filas"], config["conjuntos"]["hard"]["filas"] = len(test), len(hard)
    config["conjuntos"]["hard"]["por_tipo"] = {"ambiguo": 6, "indirecto": 6}
    codigos = frozenset(CODIGOS)
    registrados = _registrado({m: preds[m]["validacion"] for m in modelos if "validacion" in preds[m]},
                              sorted(val, key=lambda f: int(f["id"])), codigos)
    por_modelo = {m: {"horas_gpu_procesos": 0.1, "segundos_proceso": 360.0,
                      "conjuntos": {c: {"s_por_fila": 0.2, "pct_al_tope": 0.0} for c in ("test", "hard")}}
                  for m in ef.llm(config)}
    salida = SimpleNamespace(
        costos={"segundos_kernel": 3600.0, "horas_gpu_kernel": 1.0, "segundos_etapas": 3000.0, "segundos_fuera_de_etapas": 600.0,
                "duraciones_orquestador_s": {"gpu": 1.0}, "por_modelo": por_modelo, "nota": "juguete"},
        mediciones={"estado": "completo"}, etapas=[{"estado": "completa"}] * 14,
        compuertas={m: {"local_identicas": 24, "filas": 24, "minimo": 23} for m in ("06b-lr2e-4-r32", "4b-lr2e-4-r32")},
        imagen={"misma_imagen_que_el_4b": True, "docker_image_metadata": "imagen", "pip_freeze_antes_identico_al_del_4b": True,
                "entorno": {"python": "3.13"}, "controlador": "1"},
        versiones={"no_fijables": {}, "fijadas_distintas": {}},
        descarga={"archivos_bajados": 48, "coinciden_con_kaggle": 46, "con_sha256_de_kaggle": 46},
        nombres={"archivos_revisados": 50, "posible_fuga": False},
    )
    manifest = {"scian_version": "2023",
                "kaggle_datasets": {"denue_scian_evaluacion": {"archivos": {"test.jsonl": {"sha256": "c" * 64},
                                                                            "hard.jsonl": {"sha256": "d" * 64}}}}}
    refs = {c: [{"id": f["id"], "nom_estab": f["nom_estab"]} for f in fs] for c, fs in (("test", test), ("hard", hard))}
    refs["validacion"] = [{"id": f["id"], "nom_estab": f["nom_estab"], "codigo_act": f["codigo_act"]} for f in val]
    refs["train_final"] = [{"id": str(10_000 + i), "nom_estab": f"Taller de juguete {i}", "codigo_act": CODIGOS[i % 6]}
                           for i in range(20)]
    ent = ef.Entradas(
        config=config, sha_config=ef.SHA_CONFIG, manifest=manifest, codigos=codigos, titulos=TITULOS, salida=salida,
        predicciones=preds, al_tope=tope, fuentes={m: {"test": {"ruta": f"x/{m}", "sha256": "e" * 64}} for m in modelos},
        registro_baselines={"cpu": {"tiempos_s": {"ajuste_s": 4080.0}}}, sha_registro_baselines="f" * 64, validacion=val,
        referencias_nombres=refs, registrados=registrados,
        sha_registrados={m: "0" * 64 for m in registrados}, raiz=tmp_path,
    )
    return ent, etiquetas


# --- Piezas ---


def test_pareadas_alineadas_por_id_contra_un_calculo_a_mano():
    filas = [{"id": i, "codigo_act": c, "sin_nombre": False, "generico": False}
             for i, c in (("3", "461110"), ("20", "722511"), ("100", "311110"), ("7", "461110"))]
    a = {"100": "311110", "7": "722511", "3": "461110", "20": "722511"}  # aciertos en orden de int(id): 3,7,20,100 → 1,0,1,1
    b = {"20": "461110", "3": "461110", "100": "311110", "7": "461110"}  # → 1,1,0,1
    config = {"comparaciones": {g: {"metrica": "accuracy de clase", "pares": [{"a": "a", "b": "b"}]} for g in ef.GRUPOS}}
    p = ef.pareadas({"a": a, "b": b}, filas, frozenset(CODIGOS), config)["principales"][0]
    assert p["diferencia"] == 0.0
    assert p["filas_solo_a_acierta"] == 1 and p["filas_solo_b_acierta"] == 1 and p["n"] == 4
    esperado = diferencia_pareada_bootstrap([True, False, True, True], [True, True, False, True])
    assert p["ic95"] == esperado["ic95"]


def test_codigos_mas_predichos_solo_validos():
    pred = {"1": "461110", "2": "461110", "3": "texto libre", "4": "722511 ", "5": "999999"}
    top = ef.codigos_mas_predichos(pred, list(pred))
    assert top[0] == {"codigo": "461110", "filas": 2}
    assert {t["codigo"] for t in top} == {"461110", "722511", "999999"}
    assert "texto" not in repr(top)


def test_errores_4b_sigue_la_convencion_del_sorteo():
    config = copy.deepcopy(CONFIG_REAL)
    config["errores_para_analisis"]["n"] = 3
    filas = [_fila(10 + i, "test", CODIGOS[i % 6]) for i in range(20)]
    pred = {f["id"]: (f["codigo_act"] if i % 3 else "no sé") for i, f in enumerate(filas)}
    sel, info = ef.errores_4b(pred, filas, frozenset(CODIGOS), TITULOS, config)
    errores = [f["id"] for i, f in enumerate(filas) if not i % 3]
    pos = np.sort(np.random.default_rng(42).permutation(len(errores))[:3])
    assert [r["id"] for r in sel] == [errores[int(p)] for p in pos]
    assert info["n_errores"] == len(errores) and info["n"] == 3
    assert all(tuple(r) == ef.COLUMNAS_ERRORES for r in sel)
    assert all(r["categoria_regla5"] == "formato_invalido" and r["codigo_respondido"] == "" for r in sel)


def test_bytes_errores_jsonl_y_csv_con_bom():
    filas = [dict.fromkeys(ef.COLUMNAS_ERRORES, "x") for _ in range(2)]
    b = ef.bytes_errores(filas)
    assert b["csv"].startswith(b"\xef\xbb\xbf" + ",".join(ef.COLUMNAS_ERRORES).encode())
    assert b["jsonl"].count(b"\n") == 2 and b["csv"].count(b"\n") == 3
    with pytest.raises(ef.ErrorFinal):
        ef.bytes_errores([{"id": "1"}])


def test_comparar_da_rutas():
    assert ef.comparar({"a": 1, "b": {"c": [1, 2]}}, {"a": 1, "b": {"c": [1, 2]}}) == []
    assert ef.comparar({"a": 1, "b": {"c": [1, 3]}}, {"a": 2, "b": {"c": [1, 2]}, "d": 0}) == ["$.a", "$.b.c[1]", "$.d"]


def test_dos_veces_exige_bytes_identicos():
    contador = iter(range(10))
    with pytest.raises(ef.ErrorFinal, match="doble cálculo"):
        ef.dos_veces(lambda: {"x": str(next(contador)).encode()})
    assert ef.dos_veces(lambda: {"x": b"1"}) == {"x": b"1"}


# --- Ensayo en validación ---


def test_validar_reproduce_lo_registrado(tmp_path):
    ent, _ = juguete(tmp_path)
    v = ef.validar(ent)
    assert v["todo_identico"]
    assert set(v["modelos"]) == {"diccionario", "tfidf-lr-C10", "06b-lr2e-4-r32", "4b-lr2e-4-r32"}
    assert v["modelos"]["4b-lr2e-4-r32"]["respuestas_del_kernel_identicas_a_la_referencia"] == 24
    assert len(v["pareadas"]) == 6


def test_validar_reporta_diferencias(tmp_path):
    ent, _ = juguete(tmp_path)
    ent.registrados["tfidf-lr-C10"]["metricas"]["regla5"]["aciertos"] += 1
    ent.registrados["4b-lr2e-4-r32"]["contra_referencia"]["clase"]["diferencia"] = 0.5
    v = ef.validar(ent)
    assert not v["todo_identico"]
    assert v["modelos"]["tfidf-lr-C10"]["diferencias"] == ["$.regla5.aciertos"]
    malas = [p for p in v["pareadas"] if not p["identica"]]
    assert [(p["a"], p["b"], p["nivel"], p["diferencias"]) for p in malas] == [
        ("06b-lr2e-4-r32", "4b-lr2e-4-r32", "clase", ["$.diferencia"])]


def test_correr_validacion_no_abre_test_ni_escribe(tmp_path, monkeypatch):
    ent, _ = juguete(tmp_path)

    def prohibido(*a, **k):
        raise AssertionError("--validacion no debe leer etiquetas de test ni de hard")

    monkeypatch.setattr(ef, "cargar_etiquetas", prohibido)
    antes = sorted(p for p in tmp_path.rglob("*"))
    mensajes = []
    v = ef.correr_validacion(ent, avance=mensajes.append)
    assert v["todo_identico"]
    assert sorted(p for p in tmp_path.rglob("*")) == antes
    assert "Maqueta" in mensajes[0]


# --- Resultado, tabla y candados ---


def test_calcular_y_armar(tmp_path):
    ent, etiquetas = juguete(tmp_path)
    v = ef.validar(ent)
    calc, filas_err = ef.calcular(ent, etiquetas, v)
    assert calc["test"]["filas"] == 30
    assert [calc["hard"][s]["filas"] for s in ef.SUBCONJUNTOS_HARD] == [12, 6, 6]
    for m, x in calc["test"]["modelos"].items():
        assert sum(x["regla5"]["conteos"].values()) == x["regla5"]["total"] == 30
        assert ("tolerante" in x) == (m in ef.llm(ent.config))
    base = calc["test"]["modelos"]["06b-base-0shot"]
    assert base["tolerante"]["aciertos"] >= base["regla5"]["aciertos"]
    assert base["al_tope"]["al_tope"] == sum(int(f["id"]) % 9 == 0 for f in etiquetas["test"])
    assert len(calc["test"]["pareadas"]["principales"]) == 2 and len(calc["test"]["pareadas"]["secundarias_sector"]) == 7
    assert [r["modelo"] for r in calc["tabla_test"]] == [m["id"] for m in ent.config["modelos"]]
    assert {r["modelo"] for r in calc["validacion_contra_test"]} == set(v["modelos"])
    assert calc["errores_para_analisis"]["n"] == len(filas_err) <= 50
    datos = ef.preparar(ent, etiquetas, v, LIMPIO, "2026-11-25")
    r = json.loads(datos["resultado"])
    for k in ("fecha", "commit", "modelo", "configuracion", "semilla", "split", "hash_split", "tamano_muestra", "horas_gpu"):
        assert k in r
    assert ef.tabla_markdown(r).encode("utf-8") == datos["tabla"]
    assert r["errores_para_analisis"]["archivos"]["csv"]["sha256"] == sha256_bytes(datos["errores_csv"])
    assert "Negocio inventado" not in datos["resultado"].decode() and "Negocio inventado" not in datos["tabla"].decode()
    assert "Negocio inventado" in datos["errores_jsonl"].decode()


def test_candados(tmp_path):
    ent, etiquetas = juguete(tmp_path)
    filas = [*etiquetas["test"], *etiquetas["hard"]]
    llaves = {"NEGOCIO INVENTADO TEST 500"}
    ef.candados({"a": "ok"}, {"t": "solo números 1.0"}, llaves, {"500"}, filas)
    with pytest.raises(ErrorPrivacidad):
        ef.candados({"a": "Negocio inventado test 500"}, {}, llaves, set(), filas)
    with pytest.raises(ef.ErrorFinal):
        ef.candados({"a": "500"}, {}, set(), {"500"}, filas)
    with pytest.raises(ef.ErrorFinal, match="Posible fuga"):
        ef.candados({"a": "ok"}, {"tabla": "| Negocio inventado test 505 |"}, set(), set(), filas)


# --- --test ---


def test_correr_test_escribe_una_sola_vez(tmp_path):
    ent, etiquetas = juguete(tmp_path)
    llamadas = []

    def etiquetas_de(*a):
        llamadas.append(1)
        return etiquetas

    r = ef.correr_test(ent, raiz=tmp_path, fecha="2026-11-25", git=lambda: LIMPIO, etiquetas_de=etiquetas_de,
                       ignorado=lambda p: True, avance=lambda s: None)
    assert llamadas == [1]
    res = tmp_path / "results" / "2026-11-25_fase4_evaluacion-final.json"
    tabla = tmp_path / "results" / "2026-11-25_fase4_tabla-final.md"
    assert json.loads(res.read_bytes()) == r
    assert tabla.read_text(encoding="utf-8") == ef.tabla_markdown(r)
    csv_ = tmp_path / ef.CARPETA_ERRORES / f"{ef.BASE_ERRORES}.csv"
    assert sha256_bytes(csv_.read_bytes()) == r["errores_para_analisis"]["archivos"]["csv"]["sha256"]
    with pytest.raises(ef.ErrorFinal, match="ya se hizo"):
        ef.correr_test(ent, raiz=tmp_path, fecha="2026-11-26", git=lambda: LIMPIO, etiquetas_de=etiquetas_de,
                       ignorado=lambda p: True, avance=lambda s: None)
    assert llamadas == [1]


def test_correr_test_exige_arbol_limpio_e_ignorado(tmp_path):
    ent, etiquetas = juguete(tmp_path)
    sucio = EstadoGit(commit="a" * 40, arbol_limpio=False, versionado=True)
    nunca = lambda *a: pytest.fail("no debe abrir test")  # noqa: E731
    with pytest.raises(ef.ErrorFinal, match="árbol limpio"):
        ef.correr_test(ent, raiz=tmp_path, fecha="2026-11-25", git=lambda: sucio, etiquetas_de=nunca,
                       ignorado=lambda p: True, avance=lambda s: None)
    with pytest.raises(ef.ErrorFinal, match="no está ignorado"):
        ef.correr_test(ent, raiz=tmp_path, fecha="2026-11-25", git=lambda: LIMPIO, etiquetas_de=nunca,
                       ignorado=lambda p: False, avance=lambda s: None)
    assert not (tmp_path / "results").exists()


def test_cargar_etiquetas_verifica_hash_copia_y_tipos(tmp_path):
    ent, etiquetas = juguete(tmp_path)
    config = copy.deepcopy(ent.config)
    manifest = {"splits": {}}
    for conj in ef.RESERVADOS:
        regs = etiquetas[conj]
        campos = list(regs[0])
        datos = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in regs).encode("utf-8")
        (tmp_path / f"{conj}.jsonl").write_bytes(datos)
        manifest["splits"][ef.ENTRADA_SPLIT[conj]] = {"ruta": f"{conj}.jsonl", "sha256": sha256_bytes(datos),
                                                      "filas": len(regs), "campos": campos}
        config["conjuntos"][conj]["sha256"] = sha256_bytes(datos)
    copias = ent.referencias_nombres
    res = ef.cargar_etiquetas(manifest, config, copias, tmp_path)
    assert [len(res[c]) for c in ef.RESERVADOS] == [30, 12]
    otra = copy.deepcopy(copias)
    otra["test"][0]["nom_estab"] = "Otro nombre inventado"
    with pytest.raises(ef.ErrorFinal, match="copia"):
        ef.cargar_etiquetas(manifest, config, otra, tmp_path)
    config["conjuntos"]["hard"]["por_tipo"] = {"ambiguo": 7, "indirecto": 5}
    with pytest.raises(ef.ErrorFinal, match="tipos"):
        ef.cargar_etiquetas(manifest, config, copias, tmp_path)
