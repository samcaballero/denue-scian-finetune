"""Pruebas del orquestador del kernel de la Fase 4 (src/inferencia_fase4_kaggle.py), en CPU.

Usan el pre-registro real (versionado, sin nombres), los datos de juguete de tests/juguete_fase4.py
y un proceso falso en lugar de src/etapa_fase4_kaggle.py: nunca leen el DENUE ni corren torch.
"""

import ast
import copy
import json
import os
import sys
import textwrap
from pathlib import Path

import pytest

from src import barrido_kaggle as bk
from src import humo_kaggle as hk
from src import inferencia_fase4_kaggle as fo
from src.denue import REPO_RAIZ
from tests import juguete_fase4 as jg

CONFIG = jg.CONFIG_REAL


# --- Plan del config real ---


def test_plan_del_config_real():
    plan = fo.validar_config_kernel(CONFIG)
    ids = [e["id"] for e in plan["etapas"]]
    assert ids == [
        "06b-lr2e-4-r32__validacion", "06b-lr2e-4-r32__test", "06b-lr2e-4-r32__hard",
        "4b-lr2e-4-r32__validacion", "4b-lr2e-4-r32__test", "4b-lr2e-4-r32__hard",
        "06b-base-0shot__test", "06b-base-0shot__hard", "4b-base-0shot__test", "4b-base-0shot__hard",
        "06b-base-fewshot__test", "06b-base-fewshot__hard", "4b-base-fewshot__test", "4b-base-fewshot__hard",
    ]
    assert [e["modelo"] for e in plan["etapas"][::2]][:3] == ["06b-lr2e-4-r32", "06b-lr2e-4-r32", "4b-lr2e-4-r32"]
    assert [e["id"] for e in plan["etapas"] if e["compuerta"]] == ["06b-lr2e-4-r32__validacion", "4b-lr2e-4-r32__validacion"]
    por_id = {e["id"]: e for e in plan["etapas"]}
    assert (por_id["06b-lr2e-4-r32__test"]["max_seq_length"], por_id["06b-lr2e-4-r32__test"]["max_new_tokens"]) == (256, 16)
    assert (por_id["4b-base-0shot__hard"]["max_seq_length"], por_id["4b-base-0shot__hard"]["max_new_tokens"]) == (256, 32)
    assert (por_id["06b-base-fewshot__test"]["max_seq_length"], por_id["06b-base-fewshot__test"]["max_new_tokens"]) == (1600, 32)
    assert por_id["4b-base-fewshot__test"]["base"] == {"repo": "Qwen/Qwen3-4B", "revision": "1cfa9a7208912126459214e8b04321603b3df60c"}
    assert plan["criterio"] == {"modelos": ["06b-lr2e-4-r32", "4b-lr2e-4-r32"], "minimo_identicas": 980, "filas": 1000}
    ins = por_id["4b-lr2e-4-r32__hard"]["insumos_afinado"]
    assert {k: v["sufijo"] for k, v in ins.items()} == {
        "adaptador": "4b-lr2e-4-r32/adaptador/adapter_model.safetensors",
        "adaptador_config": "4b-lr2e-4-r32/adaptador/adapter_config.json",
        "predicciones_validacion": "4b-lr2e-4-r32/predicciones.jsonl",
    }
    assert ins["adaptador"]["sha256"] == "68898fd86fec5a4340c45268eb482ed3de76cdc2cf2e2812109343a38fbd943c"
    # Dentro de cada grupo, el 0.6B va antes que el 4B (precisión aprobada del pre-registro).
    tamanos = [e["tamano"] for e in plan["etapas"] if e["conjunto"] == "test"]
    assert tamanos == ["0.6B", "4B"] * 3


@pytest.mark.parametrize("cambio", [
    lambda c: c["uso_unico"]["reproduccion_validacion_en_el_kernel"].update(criterio_de_paro=None),
    lambda c: c["uso_unico"]["reproduccion_validacion_en_el_kernel"]["criterio_de_paro"].update(minimo_identicas=1001),
    lambda c: c["uso_unico"]["reproduccion_validacion_en_el_kernel"]["criterio_de_paro"].update(filas=999),
    lambda c: c["uso_unico"]["reproduccion_validacion_en_el_kernel"].update(modelos=["06b-lr2e-4-r32"]),
    lambda c: c["orden_en_el_kernel"].pop(),
    lambda c: c["decodificacion"]["base"].update(max_new_tokens=64),
    lambda c: c["decodificacion"]["afinados"].update(do_sample=True),
    lambda c: c["decodificacion"]["base"]["max_seq_length"].update({"few-shot": 2048}),
    lambda c: c["modelos"][2]["origen_kaggle"].update(carpeta_en_la_salida="otra/adaptador"),
    lambda c: c["modelos"][5].update(modo="2-shot"),
])
def test_config_que_no_sirve_aborta(cambio):
    c = copy.deepcopy(CONFIG)
    cambio(c)
    with pytest.raises(fo.AbortoFase4):
        fo.validar_config_kernel(c)


# --- Tiempos ---


def test_tiempos_salen_de_los_resultados_de_la_fase_2():
    r = json.loads((REPO_RAIZ / "results" / "2026-11-24_fase2_4b-lr2e-4-r32.json").read_bytes())["para_la_fase_4"]
    assert fo.S_FILA_7 == {"0.6B": r["06b-lr2e-4-r32"]["s_por_fila"], "4B": r["4b-lr2e-4-r32"]["s_por_fila"]}
    for m in ("06b-lr2e-4-r32", "4b-lr2e-4-r32"):
        f = json.loads((REPO_RAIZ / "results" / f"2026-11-24_fase2_{m}.json").read_bytes())["etapas"]["etapas"]["inferencia"]
        assert f["fichas_generadas_max"] == fo.FICHAS_AFINADOS
    assert fo.PRESUPUESTO_S == fo.TIEMPO_KAGGLE_S - fo.MARGEN_CIERRE_S
    assert fo.TIEMPO_KAGGLE_S <= 12 * 3600  # límite de una sesión de Kaggle


def test_estimaciones():
    plan = fo.validar_config_kernel(CONFIG)
    por_id = {e["id"]: e for e in plan["etapas"]}
    af, base, fs = por_id["4b-lr2e-4-r32__test"], por_id["4b-base-0shot__test"], por_id["06b-base-fewshot__hard"]
    assert fo.s_por_fila(af, "mejor") == fo.s_por_fila(af, "peor") == 0.4302
    assert fo.s_por_fila(base, "peor") == pytest.approx(0.4302 * 32 / 7) and fo.s_por_fila(base, "mejor") == 0.4302
    assert fo.s_por_fila(fs, "peor") == pytest.approx(0.2239 * 32 / 7 + 0.10)
    assert fo.estimacion_s(af) == round(180 + 2000 * 0.4302, 1)
    est = fo.estimaciones(plan)
    assert est["total_peor_s"] == pytest.approx(fo.S_ORQUESTADOR + fo.S_COMPROBACIONES + sum(fo.estimacion_s(e) for e in plan["etapas"]))
    assert est["total_mejor_s"] < est["total_peor_s"] < fo.PRESUPUESTO_S / 1.5
    assert all(x["tiempo_maximo_s"] == pytest.approx(2 * x["peor_s"]) for x in est["etapas"].values())
    with pytest.raises(ValueError):
        fo.s_por_fila(af, "medio")


# --- Compuerta ---


def _pred(ids, respuestas):
    return jg.jsonl({"id": i, "respuesta": r, "fichas_prompt": 60, "fichas_generadas": 7, "al_tope": False, "segundos": 0.2}
                    for i, r in zip(ids, respuestas))


@pytest.mark.parametrize("distintas,cumple", [(0, True), (20, True), (21, False)])
def test_compuerta_con_el_criterio_real(distintas, cumple):
    ids = [str(9_000_000 + i) for i in range(1000)]
    ref = jg.jsonl({"id": i, "respuesta": "461110"} for i in ids)
    nuevas = _pred(ids, ["999999"] * distintas + ["461110"] * (1000 - distintas))
    g = fo.comparar_con_referencia(nuevas, ref, fo.criterio_de_paro(CONFIG))
    assert g["identicas"] == 1000 - distintas and g["cumple"] is cumple and g["mismos_ids"]


def test_compuerta_ids_distintos_o_sin_validacion():
    crit = {"filas": 3, "minimo_identicas": 2, "modelos": []}
    ref = jg.jsonl({"id": i, "respuesta": "461110"} for i in ("1", "2", "3"))
    g = fo.comparar_con_referencia(_pred(["1", "2", "4"], ["461110"] * 3), ref, crit)
    assert not g["cumple"] and not g["mismos_ids"] and g["identicas"] == 2
    assert not fo.comparar_con_referencia(_pred(["1", "2"], ["461110"] * 2), ref, crit)["cumple"]
    g = fo.comparar_con_referencia(None, ref, crit)
    assert not g["cumple"] and g["filas_nuevas"] == 0
    assert not fo.comparar_con_referencia(b'{"id": "1"}\n', ref, crit)["cumple"]
    # Las respuestas crudas se comparan tal cual: un espacio de más ya no es idéntica.
    g = fo.comparar_con_referencia(_pred(["1", "2", "3"], ["461110 ", "461110", "461110"]), ref, crit)
    assert g["identicas"] == 2 and g["cumple"]


# --- Entrada ---


def _revisar(j, quitar=(), agregar=None):
    listado = [a for a in fo.kc.listar_entrada(j["entrada"]) if a["ruta"] not in quitar]
    hashes = {a["ruta"]: hk.sha256_archivo(j["entrada"] / a["ruta"]) for a in listado}
    for ruta, datos in (agregar or {}).items():
        listado.append({"ruta": ruta, "bytes": len(datos)})
        hashes[ruta] = jg.sha(datos)
    return fo.revisar_entrada(listado, hashes, fo.validar_config_kernel(j["config"]), j["esperados"], j["config"])


def test_revisar_entrada(tmp_path):
    j = jg.armar(tmp_path)
    rev = _revisar(j)
    assert rev["rutas"]["test.jsonl"] == "denue-scian-evaluacion/test.jsonl"
    assert rev["afinados"]["06b-lr2e-4-r32"] == {
        "adaptador": "denue-scian-06b-barrido/06b-lr2e-4-r32/adaptador/adapter_model.safetensors",
        "adaptador_config": "denue-scian-06b-barrido/06b-lr2e-4-r32/adaptador/adapter_config.json",
        "predicciones_validacion": "denue-scian-06b-barrido/06b-lr2e-4-r32/predicciones.jsonl",
    }
    with pytest.raises(fo.AbortoFase4, match="copias"):
        _revisar(j, quitar=("denue-scian-evaluacion/hard.jsonl",))
    with pytest.raises(fo.AbortoFase4, match="de más"):
        _revisar(j, agregar={"denue-scian-evaluacion/test_con_etiquetas.jsonl": b"x"})
    with pytest.raises(fo.AbortoFase4, match="adapter_model"):
        _revisar(j, quitar=("denue-scian-4b/4b-lr2e-4-r32/adaptador/adapter_model.safetensors",))
    with pytest.raises(fo.AbortoFase4, match="copias"):
        _revisar(j, agregar={"denue-scian-4b/otra/val.jsonl": b"x"})
    datos = b"copia de test con etiquetas"
    j["config"]["conjuntos"]["test"]["sha256"] = jg.sha(datos)
    with pytest.raises(fo.AbortoFase4, match="con etiquetas"):
        _revisar(j, agregar={"denue-scian-4b/sorpresa.bin": datos})


# --- Orquestación en seco, con procesos reales ---

PROCESO_FALSO = textwrap.dedent('''
    """Imita a src/etapa_fase4_kaggle.py: escribe las salidas de cada proceso según MARCAS_FASE4."""
    import argparse, hashlib, json, os, sys, time
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("--comprobaciones")
    p.add_argument("--etapa")
    a = p.parse_args()
    marcas = json.loads(os.environ.get("MARCAS_FASE4", "{}"))
    if a.comprobaciones:
        d = json.loads(Path(a.comprobaciones).read_text(encoding="utf-8"))
        ok = marcas.get("comprobaciones") != "falla"
        Path(d["salida"], "comprobaciones.json").write_text(json.dumps({"ok": ok}), encoding="utf-8")
        print("[fase4-etapa 00:00:01] comprobaciones: " + ("todo pasa" if ok else "FALLAN"), flush=True)
        sys.exit(0 if ok else 1)
    d = json.loads(Path(a.etapa).read_text(encoding="utf-8"))
    et, salida = d["etapa"], Path(d["salida"])
    med = {"estado": "en_curso", "etapa": {"id": et["id"]}}
    ruta_med = salida / "etapas" / (et["id"] + ".json")
    ruta_med.parent.mkdir(parents=True, exist_ok=True)
    if et["id"] in marcas.get("cuelgan", []):
        time.sleep(120)
    if et["id"] in marcas.get("fallan", []):
        med.update(estado="incompleta", motivo="error")
        ruta_med.write_text(json.dumps(med), encoding="utf-8")
        print("[fase4-etapa 00:00:02] " + et["id"] + ": estado incompleta", flush=True)
        sys.exit(1)
    archivo = {"validacion": "val.jsonl", "test": "test.jsonl", "hard": "hard.jsonl"}[et["conjunto"]]
    filas = [json.loads(l) for l in Path(d["rutas"][archivo]).read_text(encoding="utf-8").splitlines()]
    distintas = marcas.get("val_distintas", {}).get(et["modelo"], 0) if et["conjunto"] == "validacion" else 0
    datos = "".join(
        json.dumps({"id": r["id"], "respuesta": "999999" if i < distintas else "467111", "fichas_prompt": 60,
                    "fichas_generadas": 7, "al_tope": False, "segundos": 0.2}) + "\\n"
        for i, r in enumerate(filas)).encode("utf-8")
    pred = salida / "predicciones" / et["modelo"] / (et["conjunto"] + ".jsonl")
    pred.parent.mkdir(parents=True, exist_ok=True)
    pred.write_bytes(datos)
    med.update(estado="[tapado]", completa=True, archivo={"ruta": "x", "bytes": len(datos) - (et["id"] in marcas.get("mienten", [])),
               "sha256": "[tapado]", "filas": len(filas)},
               inferencia={"filas": len(filas), "s_por_fila": 0.2, "al_tope": 0},
               adaptador={"metodo": "[tapado]", "metodo_indice": 1} if et["tipo"] == "afinado" else None,
               metodo_recibido=d["metodo_adaptador"])
    ruta_med.write_text(json.dumps(med), encoding="utf-8")
    print("[fase4-etapa 00:00:03] " + et["id"] + ": " + str(len(filas)) + " filas", flush=True)
''')


def _instaladas(**cambios):
    v = {p.split("==")[0]: p.split("==")[1] for p in bk.PAQUETES_BARRIDO}
    v.update({k: bk.VERSIONES_06B_INICIAL[k] for k in bk.NO_FIJABLES})
    v.update(cambios)
    return v


def _entorno(tmp_path, monkeypatch, marcas=None, *, versiones=None, memoria=0, estimar=None, presupuesto=fo.PRESUPUESTO_S):
    j = jg.armar(tmp_path)
    (j["raiz"] / "proceso_falso_fase4.py").write_text(PROCESO_FALSO, encoding="utf-8")
    monkeypatch.setenv("MARCAS_FASE4", json.dumps(marcas or {}))
    instalados = []
    dep = fo.Dependencias(
        dir_salida=tmp_path / "working" / "fase4",
        dir_trabajo=tmp_path / "trabajo",
        dir_entrada=j["entrada"],
        consultar_gpus=lambda: [{"indice": i, "nombre": "Tesla T4", "memoria_mib": 15360, "driver": bk.CONTROLADOR_06B_INICIAL}
                                for i in (0, 1)],
        instalar=lambda reg, d: instalados.append(d),
        versiones=lambda nombres: versiones or _instaladas(),
        memoria_gpu=lambda: {0: memoria, 1: 0},
        estimar=estimar or (lambda e: 30.0),
        presupuesto_s=presupuesto,
        modulo_etapa="proceso_falso_fase4",
        raiz_fuentes=j["raiz"],
        python=sys.executable,
        espera_s=0.2,
    )
    return j, dep, instalados


def _mediciones(dep):
    return json.loads((dep.dir_salida / fo.ARCHIVO_MEDICIONES).read_bytes())


def _estados(m):
    return {e["id"]: e["estado"] for e in m["etapas"]}


def test_orquestacion_normal(tmp_path, monkeypatch, capsys):
    j, dep, instalados = _entorno(tmp_path, monkeypatch)
    fo.main(j["esperados"], None, j["fuentes"], dep=dep)
    m = _mediciones(dep)
    assert m["estado"] == "completo" and m["fallas"] == [] and instalados == [dep.dir_salida]
    assert list(_estados(m).values()) == ["completa"] * 14
    assert [e["id"] for e in m["etapas"]] == [e["id"] for e in m["plan"]]
    assert m["config"]["sha256"] == j["fuentes"][fo.RUTA_CONFIG]
    for mid in ("06b-lr2e-4-r32", "4b-lr2e-4-r32"):
        g = m["reproduccion_validacion"][mid]
        assert g["cumple"] and g["identicas"] == g["filas_referencia"] == jg.FILAS["validacion"]
        assert g["metodo_adaptador"] == "peft_from_pretrained"
    por_id = {e["id"]: e for e in m["etapas"]}
    assert por_id["06b-lr2e-4-r32__validacion"]["medicion"]["metodo_recibido"] is None
    assert por_id["06b-lr2e-4-r32__test"]["medicion"]["metodo_recibido"] == "peft_from_pretrained"
    assert por_id["06b-base-0shot__test"]["medicion"]["metodo_recibido"] is None
    rutas = {a["ruta"] for a in m["archivos_salida"]}
    assert "predicciones/4b-base-fewshot/hard.jsonl" in rutas and "logs/comprobaciones.log" in rutas
    assert por_id["4b-base-fewshot__hard"]["predicciones"]["filas"] == jg.FILAS["hard"]
    assert m["entrada"]["archivos_revisados"] == len(m["entrada"]["listado"]) and m["entrada"]["con_sha256_de_etiquetas"] == 0
    texto = (dep.dir_salida / fo.ARCHIVO_MEDICIONES).read_text(encoding="utf-8") + capsys.readouterr().out
    assert not any(t in texto for t in jg.textos_privados()) and jg.RESPUESTA_REFERENCIA not in texto


@pytest.mark.parametrize("distintas,cumple", [(1, True), (2, False)])
def test_compuerta_de_validacion(tmp_path, monkeypatch, distintas, cumple):
    j, dep, _ = _entorno(tmp_path, monkeypatch, {"val_distintas": {"06b-lr2e-4-r32": distintas}})
    fo.main(j["esperados"], None, j["fuentes"], dep=dep)
    m = _mediciones(dep)
    e = _estados(m)
    g = m["reproduccion_validacion"]["06b-lr2e-4-r32"]
    assert g["identicas"] == jg.FILAS["validacion"] - distintas and g["cumple"] is cumple
    esperado = "completa" if cumple else "no_corre"
    assert e["06b-lr2e-4-r32__test"] == e["06b-lr2e-4-r32__hard"] == esperado
    assert e["4b-lr2e-4-r32__test"] == "completa" and e["4b-base-fewshot__hard"] == "completa"
    assert (dep.dir_salida / "predicciones" / "06b-lr2e-4-r32" / "test.jsonl").exists() is cumple
    assert ("reproduccion_06b-lr2e-4-r32" in [f["etapa"] for f in m["fallas"]]) is not cumple


def test_una_etapa_que_falla_o_miente_no_detiene_las_demas(tmp_path, monkeypatch):
    marcas = {"fallan": ["06b-base-0shot__test", "4b-lr2e-4-r32__validacion"], "mienten": ["4b-base-0shot__hard"]}
    j, dep, _ = _entorno(tmp_path, monkeypatch, marcas)
    fo.main(j["esperados"], None, j["fuentes"], dep=dep)
    m = _mediciones(dep)
    e = _estados(m)
    assert e["06b-base-0shot__test"] == "incompleta" and e["06b-base-0shot__hard"] == "completa"
    assert e["4b-lr2e-4-r32__validacion"] == "incompleta" and e["4b-lr2e-4-r32__test"] == "no_corre"
    assert not m["reproduccion_validacion"]["4b-lr2e-4-r32"]["cumple"]
    assert e["4b-base-0shot__hard"] == "incompleta" and e["4b-base-fewshot__hard"] == "completa"
    assert not (dep.dir_salida / "predicciones" / "4b-base-0shot" / "hard.jsonl").exists()  # se descartó
    assert (dep.dir_trabajo / fo.CARPETA_DESCARTADAS / "4b-base-0shot" / "hard.jsonl").exists()
    assert m["estado"] == "completo_con_fallas" and sum(v == "completa" for v in e.values()) == 9


def test_un_cuelgue_se_mata_al_pasar_su_tiempo_maximo(tmp_path, monkeypatch):
    j, dep, _ = _entorno(tmp_path, monkeypatch, {"cuelgan": ["06b-base-0shot__hard"]},
                         estimar=lambda e: 1.0 if e["id"] == "06b-base-0shot__hard" else 30.0)
    fo.main(j["esperados"], None, j["fuentes"], dep=dep)
    m = _mediciones(dep)
    r = {e["id"]: e for e in m["etapas"]}["06b-base-0shot__hard"]
    assert r["estado"] == "incompleta" and r["matado_por_tiempo"] and r["tiempo_maximo_s"] == 2.0 and r["segundos"] < 60
    assert r["limite_de_generacion_s"] == 1.0
    assert _estados(m)["4b-base-0shot__test"] == "completa"


def test_candado_de_tiempo_omite_etapas(tmp_path, monkeypatch):
    j, dep, _ = _entorno(tmp_path, monkeypatch, estimar=lambda e: 100.0, presupuesto=50.0)
    fo.main(j["esperados"], None, j["fuentes"], dep=dep)
    e = _estados(_mediciones(dep))
    assert e["06b-lr2e-4-r32__validacion"] == "omitida" and e["06b-lr2e-4-r32__test"] == "no_corre"
    assert e["4b-base-fewshot__test"] == "omitida"
    assert not (dep.dir_salida / fo.CARPETA_PREDICCIONES).exists()


def test_memoria_no_liberada_se_registra(tmp_path, monkeypatch):
    j, dep, _ = _entorno(tmp_path, monkeypatch, memoria=2000)
    fo.main(j["esperados"], None, j["fuentes"], dep=dep)
    m = _mediciones(dep)
    assert "memoria_06b-base-0shot__test" in [f["etapa"] for f in m["fallas"]] and m["estado"] == "completo_con_fallas"


def test_comprobaciones_que_fallan_abortan_sin_etapas(tmp_path, monkeypatch):
    j, dep, instalados = _entorno(tmp_path, monkeypatch, {"comprobaciones": "falla"})
    with pytest.raises(SystemExit) as e:
        fo.main(j["esperados"], None, j["fuentes"], dep=dep)
    assert e.value.code == 1
    m = _mediciones(dep)
    assert m["estado"] == "abortado" and "comprobaciones" in m["aborto"] and "etapas" not in m
    assert instalados == [dep.dir_salida] and not (dep.dir_salida / fo.CARPETA_ETAPAS).exists()


@pytest.mark.parametrize("como", ["hash_de_datos", "config", "etiquetas", "version"])
def test_aborta_antes_de_instalar_o_de_cargar(tmp_path, monkeypatch, como):
    j, dep, instalados = _entorno(tmp_path, monkeypatch, versiones=_instaladas(xformers="0.0.35") if como == "version" else None)
    esperados, fuentes = dict(j["esperados"]), dict(j["fuentes"])
    if como == "hash_de_datos":
        esperados["val.jsonl"] = "0" * 64
    elif como == "config":
        fuentes[fo.RUTA_CONFIG] = "1" * 64
    elif como == "etiquetas":
        datos = b"una copia con codigo_act"
        (j["entrada"] / "denue-scian-4b" / "extra.jsonl").write_bytes(datos)
        config = copy.deepcopy(j["config"])
        config["conjuntos"]["hard"]["sha256"] = jg.sha(datos)
        fuentes[fo.RUTA_CONFIG] = jg.sha(jg.escribir_config(j["raiz"], config))
    with pytest.raises(SystemExit):
        fo.main(esperados, None, fuentes, dep=dep)
    m = _mediciones(dep)
    assert m["estado"] == "abortado" and not (dep.dir_salida / fo.CARPETA_LOGS).exists()
    assert instalados == ([dep.dir_salida] if como == "version" else [])


def test_esperados_deben_ser_los_cuatro_archivos(tmp_path, monkeypatch):
    j, dep, _ = _entorno(tmp_path, monkeypatch)
    with pytest.raises(fo.AbortoFase4):
        fo.main({k: v for k, v in j["esperados"].items() if k != "hard.jsonl"}, None, j["fuentes"], dep=dep)


# --- Aislamiento del orquestador ---


def test_el_orquestador_no_importa_torch_ni_numpy_ni_lee_nombres():
    arbol = ast.parse((REPO_RAIZ / "src" / "inferencia_fase4_kaggle.py").read_text(encoding="utf-8"))
    importados = set()
    for n in ast.walk(arbol):
        if isinstance(n, ast.Import):
            importados |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            importados.add(n.module)
    raices = {m.split(".")[0] for m in importados}
    assert not raices & {"torch", "transformers", "unsloth", "unsloth_zoo", "trl", "peft", "datasets", "numpy"}
    assert not importados & {"src.preregistro_fase4", "src.plantilla_fase4", "src.etapa_fase4_kaggle"}
    assert not any(isinstance(n, ast.Constant) and n.value == "nom_estab" for n in ast.walk(arbol))


def test_cli_estimar(capsys):
    fo._cli(["--estimar"])
    salida = capsys.readouterr().out
    assert "4b-base-fewshot__test" in salida and "Total: mejor" in salida and "980 de 1000" in salida
