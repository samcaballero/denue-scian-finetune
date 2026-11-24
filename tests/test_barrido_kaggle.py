"""Pruebas del kernel del barrido (src/barrido_kaggle.py), en CPU.

Usan configuraciones reales del generador, datos inventados y un proceso falso en lugar de
`clasificador_kaggle.main()`: nunca leen el DENUE ni corren torch.
"""

import ast
import hashlib
import json
import sys
import textwrap
from pathlib import Path

import pytest

from src import barrido_kaggle as bk
from src import construir_kernel as ck
from src import humo_kaggle as hk
from src.denue import REPO_RAIZ
from src.evaluar_corrida_clasificador import validacion_del_log

RESULTADO_06B = REPO_RAIZ / "results" / "2026-11-23_fase2_06b-inicial.json"
CORRIDAS = [dict(c) for c in ck.CORRIDAS_BARRIDO_06B]


@pytest.fixture(scope="module")
def resultado_06b():
    return json.loads(RESULTADO_06B.read_bytes())


# --- Constantes contra el resultado de 06b-inicial (única fuente) ---


def test_versiones_y_orden_salen_del_resultado_de_06b(resultado_06b):
    pb = resultado_06b["para_el_barrido"]
    assert bk.VERSIONES_06B_INICIAL == pb["versiones_entorno"] == pb["pip_freeze_despues"]
    assert bk.SHA256_ORDEN_06B_INICIAL == pb["sha256_orden_train"]


def test_pins_extra_son_las_versiones_de_06b(resultado_06b):
    pb = resultado_06b["para_el_barrido"]
    cambios = pb["pip_cambio_al_instalar"]
    conocidas = {k.replace("_", "-"): v for k, v in pb["versiones_entorno"].items()}
    conocidas.update(cambios["agregados"])
    conocidas.update({k: v[1] for k, v in cambios["cambiados"].items()})
    for p in bk.PAQUETES_EXTRA:
        nombre, version = p.split("==")
        assert conocidas[nombre.replace("_", "-")] == version, p
    # Todo lo que pip agregó o cambió en 06b-inicial queda fijado.
    fijados = {p.split("==")[0].replace("_", "-") for p in bk.PAQUETES_BARRIDO}
    assert set(cambios["agregados"]) | set(cambios["cambiados"]) <= fijados
    assert "xformers==0.0.36" in bk.PAQUETES_EXTRA


def test_paquetes_del_barrido_incluyen_los_de_06b_sin_cambios():
    assert bk.PAQUETES_BARRIDO[: len(hk.PAQUETES)] == tuple(hk.PAQUETES)
    nombres = [p.split("==")[0].replace("_", "-").lower() for p in bk.PAQUETES_BARRIDO]
    assert len(nombres) == len(set(nombres))
    assert not set(bk.NO_FIJABLES) & set(nombres)


def test_tiempos_salen_del_resultado_de_06b(resultado_06b):
    d = resultado_06b["corrida_kaggle"]["tiempo"]["duraciones_etapas_s"]
    antes = sum(d[k] for k in ("datos", "instalacion", "imports", "descarga", "carga", "tokens_y_plantilla", "lora"))
    assert bk.S_ANTES_DE_ENTRENAR_06B == pytest.approx(antes, abs=1e-3)
    assert bk.S_ENTRENAMIENTO_06B == d["entrenamiento"]
    assert bk.S_DESPUES_DE_ENTRENAR_06B == pytest.approx(d["adaptador"] + d["inferencia"] + d["cierre"], abs=1e-3)
    assert bk.FILAS_TRAIN_06B == resultado_06b["tamano_muestra"]["train"]
    assert bk.PASOS_06B == resultado_06b["etapas"]["etapas"]["entrenamiento"]["pasos"]


def test_estimaciones_y_tiempo_de_kaggle():
    por = {c["nombre"]: bk.estimacion_s(c) for c in CORRIDAS}
    assert por["06b-lr1e-4-r8"] == por["06b-lr2e-4-r8"] < por["06b-lr1e-4-r32"] == por["06b-lr2e-4-r32"]
    total = bk.estimacion_total_s(CORRIDAS)
    assert 1.4 * total <= bk.TIEMPO_KAGGLE_S <= 1.6 * total
    assert bk.TIEMPO_KAGGLE_S <= 12 * 3600
    assert bk.PRESUPUESTO_S == bk.TIEMPO_KAGGLE_S - bk.MARGEN_CIERRE_S
    # Con las estimaciones, las 4 caben con el candado de tiempo.
    t = bk.S_ORQUESTADOR
    for c in CORRIDAS:
        assert bk.decidir(bk.estimacion_s(c), t)["lanzar"]
        t += bk.estimacion_s(c)


# --- Funciones puras ---


def test_decidir():
    d = bk.decidir(100, 0, presupuesto_s=1000)
    assert d == {"lanzar": True, "tiempo_maximo_s": 200.0}
    assert bk.decidir(100, 850, presupuesto_s=1000)["tiempo_maximo_s"] == 150.0  # no pasa del presupuesto
    assert bk.decidir(100, 871, presupuesto_s=1000)["lanzar"] is False


def _instaladas(**cambios):
    v = {p.split("==")[0]: p.split("==")[1] for p in bk.PAQUETES_BARRIDO}
    v.update({k: bk.VERSIONES_06B_INICIAL[k] for k in bk.NO_FIJABLES})
    v.update(cambios)
    return v


def test_revisar_versiones():
    r = bk.revisar_versiones(_instaladas(), bk.CONTROLADOR_06B_INICIAL)
    assert r["fijadas_distintas"] == {} and r["advertencias"] == {}
    r = bk.revisar_versiones(_instaladas(xformers="0.0.35"), bk.CONTROLADOR_06B_INICIAL)
    assert list(r["fijadas_distintas"]) == ["xformers"]
    r = bk.revisar_versiones(_instaladas(torch="2.12.0+cu130", triton=None), "590.0")
    assert r["fijadas_distintas"] == {}
    assert set(r["advertencias"]) == {"torch", "triton", "controlador_nvidia"}


def test_parsear_memoria():
    assert bk.parsear_memoria("0, 3\n1, 1500\n") == {0: 3, 1: 1500}
    assert bk.parsear_memoria("basura\n") == {}


def test_resumen_de_mediciones(tmp_path):
    assert bk.resumen_de_mediciones(tmp_path / "no.json") == {"existe": False}
    (tmp_path / "m.json").write_text("{malo", encoding="utf-8")
    assert bk.resumen_de_mediciones(tmp_path / "m.json")["legible"] is False
    m = {"estado": "completo", "fallas": [{"etapa": "x", "error": "texto que no se copia"}], "aborto": "no se copia",
         "entrenamiento": {"pasos_hechos": 5855, "orden_train": {"sha256_orden": bk.SHA256_ORDEN_06B_INICIAL}},
         "inferencia": {"filas": 1000}}
    (tmp_path / "m.json").write_text(json.dumps(m), encoding="utf-8")
    r = bk.resumen_de_mediciones(tmp_path / "m.json")
    assert r["orden_igual_al_de_06b_inicial"] and r["etapas_con_falla"] == ["x"] and r["aborto_o_error"]
    assert "texto que no se copia" not in json.dumps(r) and "no se copia" not in json.dumps(r)


def test_validar_corridas():
    assert [c["nombre"] for c in bk.validar_corridas(CORRIDAS)] == [c["nombre"] for c in CORRIDAS]
    with pytest.raises(bk.AbortoBarrido, match="distintos"):
        bk.validar_corridas([CORRIDAS[0], CORRIDAS[0]])
    with pytest.raises(bk.AbortoBarrido, match="carpeta"):
        bk.validar_corridas([dict(CORRIDAS[0], nombre="../otro")])
    with pytest.raises(ValueError):
        bk.validar_corridas([dict(CORRIDAS[0], semilla=7)])


def test_el_orquestador_no_importa_torch_ni_lee_el_dataset_por_filas():
    arbol = ast.parse((REPO_RAIZ / "src" / "barrido_kaggle.py").read_text(encoding="utf-8"))
    importados = set()
    for n in ast.walk(arbol):
        if isinstance(n, ast.Import):
            importados |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            importados.add(n.module.split(".")[0])
    assert not importados & {"torch", "transformers", "unsloth", "unsloth_zoo", "trl", "peft", "datasets"}
    nodos = list(ast.walk(arbol))
    assert not any(isinstance(n, ast.Constant) and n.value == "nom_estab" for n in nodos)
    assert not any(isinstance(n, ast.Attribute) and n.attr == "leer_jsonl_verificado" for n in nodos)
    texto = (REPO_RAIZ / "src" / "barrido_kaggle.py").read_text(encoding="utf-8")
    assert "test.jsonl" not in texto and "hard.jsonl" not in texto


def test_correr_una_usa_clasificador_kaggle_sin_cambios(tmp_path, monkeypatch):
    llamadas = []
    monkeypatch.setattr(bk.kc, "main", lambda *a: llamadas.append((a, hk.DIR_SALIDA, hk.DIR_TRABAJO)))
    monkeypatch.setattr(hk, "DIR_SALIDA", hk.DIR_SALIDA)
    monkeypatch.setattr(hk, "DIR_TRABAJO", hk.DIR_TRABAJO)
    entrada = tmp_path / "e.json"
    entrada.write_text(json.dumps({"corrida": CORRIDAS[0], "esperados": {"a": "b"}, "fuentes": {"f": "s"},
                                   "salida": str(tmp_path / "s"), "trabajo": str(tmp_path / "t")}), encoding="utf-8")
    bk.correr_una(entrada)
    (esperados, _t, fuentes, corrida), salida, trabajo = llamadas[0]
    assert (esperados, fuentes, corrida) == ({"a": "b"}, {"f": "s"}, CORRIDAS[0])
    assert (salida, trabajo) == (tmp_path / "s", tmp_path / "t")


# --- Prueba en seco de la orquestación, con procesos reales ---

PROCESO_FALSO = textwrap.dedent('''
    """Imita a clasificador_kaggle.main() en un proceso: escribe la salida y algunas líneas del log."""
    import argparse, json, sys, time
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("--una")
    d = json.loads(Path(p.parse_args().una).read_text(encoding="utf-8"))
    c = d["corrida"]
    salida = Path(d["salida"])
    print("[clasificador 00:00:01] inicio: gpu", flush=True)
    if "CUELGA" in c["descripcion"]:
        time.sleep(120)
    if "FALLA" in c["descripcion"]:
        print("[clasificador 00:00:02] ABORTO: algo falló", flush=True)
        sys.exit(1)
    salida.mkdir(parents=True)
    print("[clasificador 00:00:03] pérdida de validación en el paso 0: 3.25", flush=True)
    print("[clasificador 00:10:00] pérdida de validación en el paso 500: 0.5", flush=True)
    orden = d["fuentes"].get("orden", "0" * 64)
    m = {"estado": "completo", "fallas": [], "entrenamiento": {"pasos_hechos": 5855, "orden_train": {"sha256_orden": orden}},
         "inferencia": {"filas": 1000}, "segundos_desde_arranque": 1.0}
    (salida / "mediciones.json").write_text(json.dumps(m), encoding="utf-8")
    (salida / "predicciones.jsonl").write_text('{"id": "1", "respuesta": "461110"}\\n', encoding="utf-8")
''')


def _entorno(tmp_path, *, versiones=None, memoria=0, estimar=lambda c: 30.0, presupuesto=10_000.0):
    raiz = tmp_path / "fuentes"
    raiz.mkdir()
    (raiz / "proceso_falso.py").write_text(PROCESO_FALSO, encoding="utf-8")
    entrada = tmp_path / "input" / "denue-scian-clasificador"
    entrada.mkdir(parents=True)
    esperados = {}
    for nombre in ("train_final.jsonl", "val.jsonl"):
        datos = f'{{"id": "1", "nom_estab": "Negocio inventado", "codigo_act": "461110", "archivo": "{nombre}"}}\n'.encode()
        (entrada / nombre).write_bytes(datos)
        esperados[nombre] = hashlib.sha256(datos).hexdigest()
    instalados = []
    dep = bk.Dependencias(
        dir_salida=tmp_path / "working",
        dir_trabajo=tmp_path / "trabajo",
        dir_entrada=tmp_path / "input",
        consultar_gpus=lambda: [{"indice": i, "nombre": "Tesla T4", "memoria_mib": 15360, "driver": bk.CONTROLADOR_06B_INICIAL}
                                for i in (0, 1)],
        instalar=lambda reg, d: instalados.append(d),
        versiones=lambda nombres: versiones or _instaladas(),
        memoria_gpu=lambda: {0: memoria, 1: 0},
        estimar=estimar,
        presupuesto_s=presupuesto,
        modulo_una="proceso_falso",
        raiz_fuentes=raiz,
        python=sys.executable,
        espera_s=0.2,
    )
    return dep, esperados, instalados


def _barrido(dep):
    return json.loads((dep.dir_salida / bk.ARCHIVO_BARRIDO).read_bytes())


def _corridas(*marcas):
    return [dict(c, descripcion=c["descripcion"] + (f" {m}" if m else "")) for c, m in zip(CORRIDAS, marcas)]


def test_orquestacion_normal(tmp_path, monkeypatch):
    monkeypatch.setattr(bk, "SHA256_ORDEN_06B_INICIAL", "0" * 64)
    dep, esperados, instalados = _entorno(tmp_path)
    bk.main(esperados, None, {"orden": "0" * 64}, CORRIDAS, dep=dep)
    b = _barrido(dep)
    assert b["estado"] == "completo" and b["fallas"] == [] and instalados == [dep.dir_salida]
    assert [c["estado"] for c in b["configuraciones"]] == ["completa"] * 4
    for c in b["configuraciones"]:
        assert c["memoria_liberada"] and c["mediciones"]["orden_igual_al_de_06b_inicial"]
        log = (dep.dir_salida / c["log"]).read_text(encoding="utf-8")
        assert validacion_del_log(log) == {0: 3.25, 500: 0.5}  # el paso 0 queda en el log de cada configuración
        assert (dep.dir_salida / c["carpeta"] / "mediciones.json").is_file()
    rutas = {a["ruta"] for a in b["archivos_salida"]}
    assert {f"{c['nombre']}/predicciones.jsonl" for c in CORRIDAS} <= rutas
    assert {f"logs/{c['nombre']}.log" for c in CORRIDAS} <= rutas and bk.ARCHIVO_BARRIDO not in rutas
    assert b["versiones"]["advertencias"] == {}


def test_una_configuracion_que_falla_no_detiene_las_demas(tmp_path, monkeypatch):
    monkeypatch.setattr(bk, "SHA256_ORDEN_06B_INICIAL", "0" * 64)
    dep, esperados, _ = _entorno(tmp_path)
    bk.main(esperados, None, {"orden": "0" * 64}, _corridas("", "FALLA", "", ""), dep=dep)
    b = _barrido(dep)
    assert [c["estado"] for c in b["configuraciones"]] == ["completa", "fallida", "completa", "completa"]
    assert b["configuraciones"][1]["codigo_de_salida"] == 1
    assert b["estado"] == "completo_con_fallas" and [f["etapa"] for f in b["fallas"]] == ["configuracion_06b-lr1e-4-r32"]


def test_un_cuelgue_se_mata_al_pasar_su_tiempo_maximo(tmp_path, monkeypatch):
    monkeypatch.setattr(bk, "SHA256_ORDEN_06B_INICIAL", "0" * 64)
    dep, esperados, _ = _entorno(tmp_path, estimar=lambda c: 1.0 if "CUELGA" in c["descripcion"] else 30.0)
    bk.main(esperados, None, {"orden": "0" * 64}, _corridas("CUELGA", "", "", ""), dep=dep)
    b = _barrido(dep)
    c0 = b["configuraciones"][0]
    assert c0["matado_por_tiempo"] and c0["tiempo_maximo_s"] == 2.0 and c0["segundos"] < 60
    assert [c["estado"] for c in b["configuraciones"]] == ["fallida", "completa", "completa", "completa"]
    assert "tiempo máximo" in b["fallas"][0]["error"]


def test_candado_de_tiempo_omite_configuraciones(tmp_path):
    dep, esperados, _ = _entorno(tmp_path, estimar=lambda c: 100.0, presupuesto=50.0)
    bk.main(esperados, None, {}, CORRIDAS, dep=dep)
    b = _barrido(dep)
    assert [c["estado"] for c in b["configuraciones"]] == ["omitida"] * 4
    assert not (dep.dir_salida / bk.CARPETA_LOGS).exists()
    assert b["estado"] == "completo_con_fallas"


def test_memoria_no_liberada_y_orden_distinto_se_registran(tmp_path):
    dep, esperados, _ = _entorno(tmp_path, memoria=2000)
    bk.main(esperados, None, {"orden": "1" * 64}, CORRIDAS[:1], dep=dep)
    b = _barrido(dep)
    etapas = [f["etapa"] for f in b["fallas"]]
    assert "memoria_06b-lr1e-4-r8" in etapas and "orden_06b-lr1e-4-r8" in etapas
    assert b["configuraciones"][0]["memoria_liberada"] is False


def test_version_fijada_distinta_aborta_antes_de_entrenar(tmp_path):
    dep, esperados, _ = _entorno(tmp_path, versiones=_instaladas(xformers="0.0.35"))
    with pytest.raises(SystemExit) as e:
        bk.main(esperados, None, {}, CORRIDAS, dep=dep)
    assert e.value.code == 1
    b = _barrido(dep)
    assert b["estado"] == "abortado" and "xformers" in b["aborto"]
    assert "configuraciones" not in b and not (dep.dir_salida / bk.CARPETA_LOGS).exists()


def test_version_no_fijable_distinta_solo_advierte(tmp_path, monkeypatch):
    monkeypatch.setattr(bk, "SHA256_ORDEN_06B_INICIAL", "0" * 64)
    dep, esperados, _ = _entorno(tmp_path, versiones=_instaladas(torch="2.12.0+cu130"))
    bk.main(esperados, None, {"orden": "0" * 64}, CORRIDAS[:1], dep=dep)
    b = _barrido(dep)
    assert b["estado"] == "completo" and set(b["advertencias"]) == {"torch"}


def test_datos_con_otro_hash_abortan_antes_de_instalar(tmp_path):
    dep, esperados, instalados = _entorno(tmp_path)
    with pytest.raises(SystemExit):
        bk.main({**esperados, "val.jsonl": "0" * 64}, None, {}, CORRIDAS, dep=dep)
    assert _barrido(dep)["estado"] == "abortado" and instalados == []


def test_el_kernel_generado_extrae_e_importa_el_orquestador_aislado(tmp_path):
    """Los módulos incrustados bastan para el orquestador y para la entrada --una de cada proceso."""
    import os
    import subprocess

    from src.scian import leer_manifest

    espec = ck.KERNELS["06b-barrido"]
    texto = ck.generar(leer_manifest(), espec=espec)[espec.archivo_kernel].decode("utf-8")
    ns = {"__name__": "kernel_en_prueba"}
    exec(compile(texto, espec.archivo_kernel, "exec"), ns)
    raiz = ns["extraer"](tmp_path / "denue_src")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(raiz)
    codigo = "import src.barrido_kaggle as b, sys; print(b.ARCHIVO_BARRIDO, b.RAIZ_FUENTES == __import__('pathlib').Path(sys.argv[1]).resolve())"
    r = subprocess.run([sys.executable, "-c", codigo, str(raiz)], env=env, cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["barrido.json", "True"]
    r = subprocess.run([sys.executable, "-m", "src.barrido_kaggle", "--help"], env=env, cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0 and "--una" in r.stdout
