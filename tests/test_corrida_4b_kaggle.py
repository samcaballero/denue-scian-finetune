"""Pruebas del kernel del 4B (src/corrida_4b_kaggle.py y espec_4b de src/construir_kernel.py), en CPU.

En la Parte A del paso 12 la ganadora del barrido aún no se conoce: las pruebas usan la
configuración inicial (tasa 2e-4, rank 16, alpha 16). Nunca leen el DENUE ni corren torch.
"""

import ast
import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from src import barrido_kaggle as bk
from src import clasificador_kaggle as kc
from src import construir_kernel as ck
from src import corrida_4b_kaggle as k4
from src import humo_kaggle as hk
from src.denue import REPO_RAIZ
from src.scian import leer_manifest

RESULTADO_HUMO = REPO_RAIZ / "results" / "2026-11-21_fase2_prueba-humo.json"
RESULTADO_06B = REPO_RAIZ / "results" / "2026-11-23_fase2_06b-inicial.json"
ESPEC = ck.espec_4b("2e-4", 2e-4, 16)
CORRIDA = dict(ESPEC.corridas[0])


@pytest.fixture(scope="module")
def manifest():
    return leer_manifest()


@pytest.fixture(scope="module")
def generados(manifest):
    return ck.generar(manifest, espec=ESPEC)


# --- Constantes contra los resultados (única fuente) ---


def test_mediciones_salen_de_los_resultados():
    humo = json.loads(RESULTADO_HUMO.read_bytes())
    rend = {(r["modelo"], r["lote"]): r for r in humo["rendimiento"]}
    assert k4.S_PASO_06B_HUMO == rend[("Qwen/Qwen3-0.6B", 32)]["s_por_paso_media"]
    assert k4.S_PASO_4B_HUMO == rend[("Qwen/Qwen3-4B", 32)]["s_por_paso_media"]
    d4 = humo["etapas"]["etapas"]["modelo_4b"]["duraciones_s"]
    assert k4.S_DESCARGA_CARGA_4B == pytest.approx(d4["descarga"] + d4["carga"])
    r06 = json.loads(RESULTADO_06B.read_bytes())
    dur = r06["corrida_kaggle"]["tiempo"]["duraciones_etapas_s"]
    assert k4.S_DESCARGA_CARGA_06B == pytest.approx(dur["descarga"] + dur["carga"])
    ent = r06["etapas"]["etapas"]["entrenamiento"]
    assert k4.S_VALIDACIONES_06B == pytest.approx(ent["segundos_evaluacion_paso_0"] + sum(e["s"] for e in ent["evaluaciones_s"]))
    assert len(ent["evaluaciones_s"]) + 1 == 13


def test_lote_32_del_4b_cabe_en_la_t4():
    """Lo medido en el paso 6: el lote 32 del 4B no se quedó sin memoria y usó un tercio de la T4."""
    humo = json.loads(RESULTADO_HUMO.read_bytes())
    r = next(x for x in humo["rendimiento"] if x["modelo"] == "Qwen/Qwen3-4B" and x["lote"] == 32)
    total = humo["etapas"]["etapas"]["gpu"]["memoria_total_mib_torch"]
    assert r["sin_memoria"] is False and r["vram_pico_mib"]["reservada"] / total < 0.35
    assert 187_350 / r["ejemplos_por_s"] / 3600 == pytest.approx(6.11, abs=0.005)  # las ~6.1 h por época del paso 8


# --- Estimación y tiempo de Kaggle ---


@pytest.mark.parametrize("rank, total_h", [(8, 6.95), (16, 6.95), (32, 7.27)])
def test_estimacion_por_rank(rank, total_h):
    c = ck.corrida_4b("2e-4", 2e-4, rank)
    d = k4.desglose_s(c)
    assert k4.estimacion_s(c) == pytest.approx(sum(d.values()), abs=0.1)
    assert k4.estimacion_total_s([c]) / 3600 == pytest.approx(total_h, abs=0.005)
    assert d["antes_de_entrenar"] == pytest.approx(292.7, abs=0.1)
    assert d["adaptador_inferencia_y_cierre"] == pytest.approx(1363.1, abs=0.1)
    assert k4.RAZON_4B_06B == pytest.approx(5.977, abs=0.001)


def test_tiempo_de_kaggle_y_candado():
    assert k4.TIEMPO_KAGGLE_S <= k4.LIMITE_SESION_KAGGLE_S
    assert k4.PRESUPUESTO_S == k4.TIEMPO_KAGGLE_S - k4.MARGEN_CIERRE_S
    for rank in (8, 16, 32):
        c = ck.corrida_4b("1e-4", 1e-4, rank)
        assert k4.TIEMPO_KAGGLE_S >= 1.5 * k4.estimacion_total_s([c])
        d = bk.decidir(k4.estimacion_s(c), bk.S_ORQUESTADOR, k4.PRESUPUESTO_S)
        assert d["lanzar"] is True
        assert d["tiempo_maximo_s"] == pytest.approx(k4.PRESUPUESTO_S - bk.S_ORQUESTADOR)


def test_validar_corridas_4b():
    assert k4.validar_corridas_4b([CORRIDA]) == [CORRIDA]
    with pytest.raises(bk.AbortoBarrido, match="exactamente una"):
        k4.validar_corridas_4b([CORRIDA, dict(CORRIDA, nombre="4b-otra")])
    with pytest.raises(bk.AbortoBarrido, match="Qwen3-4B"):
        k4.validar_corridas_4b([dict(ck.CORRIDAS_BARRIDO_06B[0])])
    with pytest.raises(bk.AbortoBarrido, match="alpha"):
        k4.validar_corridas_4b([dict(CORRIDA, lora={**CORRIDA["lora"], "lora_alpha": 32})])
    with pytest.raises(bk.AbortoBarrido, match="lote"):
        k4.validar_corridas_4b([dict(CORRIDA, lote=16)])
    with pytest.raises(bk.AbortoBarrido, match="tasa"):
        k4.validar_corridas_4b([dict(CORRIDA, learning_rate=5e-4)])


def test_candado_de_tiempo_rechaza_un_presupuesto_corto(monkeypatch):
    monkeypatch.setattr(k4, "PRESUPUESTO_S", 20_000)
    with pytest.raises(bk.AbortoBarrido, match="candado"):
        k4.validar_corridas_4b([CORRIDA])


# --- Corrida y especificación ---


def _diferencias(a, b, prefijo=""):
    res = set()
    for k in set(a) | set(b):
        ruta = f"{prefijo}{k}"
        if isinstance(a.get(k), dict) and isinstance(b.get(k), dict):
            res |= _diferencias(a[k], b[k], ruta + ".")
        elif a.get(k) != b.get(k) or type(a.get(k)) is not type(b.get(k)):
            res.add(ruta)
    return res


def test_la_inicial_en_el_4b_difiere_de_06b_inicial_solo_en_nombre_y_modelo():
    assert _diferencias(CORRIDA, dict(ck.CLASIFICADOR_06B_INICIAL.corrida)) == {
        "nombre", "descripcion", "modelo.repo", "modelo.revision",
    }
    assert CORRIDA["modelo"] == hk.MODELO_4B and CORRIDA["nombre"] == "4b-lr2e-4-r16"


@pytest.mark.parametrize("etiqueta, tasa, rank", [("1e-4", 1e-4, 8), ("2e-4", 2e-4, 32), ("1e-4", 1e-4, 32)])
def test_cada_ganadora_posible_difiere_de_06b_inicial_solo_en_tasa_rank_alpha_nombre_y_modelo(etiqueta, tasa, rank):
    c = ck.corrida_4b(etiqueta, tasa, rank)
    permitidas = {"nombre", "descripcion", "modelo.repo", "modelo.revision", "learning_rate", "lora.r", "lora.lora_alpha"}
    assert _diferencias(c, dict(ck.CLASIFICADOR_06B_INICIAL.corrida)) <= permitidas
    assert c["lora"]["r"] == c["lora"]["lora_alpha"] == rank and c["learning_rate"] == tasa


def test_etiqueta_de_tasa_distinta_no_genera():
    with pytest.raises(ck.ErrorKernel, match="etiqueta"):
        ck.corrida_4b("1e-4", 2e-4, 16)


def test_el_4b_no_esta_registrado_en_la_parte_a():
    assert "4b" not in ck.KERNELS


def test_kernel_del_4b_lleva_una_corrida_y_sus_datos(generados, manifest):
    texto = generados[ESPEC.archivo_kernel].decode("utf-8")
    ns = {"__name__": "kernel_en_prueba"}
    exec(compile(texto, ESPEC.archivo_kernel, "exec"), ns)
    archivos = manifest["kaggle_datasets"]["denue_scian_clasificador"]["archivos"]
    assert ns["ESPERADOS"] == {a: archivos[a]["sha256"] for a in ("train_final.jsonl", "val.jsonl")}
    assert ns["CORRIDAS"] == [CORRIDA]
    assert "main(ESPERADOS, T_ARRANQUE, SHA256_FUENTES, CORRIDAS)" in texto
    assert "from src.corrida_4b_kaggle import main" in texto
    assert "una sola, con el orquestador del barrido" in texto
    assert set(ns["FUENTES"]) == {*ck.BARRIDO_06B.fuentes, "src/corrida_4b_kaggle.py"}


def test_candado_del_kernel_del_4b(generados, manifest):
    kernel = generados[ESPEC.archivo_kernel].decode("utf-8")
    ck.candado(kernel, manifest, ESPEC)
    splits = manifest["splits"]
    for e in ("clasificador_test", "clasificador_hard", "clasificador_validacion", "clasificador_train_final"):
        assert splits[e]["sha256"] not in kernel
        with pytest.raises(ck.ErrorKernel):
            ck.candado(kernel + f"\n# {splits[e]['sha256']}\n", manifest, ESPEC)
    assert "test.jsonl" not in kernel and "hard.jsonl" not in kernel


def test_metadata_del_4b(generados, manifest):
    meta = json.loads(generados[ck.ARCHIVO_METADATA])
    assert meta["id"] == "samcaballero/denue-scian-4b" and meta["title"] == "DENUE SCIAN 4b"
    assert meta["code_file"] == "clasificador_4b.py"
    assert meta["is_private"] is True and meta["enable_gpu"] is True and meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["dataset_sources"] == [manifest["kaggle_datasets"]["denue_scian_clasificador"]["id_kaggle"]]
    assert meta["docker_image"] == ck.DOCKER_06B_INICIAL == json.loads(
        (REPO_RAIZ / "notebooks" / "kaggle-06b-barrido" / "kernel-metadata.json").read_bytes()
    )["docker_image"]


def test_corrida_unica_mal_formada_no_genera(manifest):
    import dataclasses

    with pytest.raises(ck.ErrorKernel, match="exactamente una"):
        ck.generar(manifest, espec=dataclasses.replace(ESPEC, corridas=(CORRIDA, dict(CORRIDA, nombre="4b-x"))))


def test_modulos_del_4b_solo_usan_la_biblioteca_estandar():
    permitidos = set(sys.stdlib_module_names) | {"src"}
    for fuente in ESPEC.fuentes:
        arbol = ast.parse((REPO_RAIZ / fuente).read_text(encoding="utf-8"))
        for nodo in arbol.body:
            if isinstance(nodo, ast.Import):
                assert all(a.name.split(".")[0] in permitidos for a in nodo.names), fuente
            elif isinstance(nodo, ast.ImportFrom) and nodo.level == 0:
                assert nodo.module.split(".")[0] in permitidos | {"__future__"}, fuente


def test_el_adaptador_se_guarda_antes_de_la_inferencia():
    """En clasificador_kaggle._correr, la etapa `adaptador` (save_pretrained) va antes de `inferencia`."""
    arbol = ast.parse((REPO_RAIZ / "src" / "clasificador_kaggle.py").read_text(encoding="utf-8"))
    correr = next(n for n in arbol.body if isinstance(n, ast.FunctionDef) and n.name == "_correr")
    etapas = []
    for nodo in correr.body:
        if isinstance(nodo, ast.With):
            llamada = nodo.items[0].context_expr
            if isinstance(llamada, ast.Call) and getattr(llamada.func, "attr", "") == "etapa":
                etapas.append((llamada.args[0].value, ast.dump(nodo)))
    nombres = [e for e, _ in etapas]
    assert nombres.index("entrenamiento") + 1 == nombres.index("adaptador") == nombres.index("inferencia") - 1
    assert "save_pretrained" in dict(etapas)["adaptador"]


def test_el_kernel_del_4b_extrae_e_importa_su_modulo_aislado(generados, tmp_path):
    texto = generados[ESPEC.archivo_kernel].decode("utf-8")
    ns = {"__name__": "kernel_en_prueba"}
    exec(compile(texto, ESPEC.archivo_kernel, "exec"), ns)
    raiz = ns["extraer"](tmp_path / "denue_src")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(raiz)
    codigo = "import sys, src.corrida_4b_kaggle as k; print(k.TIEMPO_KAGGLE_S, 'torch' in sys.modules)"
    r = subprocess.run([sys.executable, "-c", codigo], env=env, cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == [str(k4.TIEMPO_KAGGLE_S), "False"]


# --- Prueba en seco con un proceso falso (como en el barrido) ---

PROCESO_FALSO = textwrap.dedent('''
    import argparse, json
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("--una")
    d = json.loads(Path(p.parse_args().una).read_text(encoding="utf-8"))
    salida = Path(d["salida"])
    salida.mkdir(parents=True)
    print("[clasificador 00:00:03] pérdida de validación en el paso 0: 3.25", flush=True)
    m = {"estado": "completo", "fallas": [], "entrenamiento": {"pasos_hechos": 5855,
         "orden_train": {"sha256_orden": d["fuentes"]["orden"]}}, "inferencia": {"filas": 1000}}
    (salida / "mediciones.json").write_text(json.dumps(m), encoding="utf-8")
''')


def test_simulacro_del_orquestador_con_la_corrida_del_4b(tmp_path):
    raiz = tmp_path / "fuentes"
    raiz.mkdir()
    (raiz / "proceso_falso.py").write_text(PROCESO_FALSO, encoding="utf-8")
    entrada = tmp_path / "input" / "d"
    entrada.mkdir(parents=True)
    esperados = {}
    for nombre in ("train_final.jsonl", "val.jsonl"):
        datos = f'{{"id": "1", "nom_estab": "Negocio inventado", "codigo_act": "461110", "f": "{nombre}"}}\n'.encode()
        (entrada / nombre).write_bytes(datos)
        esperados[nombre] = hashlib.sha256(datos).hexdigest()
    instaladas = {**bk.VERSIONES_06B_INICIAL, **dict(p.split("==") for p in bk.PAQUETES_BARRIDO)}
    dep = bk.Dependencias(
        dir_salida=tmp_path / "working", dir_trabajo=tmp_path / "trabajo", dir_entrada=tmp_path / "input",
        consultar_gpus=lambda: [{"indice": i, "nombre": "Tesla T4", "driver": bk.CONTROLADOR_06B_INICIAL} for i in (0, 1)],
        instalar=lambda reg, d: None, versiones=lambda nombres: {n: instaladas.get(n) for n in nombres},
        memoria_gpu=lambda: {0: 0, 1: 0}, estimar=k4.estimacion_s, presupuesto_s=k4.PRESUPUESTO_S,
        modulo_una="proceso_falso", raiz_fuentes=raiz, python=sys.executable, espera_s=0.2,
    )
    k4.main(esperados, None, {"orden": bk.SHA256_ORDEN_06B_INICIAL}, [CORRIDA], dep=dep)
    b = json.loads((dep.dir_salida / bk.ARCHIVO_BARRIDO).read_bytes())
    assert b["estado"] == "completo" and b["fallas"] == []
    assert "Qwen3-4B" in b["descripcion"] and b["configuracion"]["tiempo_kaggle_s"] == k4.TIEMPO_KAGGLE_S
    assert b["configuracion"]["corridas"] == [CORRIDA]
    (c,) = b["configuraciones"]
    assert c["estado"] == "completa" and c["mediciones"]["orden_igual_al_de_06b_inicial"] is True
    assert c["estimado_s"] == k4.estimacion_s(CORRIDA)
    assert c["tiempo_maximo_s"] == pytest.approx(k4.PRESUPUESTO_S - c["transcurrido_al_empezar_s"], abs=1)
    assert (dep.dir_salida / "logs" / f"{CORRIDA['nombre']}.log").is_file()
    assert kc.ARCHIVO_MEDICIONES in {Path(a["ruta"]).name for a in b["archivos_salida"]}
