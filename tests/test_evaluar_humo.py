"""Pruebas de la evaluación local de la prueba de punta a punta (Fase 2, paso 6c).

Solo usan nombres, ids, códigos, mediciones y logs inventados en carpetas temporales; nunca
leen el DENUE, los splits ni la salida real de Kaggle.
"""

import builtins
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path

import pytest

from src.evaluar_humo import (
    ARCHIVO_LOG,
    ErrorHumo,
    _etapa,
    alguna_coincidencia,
    buscar_nombres,
    candado_resultado,
    cargar_entradas,
    duracion,
    estimar_horas,
    evaluar_corrida,
    leer_log,
    posible_fuga,
    referencias_de_nombres,
    rendimiento,
    resumen_etapas,
    resumen_perdida,
    serializar,
    textos_de_json,
    version_menor,
)
from src.humo_kaggle import BASE_16_ESPEJO, MODELO_06B, MODELO_4B, PAQUETES, estadisticas_pasos
from src.metricas_clasificador import ErrorMetricas

CODIGOS = ("467111", "722511", "461110")
CATALOGO_CSV = "codigo,titulo\n461110,Abarrotes inventados\n467111,Ferretería inventada\n722511,Restaurante inventado\n"
FUENTES = {"src/inventado.py": "a" * 64}


def _sha(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def _reg(i, nombre, codigo):
    return {"id": str(i), "nom_estab": nombre, "codigo_act": codigo}


def _jsonl(regs) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in regs).encode("utf-8")


TRAIN = [_reg(i, f"Fonda Ficticia Número {i}", CODIGOS[i % 3]) for i in range(1, 96)] + [_reg(96, "Zapatería", "467111")]
VAL = [_reg(100 + i, f"Estética Imaginaria del Barrio {i}", CODIGOS[i % 3]) for i in range(4)]
# Respuestas: un acierto, un acierto con orillas, un código inexistente y un formato inválido.
RESPUESTAS = {"100": "467111", "101": " 722511\n", "102": "999999", "103": "\t467111"}


def _repo(tmp_path: Path):
    """Repo de juguete con catálogo, muestras fuera del repo, un test que no debe abrirse y su manifest."""
    raiz = tmp_path / "repo"
    (raiz / "data" / "splits").mkdir(parents=True)
    cat = CATALOGO_CSV.encode("utf-8")
    (raiz / "data" / "scian_2023_clases.csv").write_bytes(cat)
    test = _jsonl([_reg(5000, "Taller de prueba", "467111")])
    (raiz / "data" / "splits" / "test.jsonl").write_bytes(test)
    humo = tmp_path / "kaggle-datasets" / "humo"
    humo.mkdir(parents=True)
    shas = {}
    for nombre, regs in (("humo_train.jsonl", TRAIN), ("humo_val.jsonl", VAL)):
        datos = _jsonl(regs)
        (humo / nombre).write_bytes(datos)
        shas[nombre] = _sha(datos)
    manifest = {
        "scian_version": "2023",
        "catalogos": {"scian_2023_clases": {"ruta": "data/scian_2023_clases.csv", "sha256": _sha(cat)}},
        "splits": {"clasificador_test": {"ruta": "data/splits/test.jsonl", "sha256": _sha(test)}},
        "kaggle_datasets": {
            "denue_scian_humo": {
                "archivos": {
                    n: {"ruta": f"../kaggle-datasets/humo/{n}", "sha256": shas[n]} for n in ("humo_train.jsonl", "humo_val.jsonl")
                }
            }
        },
    }
    return raiz, manifest


def _corrida(lote: int, pasos: int, s: float, perdida_inicial: float = 3.0) -> dict:
    lista = [
        {"paso": i + 1, "s": s, "ejemplos": lote, "fichas_reales": 70 * lote, "fichas_con_relleno": 80 * lote, "fichas_etiquetadas": 7 * lote}
        for i in range(pasos)
    ]
    return {
        "lote": lote,
        "fp16": True,
        "segundos_train_total": round(s * pasos, 2),
        "vram_pico_mib": {"asignada": 50.0 * lote, "reservada": 60.0 * lote},
        "estadisticas": estadisticas_pasos([p["s"] for p in lista], [p["ejemplos"] for p in lista], [p["fichas_reales"] for p in lista]),
        "pasos": lista,
        "perdida": [{"paso": i + 1, "loss": perdida_inicial - 0.02 * i, "learning_rate": 1e-4} for i in range(pasos)],
    }


def _tabla():
    prompt = [{"i": 0, "id": 151644, "etiqueta": -100, "token": "<|im_start|>"}, {"i": 1, "id": 5, "etiqueta": -100, "token": "x"}]
    codigo = [{"i": 2 + j, "id": 19 + j, "etiqueta": 19 + j, "token": d} for j, d in enumerate("467111")]
    return prompt + codigo + [{"i": 8, "id": 151645, "etiqueta": 151645, "token": "<|im_end|>"}]


def _mediciones(esperados, archivos_salida, sha_pred, sha_gguf) -> dict:
    versiones = {p.split("==")[0]: p.split("==")[1] for p in PAQUETES} | {"torch": "2.11.0+cu130"}
    return {
        "estado": "completo",
        "inicio_utc": "2026-11-21T22:51:43+00:00",
        "segundos_desde_arranque": 714.1,
        "duraciones_s": {"instalacion": 30.0, "imports": 40.0, "descarga_06b": 15.0, "carga_06b": 5.0},
        "fallas": [],
        "configuracion": {"semilla": 42, "lote": 16},
        "fuentes_incrustadas_sha256": dict(FUENTES),
        "gpu": {
            "nvidia_smi": [{"indice": 0, "nombre": "Tesla T4"}, {"indice": 1, "nombre": "Tesla T4"}],
            "cuda_visible_devices": "0", "torch_device_count": 1, "torch_device_name": "Tesla T4",
            "memoria_total_mib": 14912, "capacidad_de_computo": [7, 5], "bf16_nativo": False,
        },
        "datos": {a: {"sha256": h, "sha256_verificado": True, "filas": n, "clases": 3}
                  for (a, h), n in zip(esperados.items(), (len(TRAIN), len(VAL)))},
        "comprobaciones": {
            "d_nombres_con_marca_especial": 0,
            "c_tokens_especiales_un_solo_token": {"ok": True},
            "a_prompt_generacion_igual_a_plantilla": {"textos": len(TRAIN) + len(VAL) + 1, "distintos": 0},
            "b_texto_entrenamiento_frente_a_plantilla": {"conteo": {"igual": 0, "igual_mas_salto_final": 100, "otro": 0}},
        },
        "instalacion": {"segundos_pip_install": 30.0, "cambios_freeze": {"agregados": {"unsloth": "x"}, "quitados": {}, "cambiados": {"peft": ["a", "b"]}}},
        "entorno": {"python": "3.13", "torch_cuda": "13.0", "versiones": versiones},
        "modelo_06b": {"revision_descargada": MODELO_06B["revision"], "name_or_path": MODELO_06B["repo"], "cuantizacion": {"bnb_4bit_quant_type": "nf4"}},
        "tokens": {"mayores_a_max_seq": 0, "prompt_no_es_prefijo": 0, "etiquetas_distintas_de_codigo_y_fin": 0, "largo_max": 90, "largo_medio": 70.0,
                   "fichas_etiquetadas_por_fila": {"7": 100}},
        "ejemplo_etiquetas": {"codigo": "467111", "fichas": 9, "fichas_con_etiqueta": [*"467111", "<|im_end|>"], "tabla": _tabla()},
        "entrenamiento_06b": _corrida(16, math.ceil(len(TRAIN) / 16), 0.4),
        "inferencia": {"filas": len(VAL), "s_por_fila": 0.25, "terminacion": {"token_de_fin": len(VAL)}, "fichas_generadas_max": 7,
                       "respaldo_parametros_de_muestreo": False, "generation_config": {"do_sample": False, "num_beams": 1},
                       "archivo": {"nombre": "predicciones.jsonl", "sha256": sha_pred}},
        "gguf": {"exportado": True, "archivo": "gguf/modelo.gguf", "bytes": 4, "sha256": sha_gguf, "otros_gguf_generados": [],
                 "plantilla_gguf_igual_a_tokenizador": True, "prompt_inventado": {"fichas": 65}},
        "base_16_bits": {"camino": "espejo", "main_actual_del_espejo": BASE_16_ESPEJO["revision"], "exportar": True,
                         "comparacion_tensores": {"identicos": True, "distintos": [], "solo_oficial": ["lm_head.weight"], "solo_espejo": []}},
        "lotes_06b": {"32": _corrida(32, 25, 0.6), "64": {"lote": 64, "sin_memoria": True, "error": "RuntimeError: out of memory"}},
        "modelo_4b": {
            "revision_descargada": MODELO_4B["revision"], "name_or_path": MODELO_4B["repo"], "cuantizacion": {"bnb_4bit_quant_type": "nf4"},
            "tokenizador_igual_al_06b": True, "duraciones_s": {"descarga": 60.0, "carga": 10.0},
            "lote_16": _corrida(16, 50, 2.0, 2.2), "lote_32": _corrida(32, 25, 3.5, 2.2),
        },
        "descargas_hf": [{"repo": m["repo"], "revision": m["revision"]} for m in (MODELO_06B, MODELO_4B, BASE_16_ESPEJO)],
        "archivos_salida": archivos_salida,
    }


def _salida(tmp_path: Path, manifest, respuestas=RESPUESTAS, log=None, ajustar=None) -> Path:
    """Carpeta de salida de Kaggle inventada, con humo_mediciones.json coherente con sus archivos."""
    carpeta = tmp_path / "kaggle-output" / "humo-6b"
    (carpeta / "adaptador").mkdir(parents=True)
    (carpeta / "gguf").mkdir()
    pred = "".join(json.dumps({"id": i, "respuesta": r}, ensure_ascii=False) + "\n" for i, r in respuestas.items()).encode("utf-8")
    archivos = {
        "predicciones.jsonl": pred,
        "adaptador/adapter_model.safetensors": b"\x00" * 16,
        "adaptador/README.md": b"# Model card inventada\n",
        "gguf/modelo.gguf": b"GGUF",
        "pip_freeze_antes.txt": b"paquete==1.0\n",
    }
    for ruta, datos in archivos.items():
        (carpeta / ruta).write_bytes(datos)
    salida = [{"ruta": r, "bytes": len(d), "sha256": _sha(d)} for r, d in sorted(archivos.items())]
    esperados = {a: e["sha256"] for a, e in manifest["kaggle_datasets"]["denue_scian_humo"]["archivos"].items()}
    m = _mediciones(esperados, salida, _sha(pred), _sha(archivos["gguf/modelo.gguf"]))
    if ajustar:
        ajustar(m)
    (carpeta / "humo_mediciones.json").write_bytes(json.dumps(m, ensure_ascii=False, indent=2).encode("utf-8"))
    entradas = log if log is not None else [
        {"stream_name": "stdout", "time": 0.9, "data": "[humo 22:51:43] inicio: gpu\n"},
        {"stream_name": "stderr", "time": 723.6, "data": "aviso inventado\n"},
    ]
    (carpeta / ARCHIVO_LOG).write_bytes(json.dumps(entradas).encode("utf-8"))
    return carpeta


def _contexto():
    return {"fecha": "2026-11-21", "commit": "c" * 40, "arbol_limpio": True, "commit_kernel": "k" * 40,
            "kernel_igual_al_generado": True, "fuentes_actuales_sha256": dict(FUENTES)}


# --- Flujo completo ---


def test_flujo_completo_y_nunca_abre_test(tmp_path, monkeypatch):
    raiz, manifest = _repo(tmp_path)
    carpeta = _salida(tmp_path, manifest)
    abiertos: list[str] = []
    original = io.open

    def espia(archivo, *args, **kwargs):
        if not isinstance(archivo, int):
            abiertos.append(Path(os.fspath(archivo)).name)
        return original(archivo, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", espia)
    monkeypatch.setattr(io, "open", espia)
    ent = cargar_entradas(carpeta, manifest, raiz)
    res = evaluar_corrida(ent, manifest, _contexto())
    monkeypatch.undo()
    assert {"humo_train.jsonl", "humo_val.jsonl", "scian_2023_clases.csv", "predicciones.jsonl"} <= set(abiertos)
    assert "test.jsonl" not in abiertos

    assert res["reportable"] is False and res["semilla"] == 42
    for campo in ("fecha", "commit", "modelo", "configuracion", "split", "hash_split", "tamano_muestra", "horas_gpu"):
        assert campo in res
    mt = res["metricas_regla5"]
    assert mt["total"] == 4
    assert mt["conteos"] == {"formato_invalido": 1, "codigo_inexistente": 1, "codigo_existente": 2}
    assert mt["aciertos"] == 2 and mt["pct_accuracy"] == 50.0
    assert res["etapas"]["todas_ok"] is False  # el lote 64 del 0.6B se quedó sin memoria
    e = res["etapas"]["etapas"]
    assert [k for k, v in e.items() if v["estado"] != "ok"] == ["lotes_06b"]
    assert e["estado_final"]["estado"] == "ok" and e["estado_final"]["estado_del_kernel"] == "completo"
    assert res["privacidad"]["posible_fuga"] is False
    assert res["corrida_kaggle"]["descarga"]["coinciden_con_kaggle"] == 5

    texto = serializar(res).decode("utf-8")
    assert texto.endswith("}\n")
    ref = referencias_de_nombres(TRAIN + VAL)
    assert not alguna_coincidencia(buscar_nombres(texto, ref))
    for r in TRAIN + VAL:
        assert r["nom_estab"] not in texto
    for respuesta in RESPUESTAS.values():  # sin respuestas crudas
        assert json.dumps(respuesta)[1:-1] not in texto or respuesta.strip().isdigit()


def test_estimacion_excluye_lotes_sin_memoria(tmp_path):
    raiz, manifest = _repo(tmp_path)
    res = evaluar_corrida(cargar_entradas(_salida(tmp_path, manifest), manifest, raiz), manifest, _contexto())
    est = res["estimacion_horas_gpu"]["por_modelo"]
    assert est["qwen3_0.6b"]["lote"] == 32 and est["qwen3_0.6b"]["lotes_considerados"] == [16, 32]
    assert est["qwen3_4b"]["lote"] == 32


# --- Paradas ---


def test_parada_si_un_archivo_bajado_no_coincide(tmp_path):
    raiz, manifest = _repo(tmp_path)
    carpeta = _salida(tmp_path, manifest)
    (carpeta / "adaptador" / "README.md").write_bytes(b"cambiado\n")
    with pytest.raises(ErrorHumo, match="distintos 1"):
        evaluar_corrida(cargar_entradas(carpeta, manifest, raiz), manifest, _contexto())


def test_parada_si_salio_un_archivo_de_datos(tmp_path):
    raiz, manifest = _repo(tmp_path)
    carpeta = _salida(tmp_path, manifest)
    (carpeta / "humo_val.jsonl").write_bytes((tmp_path / "kaggle-datasets" / "humo" / "humo_val.jsonl").read_bytes())
    with pytest.raises(ErrorHumo, match="con hash del manifest 1"):
        evaluar_corrida(cargar_entradas(carpeta, manifest, raiz), manifest, _contexto())


def test_parada_si_falta_el_log(tmp_path):
    raiz, manifest = _repo(tmp_path)
    carpeta = _salida(tmp_path, manifest)
    (carpeta / ARCHIVO_LOG).unlink()
    with pytest.raises(FileNotFoundError):
        cargar_entradas(carpeta, manifest, raiz)


def test_parada_por_nombre_en_el_log_sin_mostrarlo(tmp_path):
    raiz, manifest = _repo(tmp_path)
    nombre = TRAIN[10]["nom_estab"]
    log = [{"stream_name": "stdout", "time": 1.0, "data": f"ejemplo filtrado: {nombre}\n"}]
    carpeta = _salida(tmp_path, manifest, log=log)
    with pytest.raises(ErrorHumo, match="Posible fuga") as e:
        evaluar_corrida(cargar_entradas(carpeta, manifest, raiz), manifest, _contexto())
    assert nombre not in str(e.value)


def test_parada_por_llave_normalizada_en_las_mediciones(tmp_path):
    raiz, manifest = _repo(tmp_path)

    def ajustar(m):
        m["nota_inventada"] = "estetica-imaginaria  DEL barrio 2"  # misma llave que un nombre de VAL

    carpeta = _salida(tmp_path, manifest, ajustar=ajustar)
    with pytest.raises(ErrorHumo, match="humo_mediciones.json"):
        evaluar_corrida(cargar_entradas(carpeta, manifest, raiz), manifest, _contexto())


def test_parada_si_los_ids_no_son_los_de_validacion(tmp_path):
    raiz, manifest = _repo(tmp_path)
    respuestas = dict(RESPUESTAS)
    del respuestas["103"]
    respuestas["999"] = "467111"
    carpeta = _salida(tmp_path, manifest, respuestas=respuestas)
    with pytest.raises(ErrorMetricas, match="faltan 1 y sobran 1"):
        evaluar_corrida(cargar_entradas(carpeta, manifest, raiz), manifest, _contexto())


def test_parada_si_una_muestra_cambio(tmp_path):
    raiz, manifest = _repo(tmp_path)
    carpeta = _salida(tmp_path, manifest)
    (tmp_path / "kaggle-datasets" / "humo" / "humo_val.jsonl").write_bytes(_jsonl(VAL[:3]))
    with pytest.raises(RuntimeError, match="SHA-256"):
        cargar_entradas(carpeta, manifest, raiz)


def test_candado_del_resultado():
    ref = referencias_de_nombres(TRAIN + VAL)
    candado_resultado({"total": 4, "nota": "sin nombres"}, ref)
    for malo in ({"x": TRAIN[0]["nom_estab"]}, {"x": "fonda ficticia numero 5"}, {"zapateria": 1}):
        with pytest.raises(ErrorHumo, match="no se escribe") as e:
            candado_resultado(malo, ref)
        assert "Fonda" not in str(e.value) and "Zapater" not in str(e.value)


# --- Funciones puras ---


def test_buscar_nombres():
    ref = referencias_de_nombres(TRAIN + VAL)
    texto = "\n".join([
        "nada que ver",
        f"crudo: {TRAIN[3]['nom_estab']}",
        "normalizado: FONDA-FICTICIA, numero 7!",
        "una palabra: zapaterías no, Zapatería sí",
        "subcadena de palabra: ZAPATERIAX no cuenta",
    ])
    r = buscar_nombres(texto, ref)
    assert r["nombres_crudos"] == 2 and r["lineas_nombres_crudos"] == [2, 4]  # el nombre de una palabra también es crudo
    assert r["llaves_varias_palabras"] == 2 and r["lineas_llaves_varias_palabras"] == [2, 3]
    assert r["llaves_una_palabra"] == 1 and r["lineas_llaves_una_palabra"] == [4]
    assert r["largos_llaves_una_palabra"] == [len("ZAPATERIA")]
    assert posible_fuga(r) and alguna_coincidencia(r)
    limpio = buscar_nombres("FONDA FICTICIA sin número\nSCIAN 467111", ref)
    assert not alguna_coincidencia(limpio)


def test_referencias_ignoran_nombres_cortos_y_llaves_vacias():
    ref = referencias_de_nombres([_reg(1, "AB", "467111"), _reg(2, "...", "467111"), _reg(3, " Oxxo ", "461110")])
    assert ref["nombres"] == ["...", "Oxxo"]
    assert ref["una"] == frozenset({"AB", "OXXO"}) and ref["varias"] == frozenset()


def test_textos_de_json():
    assert textos_de_json({"a": ["b", {"c": 1, "d": "e"}]}).split("\n") == ["a", "b", "c", "d", "e"]


def test_leer_log():
    arr = [{"stream_name": "stdout", "time": 1.5, "data": "hola\n"}]
    assert leer_log(json.dumps(arr).encode()) == arr
    assert leer_log(b"texto plano\n") == [{"stream_name": None, "time": None, "data": "texto plano\n"}]
    with pytest.raises(ErrorHumo):
        leer_log(b'{"data": "no es arreglo"}')
    with pytest.raises(ErrorHumo):
        leer_log(b'[{"time": 1}]')


def test_version_menor():
    assert version_menor("2.11.0+cu130", (2, 15))
    assert version_menor("1.99", (2, 15))
    assert not version_menor("2.15.0", (2, 15))
    assert not version_menor("3.0", (2, 15))
    assert not version_menor("", (2, 15))


def test_resumen_perdida():
    p = resumen_perdida([{"loss": v, "learning_rate": 0.1} for v in (3.0, float("nan"), 2.5, 2.0, 1.5, 1.0)], k=2)
    assert p["no_finitos"] == 1 and p["pasos_registrados"] == 6
    assert p["media_primeros_2"] == 2.75 and p["media_ultimos_2"] == 1.25 and p["minima"] == 1.0
    corta = resumen_perdida([{"loss": v} for v in (3.0, 1.0, 2.0)], k=10)  # los primeros y los últimos no se enciman
    assert corta["media_primeros_1"] == 3.0 and corta["media_ultimos_1"] == 2.0
    assert resumen_perdida([]) == {"pasos_registrados": 0, "no_finitos": 0}


def test_rendimiento_y_estimar_horas():
    m = {"entrenamiento_06b": _corrida(16, 25, 0.4), "lotes_06b": {"32": _corrida(32, 25, 0.5)},
         "modelo_4b": {"lote_16": _corrida(16, 25, 2.0), "lote_32": {"sin_memoria": True}}}
    rend = rendimiento(m)
    assert [(r["modelo"], r["lote"], r["sin_memoria"]) for r in rend] == [
        (MODELO_06B["repo"], 16, False), (MODELO_06B["repo"], 32, False), (MODELO_4B["repo"], 16, False), (MODELO_4B["repo"], 32, True)
    ]
    assert rend[1]["ejemplos_por_s"] == 64.0 and rend[1]["relleno_pct"] == 12.5 and rend[1]["recalculo_desde_los_pasos_coincide"]
    fijo = {MODELO_06B["repo"]: {"total": 360.0}, MODELO_4B["repo"]: {"total": 0.0}}
    est = estimar_horas(rend, (230_400,), fijo, {MODELO_06B["repo"]: 0.2})
    e06 = est["qwen3_0.6b"]
    assert e06["lote"] == 32 and e06["ejemplos_por_s"] == 64.0
    assert e06["tabla"] == [{"filas_train": 230_400, "pasos": 7200, "horas_entrenamiento": 1.0,
                             "horas_contraste_mediana_s_por_paso": 1.0, "horas_con_costo_fijo": 1.1}]
    assert e06["inferencia_validacion_1000_filas_s"] == 200.0
    assert est["qwen3_4b"]["lote"] == 16 and est["qwen3_4b"]["lotes_considerados"] == [16]
    assert est["qwen3_4b"]["inferencia_validacion_1000_filas_s"] is None


def test_duracion():
    d = duracion([{"time": 0.9}, {"time": 723.6}, {"time": None}], {"inicio_utc": "2026-11-21T22:51:43+00:00", "segundos_desde_arranque": 714.1})
    assert d["fin_script_utc"] == "2026-11-21T23:03:37+00:00"
    assert d["log_ultima_marca_s"] == 723.6 and d["cota_superior_cli_s"] == 858.0 and d["pagina_kaggle_s"] == 960
    assert d["horas"] == {"script": 0.198, "log": 0.201, "cota_superior_cli": 0.238, "pagina_kaggle": 0.267}


def _etapas(m):
    esperados = {a: d["sha256"] for a, d in m["datos"].items()}
    return resumen_etapas(m, esperados=esperados, fuentes_actuales=FUENTES, kernel_igual_al_generado=True,
                          descarga={"gguf_coincide": True}, filas_train=len(TRAIN), filas_val=len(VAL))


def test_resumen_etapas_detecta_fallas():
    base = _mediciones({"humo_train.jsonl": "1" * 64, "humo_val.jsonl": "2" * 64}, [], "p", "g")
    base["lotes_06b"]["64"] = _corrida(64, 25, 1.0)
    assert _etapas(base)["todas_ok"] is True

    casos = {
        "gpu": lambda m: m["gpu"]["nvidia_smi"].append({"nombre": "Tesla P100"}),
        "datos": lambda m: m.update(fuentes_incrustadas_sha256={"src/inventado.py": "b" * 64}),
        "instalacion": lambda m: m["instalacion"]["cambios_freeze"]["cambiados"].update(torch=["2.11", "2.15"]),
        "revisiones": lambda m: m["descargas_hf"].append({"repo": "otro/repo", "revision": "x"}),
        "plantilla_y_tokens": lambda m: m["tokens"].update(mayores_a_max_seq=1),
        "entrenamiento_06b": lambda m: m["entrenamiento_06b"]["perdida"][3].update(loss=float("nan")),
        "inferencia": lambda m: m["inferencia"].update(terminacion={"token_de_fin": 3, "limite_de_tokens": 1}),
        "espejo_16_bits": lambda m: m["base_16_bits"]["comparacion_tensores"].update(identicos=False),
        "gguf": lambda m: m["gguf"].update(exportado=False),
        "estado_final": lambda m: m.update(estado="completo_con_fallas", fallas=[{"etapa": "gguf", "error": "x"}]),
    }
    for etapa, cambiar in casos.items():
        m = copy.deepcopy(base)
        cambiar(m)
        e = _etapas(m)
        assert e["todas_ok"] is False
        assert [k for k, v in e["etapas"].items() if v["estado"] != "ok"] == [etapa], etapa

    m = copy.deepcopy(base)
    m["modelo_4b"] = {"omitido": "El reloj pasó de 60 min antes del 4B."}
    m["descargas_hf"] = m["descargas_hf"][:1]
    e = _etapas(m)
    assert e["etapas"]["modelo_4b"]["estado"] == "omitido"
    assert "qwen3_4b_revision_fijada" not in e["etapas"]["revisiones"]["comprobaciones"]


def test_tabla_de_etiquetas_mal_formada_falla():
    m = _mediciones({"humo_train.jsonl": "1" * 64, "humo_val.jsonl": "2" * 64}, [], "p", "g")
    m["ejemplo_etiquetas"]["tabla"][1]["etiqueta"] = 5  # un token del prompt con etiqueta
    assert _etapas(m)["etapas"]["plantilla_y_tokens"]["comprobaciones"]["tabla_del_ejemplo_inventado"] is False


def test_etapa_rechaza_claves_reservadas():
    with pytest.raises(ValueError):
        _etapa({"x": True}, estado="completo")
