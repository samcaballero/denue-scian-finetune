"""Pruebas de la evaluación local de una corrida del clasificador (src/evaluar_corrida_clasificador.py).

Solo usan nombres, ids, códigos y salidas inventados; nunca leen datos reales.
"""

import builtins
import hashlib
import io
import json
import os
import random
from pathlib import Path

import pytest

from src import evaluar_corrida_clasificador as ec
from src.evaluador_clasificador import SECTORES_2023, aciertos_por_nivel, evaluar_validacion, validar_referencias
from src.evaluar_humo import buscar_nombres, referencias_de_nombres
from src.humo_kaggle import TAPADO, serializar_predicciones
from src.splits import serializar

# --- Búsqueda indexada: lo mismo que la del 6c ---

PALABRAS = ["TACOS", "EL", "GÜERO", "FERRE", "LUNA", "AZUL", "123", "MX", "DOÑA", "PAN", "SOL", "ABC", "N°5"]


def _nombres_inventados(rng: random.Random, n: int) -> list[dict]:
    filas = []
    for i in range(n):
        k = rng.randint(1, 4)
        nombre = " ".join(rng.choice(PALABRAS) for _ in range(k))
        if rng.random() < 0.2:
            nombre = nombre.title()
        if rng.random() < 0.1:
            nombre = nombre + "."
        filas.append({"id": str(i), "nom_estab": nombre, "codigo_act": "461110"})
    filas.append({"id": "9998", "nom_estab": "LINEA\nPARTIDA", "codigo_act": "461110"})
    filas.append({"id": "9999", "nom_estab": "  X  ", "codigo_act": "461110"})  # corto: no se busca crudo
    return filas


def _texto_inventado(rng: random.Random, filas) -> str:
    lineas = []
    for _ in range(80):
        partes = []
        for _ in range(rng.randint(0, 6)):
            r = rng.random()
            if r < 0.3:
                partes.append(rng.choice(filas)["nom_estab"])
            elif r < 0.5:
                partes.append(rng.choice(PALABRAS).lower())
            elif r < 0.7:
                partes.append(f'"clave_{rng.randint(0, 9)}": {rng.random():.3f},')
            else:
                partes.append(rng.choice(PALABRAS) + rng.choice(["_X", "-1", "", "Z"]))
        lineas.append(" ".join(partes))
    lineas.append("LINEA\nPARTIDA al final")
    return "\n".join(lineas)


@pytest.mark.parametrize("semilla", range(6))
def test_busqueda_indexada_igual_a_la_del_6c(semilla):
    rng = random.Random(semilla)
    filas = _nombres_inventados(rng, 60)
    ref = referencias_de_nombres(filas)
    idx = ec.indice_de_nombres(ref)
    for _ in range(3):
        texto = _texto_inventado(rng, filas)
        resumen, encontrados = ec.buscar_nombres_indexada(texto, idx)
        assert resumen == buscar_nombres(texto, ref)
        assert encontrados["crudos_aislados"] <= encontrados["crudos"]


def test_busqueda_indexada_texto_vacio_y_sin_coincidencias():
    ref = referencias_de_nombres([{"id": "1", "nom_estab": "TACOS EL GÜERO", "codigo_act": "722511"}])
    idx = ec.indice_de_nombres(ref)
    for texto in ("", "\n\n", "nada que ver aquí"):
        assert ec.buscar_nombres_indexada(texto, idx)[0] == buscar_nombres(texto, ref)


# --- Clasificación de coincidencias y posible fuga ---


def _clasificar(nombres, texto, kernel="def main():\n    pass\n"):
    ref = referencias_de_nombres([{"id": str(i), "nom_estab": n, "codigo_act": "461110"} for i, n in enumerate(nombres)])
    _, enc = ec.buscar_nombres_indexada(texto, ec.indice_de_nombres(ref))
    return ec.clasificar(enc, kernel, ec.palabras_del_texto(kernel))


def test_nombre_corto_solo_dentro_de_un_token_se_acepta():
    for texto in ("TOKENIZERS_PARAABCLELISM=false", "xxABCyy", "pkg-ABCD==1.0", "ABCD_ALGO"):
        c = _clasificar(["ABC", "ABCD"], texto)
        assert c["nombres_crudos_con_letras"] >= 1
        assert c["nombres_crudos_posible_fuga"] == 0 and not ec.posible_fuga(c), texto


def test_nombre_corto_aislado_es_posible_fuga():
    c = _clasificar(["ABC"], "un aviso: ABC terminó")
    assert c["nombres_crudos_posible_fuga"] == 1 and ec.posible_fuga(c)
    # Si aparece dentro de un token y también aislado, cuenta como posible fuga.
    c = _clasificar(["ABC"], "xxABCyy\n(ABC)")
    assert ec.posible_fuga(c)


def test_nombre_de_cinco_o_mas_dentro_de_un_token_es_posible_fuga():
    c = _clasificar(["ABCDE"], "XABCDEX")
    assert c["nombres_crudos_cortos_solo_dentro_de_un_token"] == 0
    assert c["nombres_crudos_posible_fuga"] == 1 and ec.posible_fuga(c)
    assert ec.LARGO_MAXIMO_DENTRO_DE_TOKEN == 4


def test_solo_digitos_se_acepta_y_llaves_de_varias_palabras_son_fuga():
    c = _clasificar(["123", "4 56"], "paso 123 de 4-56")
    # "123" aparece crudo; "4 56" solo como llave (en el texto va con guion).
    assert c["nombres_crudos_solo_digitos"] == 1 and c["llaves_varias_palabras_solo_digitos"] == 1
    assert not ec.posible_fuga(c)
    c = _clasificar(["TACOS LUNA"], "registro: tacos-luna")
    assert c["llaves_varias_palabras_con_letras"] == 1 and ec.posible_fuga(c)


def test_coincidencias_en_el_texto_del_kernel_se_cuentan():
    kernel = "TOKENIZERS = 1\nprint('CUDA listo')\n"
    c = _clasificar(["CUDA", "SOL"], "aviso CUDA_ALLOC y SOL", kernel)
    assert c["nombres_crudos_con_letras_en_el_kernel"] == 1
    assert c["llaves_una_palabra_con_letras_en_el_kernel"] == 1


# --- Descarga ---


def _inv(**archivos):
    return [{"ruta": r, "bytes": b, "sha256": s} for r, (b, s) in sorted(archivos.items())]


def test_verificar_descarga():
    m = {"archivos_salida": _inv(**{"predicciones.jsonl": (10, "a" * 64), "adaptador/x.json": (5, "b" * 64)}),
         "inferencia": {"archivo": {"sha256": "a" * 64}}}
    inv = _inv(**{"predicciones.jsonl": (10, "a" * 64), "adaptador/x.json": (5, "b" * 64), "mediciones.json": (3, "c" * 64),
                  "k.log": (4, "d" * 64)})
    res = ec.verificar_descarga(inv, m, {"f" * 64}, "k.log")
    assert res["coinciden_con_kaggle"] == 2 and res["sin_sha256_de_kaggle"] == ["k.log", "mediciones.json"]
    otro_tamano = [dict(a, bytes=6) if a["ruta"] == "adaptador/x.json" else a for a in inv]
    with pytest.raises(ec.ErrorCorrida, match="distintos 1"):
        ec.verificar_descarga(otro_tamano, m, set(), "k.log")
    with pytest.raises(ec.ErrorCorrida, match="con hash del manifest 1"):
        ec.verificar_descarga(inv, m, {"b" * 64}, "k.log")
    with pytest.raises(ec.ErrorCorrida, match="extra 1"):
        ec.verificar_descarga(inv + _inv(**{"otro.txt": (1, "e" * 64)}), m, set(), "k.log")
    with pytest.raises(ec.ErrorCorrida, match="sin log o mediciones 1"):
        ec.verificar_descarga([a for a in inv if a["ruta"] != "k.log"], m, set(), "k.log")
    with pytest.raises(ec.ErrorCorrida, match="predicciones distintas"):
        ec.verificar_descarga(inv, {**m, "inferencia": {"archivo": {"sha256": "0" * 64}}}, set(), "k.log")


# --- Curvas, argumentos y validación del log ---


def test_validacion_del_log_y_curva():
    texto = (
        "[clasificador 10:00:00] pérdida de validación en el paso 0: 3.5\n"
        "[clasificador 10:30:00] pérdida de validación en el paso 500: 0.25\n"
        "otra línea\n"
        "[clasificador 10:30:00] pérdida de validación en el paso 500: 0.25\n"
        "[clasificador 11:00:00] pérdida de validación en el paso 1000: None\n"
    )
    log = ec.validacion_del_log(texto)
    assert log == {0: 3.5, 500: 0.25, 1000: None}
    curva = ec.curva_de_validacion([{"paso": 500, "eval_loss": 0.25}, {"paso": 1000, "eval_loss": 0.2}], log)
    assert [(p["paso"], p["fuente"]) for p in curva] == [(0, "log (on_evaluate)"), (500, "mediciones.json"), (1000, "mediciones.json")]
    with pytest.raises(ec.ErrorCorrida, match="dos pérdidas"):
        ec.validacion_del_log(texto + "pérdida de validación en el paso 0: 3.6\n")


def test_curva_de_train_agregada():
    train = [{"paso": p, "loss": float(p)} for p in range(1, 121)] + [{"paso": 121, "loss": float("nan")}]
    v = ec.curva_de_train_agregada(train, ventana=50)
    assert [(x["paso_inicial"], x["paso_final"], x["pasos"]) for x in v] == [(1, 50, 50), (51, 100, 50), (101, 120, 20)]
    assert v[0]["media"] == 25.5


def test_revisar_argumentos():
    esperados = {"learning_rate": 2e-4, "save_strategy": "no", "train_sampling_strategy": "random", "seed": 42}
    args = {**esperados, "save_strategy": TAPADO, "train_sampling_strategy": TAPADO}
    tapados = ["$.entrenamiento.argumentos.save_strategy", "$.entrenamiento.argumentos.train_sampling_strategy"]
    r = ec.revisar_argumentos(args, esperados, tapados, {"NO", "RANDOM"})
    assert r["mismas_claves"] and not r["distintos"] and not r["tapados_sin_explicar"]
    assert r["tapados_explicados"] == ["save_strategy", "train_sampling_strategy"]
    r = ec.revisar_argumentos(args, esperados, tapados, {"NO"})  # RANDOM no es llave: el tapado no se explica
    assert r["tapados_sin_explicar"] == ["train_sampling_strategy"]
    r = ec.revisar_argumentos({**esperados, "learning_rate": 1e-4}, esperados, [], set())
    assert r["distintos"] == ["learning_rate"]
    r = ec.revisar_argumentos(esperados, esperados, ["$.datos.algo"], set())
    assert r["tapados_fuera_de_los_argumentos"] == ["$.datos.algo"]


def test_argumentos_esperados_son_los_del_kernel():
    from src.construir_kernel import KERNELS

    corrida = KERNELS["06b-inicial"].corrida
    a = ec.argumentos_esperados(corrida, 5855)
    assert a["warmup_steps"] == 176 and a["fp16"] is True and a["bf16"] is False
    assert a["output_dir"] == "/tmp/denue_trabajo/trainer_06b-inicial"
    assert a["train_sampling_strategy"] == "random" and a["seed"] == a["data_seed"] == 42


# --- Alineación, TF-IDF y comparación ---

CATALOGO = frozenset(f"{s[:2]}1110" for s in SECTORES_2023) | {"461120", "722511"}


def _val(n=40, semilla=3):
    rng = random.Random(semilla)
    codigos = sorted(CATALOGO)
    filas = [
        {"id": str(7_000_000 + 37 * i), "codigo_act": rng.choice(codigos), "sin_nombre": rng.random() < 0.2,
         "generico": rng.random() < 0.3}
        for i in range(n)
    ]
    rng.shuffle(filas)
    return filas


def _respuestas(filas, acierto, semilla):
    rng = random.Random(semilla)
    otros = sorted(CATALOGO)
    return {f["id"]: (f["codigo_act"] if rng.random() < acierto else rng.choice(otros)) for f in filas}


def test_alinear():
    val = _val()
    pred = {f["id"]: f["codigo_act"] for f in val}
    paquete = [{"id": f["id"], "nom_estab": "x", "codigo_act": f["codigo_act"]} for f in val]
    filas = ec.alinear(pred, val, paquete)
    assert [int(f["id"]) for f in filas] == sorted(int(f["id"]) for f in val)
    with pytest.raises(ec.ErrorCorrida, match="predicciones y"):
        ec.alinear(dict(list(pred.items())[:-1]), val, paquete)
    otro = dict(pred)
    otro.pop(val[0]["id"])
    otro["1"] = "461110"
    with pytest.raises(ec.ErrorCorrida, match="no son los mismos"):
        ec.alinear(otro, val, paquete)
    malo = [dict(p) for p in paquete]
    malo[0]["codigo_act"] = "999999"
    with pytest.raises(ec.ErrorCorrida, match="mismas etiquetas"):
        ec.alinear(pred, val, malo)


def test_aciertos_alineados_en_orden_de_id():
    val = _val()
    pred = _respuestas(val, 0.6, 1)
    filas = validar_referencias(val)
    a = ec.aciertos_alineados(pred, filas, CATALOGO)
    assert a["clase"] == [pred[f["id"]] == f["codigo_act"] for f in filas]


def _registro_tfidf(val, pred):
    filas = validar_referencias(val)
    m = evaluar_validacion(pred, filas, CATALOGO)
    n = aciertos_por_nivel(pred, filas, CATALOGO)
    reg = {"clave": "tfidf_nombre_C10", "commit": "c" * 40, "metricas": m, "aciertos": {"clase": n["clase"], "sector": n["sector"]}}
    res = {"commit": "c" * 40, "detalle": {"C": 10.0}, "metricas": m}
    return filas, reg, res


def test_comprobar_registro_tfidf_alineado():
    val = _val(200)
    filas, reg, res = _registro_tfidf(val, _respuestas(val, 0.6, 2))
    r = ec.comprobar_registro_tfidf(reg, res, filas)
    assert all(v for k, v in r["comprobaciones"].items() if k != "trae_codigos_predichos")
    assert "no es una comprobación fila por fila" in r["fuerza"]


def test_comprobar_registro_tfidf_desalineado_o_con_codigos():
    val = _val(200)
    filas, reg, res = _registro_tfidf(val, _respuestas(val, 0.6, 2))
    clase = list(reg["aciertos"]["clase"])
    sector = list(reg["aciertos"]["sector"])
    i = next(k for k in range(len(clase)) if clase[k])
    j = next(k for k in range(len(clase)) if not clase[k] and not sector[k])
    clase[i], clase[j] = clase[j], clase[i]
    sector[i], sector[j] = sector[j], sector[i]
    malo = {**reg, "aciertos": {"clase": clase, "sector": sector}}
    with pytest.raises(ec.ErrorCorrida, match="alineación"):
        ec.comprobar_registro_tfidf(malo, res, filas)
    with pytest.raises(ec.ErrorCorrida, match="códigos predichos"):
        ec.comprobar_registro_tfidf({**reg, "respuestas": ["461110"]}, res, filas)
    with pytest.raises(ec.ErrorCorrida, match="mismo_commit"):
        ec.comprobar_registro_tfidf({**reg, "commit": "d" * 40}, res, filas)


def test_comparacion_signo_corrida_menos_tfidf():
    val = _val(200)
    filas, reg, res = _registro_tfidf(val, _respuestas(val, 0.4, 2))
    pred = _respuestas(val, 0.8, 5)
    m = evaluar_validacion(pred, filas, CATALOGO)
    a = ec.aciertos_alineados(pred, filas, CATALOGO)
    comp = ec.comparacion(a, m, res, {"metricas": m}, reg, {"x": 1}, "data/baselines/c/t.json", "0" * 64)
    d = comp["diferencia_pareada_contra_tfidf"]
    assert d["clase"]["diferencia"] == pytest.approx(m["regla5"]["accuracy"] - res["metricas"]["regla5"]["accuracy"])
    assert d["clase"]["diferencia"] > 0 and d["alineacion_corrida"]["comprobada_fila_por_fila"] is True


# --- Candado del resultado ---


def test_candado_resultado():
    ref = referencias_de_nombres([{"id": "1", "nom_estab": "Tacos Luna", "codigo_act": "722511"},
                                  {"id": "2", "nom_estab": "RANDOM", "codigo_act": "461110"}])
    ec.candado_resultado({"a": "otra cosa", "b": "2026-11-23", "c": ["461110"]}, ref)
    for malo in ({"a": "tacos luna"}, {"a": ["random"]}):
        with pytest.raises(ec.ErrorCorrida, match="candado"):
            ec.candado_resultado(malo, ref)


# --- Flujo completo con datos inventados ---


def _escribir(ruta: Path, datos: bytes) -> str:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_bytes(datos)
    return hashlib.sha256(datos).hexdigest()


def _repo_inventado(tmp_path, monkeypatch, log_extra=""):
    raiz = tmp_path / "repo"
    catalogo = sorted(CATALOGO)
    cat = ("codigo,titulo\n" + "".join(f"{c},Título inventado {c}\n" for c in catalogo)).encode("utf-8")
    sha_cat = _escribir(raiz / "data" / "cat.csv", cat)
    val = sorted(_val(30), key=lambda f: int(f["id"]))  # los splits van en orden de int(id)
    val_regs = [{"id": f["id"], "nom_estab": f"Negocio Inventado Número {i}", "codigo_act": f["codigo_act"],
                 "llave": f"NEGOCIO INVENTADO NUMERO {i}", "sin_nombre": f["sin_nombre"], "generico": f["generico"],
                 "igual_titulo": False} for i, f in enumerate(val)]
    splits = {}
    for nombre, regs, entrada in (("val.jsonl", val_regs, "clasificador_validacion"),
                                  ("test.jsonl", val_regs[:2], "clasificador_test"),
                                  ("hard.jsonl", val_regs[:1], "clasificador_hard"),
                                  ("train_final.jsonl", val_regs[:3], "clasificador_train_final")):
        datos = serializar(regs)
        splits[entrada] = {"ruta": f"data/splits/{nombre}", "sha256": _escribir(raiz / "data" / "splits" / nombre, datos),
                           "filas": len(regs)}
    paquete_val = [{"id": r["id"], "nom_estab": r["nom_estab"], "codigo_act": r["codigo_act"]} for r in val_regs]
    paquete_train = [{"id": str(8_000_000 + i), "nom_estab": f"Ferretería Inventada {i}", "codigo_act": catalogo[i % len(catalogo)]}
                     for i in range(25)]
    arch = {}
    for nombre, regs in (("train_final.jsonl", paquete_train), ("val.jsonl", paquete_val)):
        datos = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in regs).encode("utf-8")
        arch[nombre] = {"ruta": f"paquete/{nombre}", "sha256": _escribir(raiz / "paquete" / nombre, datos)}
    manifest = {"scian_version": "2023", "splits": splits,
                "catalogos": {"scian_2023_clases": {"ruta": "data/cat.csv", "sha256": sha_cat}},
                "kaggle_datasets": {"denue_scian_clasificador": {"id_kaggle": "x/y", "archivos": arch}}}

    filas = validar_referencias(val_regs)
    pred_tfidf = _respuestas(val_regs, 0.5, 11)
    m_tfidf = evaluar_validacion(pred_tfidf, filas, CATALOGO)
    n_tfidf = aciertos_por_nivel(pred_tfidf, filas, CATALOGO)
    _escribir(raiz / ec.RESULTADO_TFIDF, json.dumps({"commit": "c" * 40, "detalle": {"C": 10.0}, "metricas": m_tfidf}).encode())
    _escribir(raiz / ec.RESULTADO_DICCIONARIO, json.dumps({"metricas": m_tfidf}).encode())
    _escribir(raiz / "data" / "baselines" / ("c" * 12) / "tfidf_nombre_C10.json",
              json.dumps({"clave": "tfidf_nombre_C10", "commit": "c" * 40, "metricas": m_tfidf,
                          "aciertos": {"clase": n_tfidf["clase"], "sector": n_tfidf["sector"]}}).encode())

    carpeta = raiz / "kaggle-output" / "prueba"
    pred = serializar_predicciones([(r["id"], r["codigo_act"] if i % 3 else "461110") for i, r in enumerate(val_regs)])
    sha_pred = _escribir(carpeta / "predicciones.jsonl", pred)
    med = {"estado": "completo", "fallas": [], "archivos_salida": [{"ruta": "predicciones.jsonl", "bytes": len(pred), "sha256": sha_pred}],
           "inferencia": {"archivo": {"sha256": sha_pred}}, "configuracion": {}, "entrenamiento": {},
           "segundos_desde_arranque": 100.0, "inicio_utc": "2026-01-01T00:00:00+00:00"}
    _escribir(carpeta / "mediciones.json", json.dumps(med).encode())
    log = [{"stream_name": "stdout", "time": 1.5, "data": "[clasificador 00:00:01] inicio: gpu\n" + log_extra}]
    _escribir(carpeta / "prueba.log", json.dumps(log).encode())
    monkeypatch.setitem(ec.CORRIDAS, "prueba", {
        "kernel": "06b-inicial", "carpeta": "kaggle-output/prueba", "log": "prueba.log",
        "resultado": "results/prueba.json", "figura": "docs/prueba.png",
        "observaciones": {**ec.CORRIDAS["06b-inicial"]["observaciones"]},
    })
    contexto = {"fecha": "2026-01-01", "commit": "d" * 40, "arbol_limpio": True, "commit_kernel": "e" * 40,
                "kernel_igual_al_generado": True, "fuentes_actuales_sha256": {}, "texto_kernel": "def main():\n    pass\n"}
    return raiz, manifest, contexto, val_regs


def test_flujo_completo_sin_test_ni_hard_y_sin_ids(tmp_path, monkeypatch):
    raiz, manifest, contexto, val_regs = _repo_inventado(tmp_path, monkeypatch)
    abiertos: list[str] = []
    original = io.open

    def espia(archivo, *args, **kwargs):
        if not isinstance(archivo, int):
            abiertos.append(Path(os.fspath(archivo)).name)
        return original(archivo, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", espia)
    monkeypatch.setattr(io, "open", espia)
    ent = ec.cargar_entradas("prueba", manifest, raiz)
    res, privacidad = ec.evaluar_corrida(ent, manifest, contexto)
    monkeypatch.undo()
    assert "val.jsonl" in abiertos and "predicciones.jsonl" in abiertos  # el espía sí ve las aperturas
    assert "test.jsonl" not in abiertos and "hard.jsonl" not in abiertos
    assert privacidad["posible_fuga"] is False
    texto = ec.serializar(res).decode("utf-8")
    assert not any(r["id"] in texto for r in val_regs)
    assert "Inventad" not in texto and "INVENTAD" not in texto
    assert res["metricas"]["regla5"]["total"] == 30
    d = res["comparacion"]["diferencia_pareada_contra_tfidf"]
    tabla = res["comparacion"]["tabla"]
    assert d["clase"]["diferencia"] == pytest.approx(tabla[0]["accuracy"] - tabla[1]["accuracy"], abs=2e-6)  # cada término va redondeado a 6 decimales
    assert res["hash_split"]["validacion"] == manifest["splits"]["clasificador_validacion"]["sha256"]


def test_flujo_completo_para_ante_un_nombre_en_el_log(tmp_path, monkeypatch):
    raiz, manifest, contexto, _ = _repo_inventado(tmp_path, monkeypatch, log_extra="cargando Ferretería Inventada 7 ...\n")
    ent = ec.cargar_entradas("prueba", manifest, raiz)
    with pytest.raises(ec.ErrorCorrida, match="Posible fuga"):
        ec.evaluar_corrida(ent, manifest, contexto)


# --- Figura ---


def test_figura_desde_el_resultado(tmp_path):
    from src.construir_kernel import KERNELS

    res = {
        "configuracion": {"corrida": KERNELS["06b-inicial"].corrida},
        "modelo": {"nombre": "qwen3_0.6b_qlora_06b-inicial"},
        "etapas": {"etapas": {"entrenamiento": {"pasos": 120}}},
        "curvas": {
            "ventana_pasos": 50,
            "train_por_ventana": [{"paso_inicial": 1, "paso_final": 50, "pasos": 50, "media": 2.0},
                                  {"paso_inicial": 51, "paso_final": 100, "pasos": 50, "media": 1.0},
                                  {"paso_inicial": 101, "paso_final": 120, "pasos": 20, "media": 0.5}],
            "validacion": [{"paso": 0, "eval_loss": 3.0}, {"paso": 100, "eval_loss": 0.9}, {"paso": 120, "eval_loss": 0.6}],
        },
    }
    ruta = tmp_path / "f.png"
    ec.figura_curvas(res, ruta)
    datos = ruta.read_bytes()
    assert datos.startswith(b"\x89PNG") and b"Software" not in datos and b"matplotlib" not in datos.lower()
