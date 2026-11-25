"""Evaluación final del clasificador en test y hard set (Fase 4, paso 17): una sola vez, en la laptop.

Manda evals/fase4_preregistro.json (SHA-256 SHA_CONFIG): modelos, métricas, comparaciones, uso único y
errores para el análisis. El evaluador es el único del proyecto (src/evaluador_clasificador.py y, para la
regla 5, src/metricas_clasificador.py), sin cambios; la métrica tolerante va en src/metrica_tolerante.py.

Entradas, todas verificadas antes de calcular:
- las predicciones de los 6 LLM, de la salida del kernel en kaggle-output/fase4/ (src/salida_fase4.py:
  bytes y SHA-256 contra mediciones.json, revisión de nombres, ids y compuertas);
- las de los 2 baselines, en data/fase4/baselines/, con el SHA-256 del registro del paso 16
  (results/2026-11-25_fase4_baselines-predicciones.json);
- que cada archivo cubra exactamente los ids de su conjunto; el catálogo SCIAN de la versión fijada.

Modos:
  .venv/Scripts/python -m src.evaluacion_final_fase4                # punto de control: no abre test ni escribe
  .venv/Scripts/python -m src.evaluacion_final_fase4 --validacion   # ensayo en validación: solo imprime
  .venv/Scripts/python -m src.evaluacion_final_fase4 --test         # una sola vez, con el visto bueno de Samuel

--validacion recalcula en validación (los dos afinados con sus respuestas del kernel y los dos baselines) y
lo compara con lo registrado; además arma una maqueta completa del resultado y de la tabla con números de
validación y les pasa los mismos candados que --test, para que una coincidencia de los textos fijos con un
nombre aparezca antes de abrir test. Nunca lee data/splits/test.jsonl ni hard.jsonl.

--test es lo único que lee codigo_act de test y hard (cargar_etiquetas). Se niega si ya existe el resultado,
la tabla o el archivo de errores; exige el árbol limpio y este script en git; calcula todo dos veces en
memoria y solo escribe si los bytes son idénticos y pasan los candados (verificar_privacidad, ids y la
revisión de nombres con la regla acotada contra test y hard). Escribe:
- results/<fecha>_fase4_evaluacion-final.json (regla 4; sin nombres, llaves, ids ni respuestas crudas);
- results/<fecha>_fase4_tabla-final.md, generada desde ese JSON (solo números e ids de modelo);
- data/fase4/errores_4b-lr2e-4-r32_test.jsonl y .csv (ignorados por git: llevan nombres), los 50 errores
  del 4B sorteados según el config; en results/ solo van su conteo, bytes y SHA-256.

El chat de Claude del proyecto propuso el paso; Samuel lo aprobó; Claude Code lo redactó.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import json
import platform
import sys
import traceback
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import src.baselines as bl
import src.salida_fase4 as sf
from src.baselines_fase4 import EstadoGit, _git, comprobar_head, verificar_sin_ids
from src.denue import REPO_RAIZ
from src.evaluador_clasificador import (
    DEFINICION_PAREADA,
    DEFINICIONES,
    NIVELES,
    REMUESTREOS,
    ErrorEvaluador,
    aciertos_por_nivel,
    cargar_codigos,
    cargar_referencias,
    diferencia_pareada_bootstrap,
    evaluar_validacion,
)
from src.evaluar_corrida_clasificador import buscar_nombres_indexada, clasificar, indice_de_nombres, posible_fuga
from src.evaluar_humo import referencias_de_nombres
from src.exploracion_clasificador import ErrorPrivacidad, verificar_privacidad
from src.metrica_tolerante import DEFINICION as DEFINICION_TOLERANTE
from src.metrica_tolerante import evaluar_tolerante
from src.metricas_clasificador import CODIGO_EXISTENTE, ErrorMetricas, categoria, leer_predicciones, limpiar
from src.muestra_humo import ErrorMuestra, codigo_versionado, leer_verificado
from src.normalizacion import normalizar_nombre
from src.preregistro_fase4 import ErrorPreregistro, parsear_jsonl
from src.prompt_clasificador import RE_CODIGO
from src.splits import esta_ignorado, sha256_bytes

SCRIPT = "src/evaluacion_final_fase4.py"
EXPERIMENTO = "fase4_paso17_evaluacion_final"
SEMILLA = 42
RUTA_CONFIG = "evals/fase4_preregistro.json"
SHA_CONFIG = "7de9d3fc1335179540e35f91cba0ac98920f9f00a4ddc28a680b148898b6156e"
RESULTADO_BASELINES = "results/2026-11-25_fase4_baselines-predicciones.json"
PATRON_RESULTADO = "*_fase4_evaluacion-final.json"
PATRON_TABLA = "*_fase4_tabla-final.md"
CARPETA_ERRORES = "data/fase4"
BASE_ERRORES = "errores_4b-lr2e-4-r32_test"
RESERVADOS = ("test", "hard")
SUBCONJUNTOS_HARD = ("total", "ambiguo", "indirecto")
GRUPOS = ("principales", "secundarias", "secundarias_sector")
NIVEL_DE_METRICA = {"accuracy de clase": "clase", "accuracy de sector oficial": "sector"}
ENTRADA_SPLIT = {"test": "clasificador_test", "hard": "clasificador_hard", "validacion": "clasificador_validacion"}
TOP_CODIGOS = 5
# Campos cuyo valor es un código SCIAN respondido (concentración de respuestas): quedan fuera del chequeo de ids.
CAMPOS_CODIGO = frozenset({"codigo", "codigo_valido_mas_predicho"})
DECIMALES = 6
# Pareadas de validación que se pueden recalcular (los base no corren en validación) y dónde está lo registrado.
PAREADAS_VALIDACION = (
    ("tfidf-lr-C10", "06b-lr2e-4-r32", "06b-lr2e-4-r32", ("comparacion", "diferencia_pareada_contra_tfidf")),
    ("tfidf-lr-C10", "4b-lr2e-4-r32", "4b-lr2e-4-r32", ("comparacion", "diferencia_pareada_contra_tfidf")),
    ("06b-lr2e-4-r32", "4b-lr2e-4-r32", "4b-lr2e-4-r32", ("contra_referencia",)),
)
CLAVES_METRICAS = ("regla5", "ic95_accuracy", "por_nivel", "macro_f1_sector", "subgrupos", "concentracion",
                   "clases_en_la_referencia")
COLUMNAS_ERRORES = (
    "id", "nom_estab", "codigo_act", "titulo_act", "respuesta", "categoria_regla5", "codigo_respondido",
    "titulo_respondido", "nivel_mas_fino_acertado", "acierta_sector", "sin_nombre", "generico", "igual_titulo",
)
REFERENCIA_REGLA5 = "Regla 5 de CLAUDE.md, implementada en src/metricas_clasificador.py (DEFINICION), sin cambios."
DEFINICION_AL_TOPE = ("% de las respuestas que llegan al tope de fichas nuevas sin token de fin (16 en los afinados, "
                      "32 en los base), contado en el JSONL de cada etapa del kernel.")


class ErrorFinal(ValueError):
    """PARADA de la evaluación final; el mensaje nunca incluye nombres, llaves, ids ni respuestas."""


ERRORES_PROPIOS = (ErrorFinal, sf.ErrorSalida, ErrorEvaluador, ErrorMetricas, ErrorMuestra, ErrorPreregistro,
                   ErrorPrivacidad)


# --- Entradas ---


@dataclass
class Entradas:
    config: dict
    sha_config: str
    manifest: dict
    codigos: frozenset
    titulos: dict[str, str]
    salida: object  # SimpleNamespace de src/salida_fase4.revisar
    predicciones: dict[str, dict[str, dict[str, str]]]  # modelo → conjunto → {id: respuesta}
    al_tope: dict[str, dict[str, dict[str, bool]]]  # LLM → conjunto → {id: al tope}
    fuentes: dict[str, dict[str, dict]]  # modelo → conjunto → {ruta, sha256, filas}
    registro_baselines: dict
    sha_registro_baselines: str
    validacion: list[dict]  # filas de data/splits/val.jsonl (con marcas)
    referencias_nombres: dict[str, list[dict]]  # los 4 archivos que vio el kernel (id y nom_estab)
    registrados: dict[str, dict] = field(default_factory=dict)  # resultados de validación por modelo
    sha_registrados: dict[str, str] = field(default_factory=dict)
    raiz: Path = REPO_RAIZ


def ids_modelos(config: Mapping) -> list[str]:
    return [m["id"] for m in config["modelos"]]


def llm(config: Mapping) -> list[str]:
    return list(config["metricas"]["tolerante"]["modelos"])


def baselines(config: Mapping) -> list[str]:
    return [m["id"] for m in config["modelos"] if m["tipo"] == "baseline"]


def modelo_config(config: Mapping, mid: str) -> dict:
    return next(m for m in config["modelos"] if m["id"] == mid)


def cargar_entradas(raiz: Path = REPO_RAIZ, manifest: Mapping | None = None,
                    avance: Callable[[str], None] = print) -> Entradas:
    """Todo lo que la evaluación lee, verificado. Nunca abre data/splits/test.jsonl ni hard.jsonl."""
    from src.scian import CATALOGO, leer_catalogo, leer_manifest

    manifest = dict(manifest if manifest is not None else leer_manifest())
    config_bytes = (Path(raiz) / RUTA_CONFIG).read_bytes()
    sha_config = sha256_bytes(config_bytes)
    if sha_config != SHA_CONFIG:
        raise ErrorFinal(f"El SHA-256 de {RUTA_CONFIG} es {sha_config}, no {SHA_CONFIG}; me detengo.")
    config = json.loads(config_bytes.decode("utf-8"))
    for conj, entrada in ENTRADA_SPLIT.items():
        if config["conjuntos"][conj]["sha256"] != manifest["splits"][entrada]["sha256"]:
            raise ErrorFinal(f"El SHA-256 de {conj} del config no es el de splits.{entrada} del manifest.")

    avance("Salida del kernel: descarga, nombres (regla acotada), ids y compuertas (src/salida_fase4.py)...")
    salida = sf.revisar(raiz, manifest)
    if salida.problemas:
        raise ErrorFinal(f"La salida del kernel tiene problemas: {salida.problemas}.")
    avance(f"  {len(salida.etapas)} etapas completas; compuertas "
           + ", ".join(f"{m} {g['local_identicas']}/{g['filas']}" for m, g in salida.compuertas.items())
           + "; sin posible fuga de nombres.")

    refs = sf.referencias(manifest, raiz)
    ids = {"test": [r["id"] for r in refs["test"]], "hard": [r["id"] for r in refs["hard"]]}
    validacion = cargar_referencias(manifest, ENTRADA_SPLIT["validacion"], raiz)
    ids["validacion"] = [r["id"] for r in validacion]
    if sorted(ids["validacion"], key=int) != [r["id"] for r in refs["validacion"]]:
        raise ErrorFinal("Los ids de data/splits/val.jsonl no son los del val.jsonl del paquete.")

    predicciones: dict[str, dict[str, dict[str, str]]] = {}
    al_tope: dict[str, dict[str, dict[str, bool]]] = {}
    fuentes: dict[str, dict[str, dict]] = {}
    for mid in llm(config):
        conjs = ["validacion", *RESERVADOS] if modelo_config(config, mid)["tipo"] == "afinado" else list(RESERVADOS)
        for conj in conjs:
            p = salida.predicciones[(mid, conj)]
            predicciones.setdefault(mid, {})[conj] = sf.respuestas(salida, mid, conj)
            al_tope.setdefault(mid, {})[conj] = sf.al_tope(salida, mid, conj)
            fuentes.setdefault(mid, {})[conj] = {"ruta": f"{sf.CARPETA}/{p['ruta']}", "sha256": p["sha256"],
                                                 "filas": len(p["registros"]), "verificado_contra": "mediciones.json"}

    datos_reg = (Path(raiz) / RESULTADO_BASELINES).read_bytes()
    registro = json.loads(datos_reg.decode("utf-8"))
    if registro.get("configuracion", {}).get("preregistro", {}).get("sha256") != SHA_CONFIG:
        raise ErrorFinal(f"{RESULTADO_BASELINES} no registra el SHA-256 del config.")
    for mid in baselines(config):
        for conj in ("validacion", *RESERVADOS):
            a = registro["predicciones"][mid][conj]
            datos = leer_verificado(Path(raiz) / a["ruta"], a["sha256"])
            predicciones.setdefault(mid, {})[conj] = leer_predicciones(datos)
            fuentes.setdefault(mid, {})[conj] = {"ruta": a["ruta"], "sha256": a["sha256"], "filas": a["filas"],
                                                 "verificado_contra": RESULTADO_BASELINES}
    for mid, por_conj in predicciones.items():
        for conj, pred in por_conj.items():
            if sorted(pred, key=int) != sorted(ids[conj], key=int):
                raise ErrorFinal(f"Las predicciones de {mid} en {conj} no cubren exactamente los ids del conjunto.")
    faltan = [m for m in ids_modelos(config) if set(predicciones.get(m, {})) < set(RESERVADOS)]
    if faltan:
        raise ErrorFinal(f"Faltan predicciones de test o hard de {faltan}.")

    entrada_cat = manifest["catalogos"][CATALOGO[manifest["scian_version"]]]
    titulos = leer_catalogo(Path(raiz) / entrada_cat["ruta"])
    codigos = cargar_codigos(manifest, raiz)
    if set(titulos) != set(codigos):
        raise ErrorFinal("Los títulos del catálogo no son los de sus códigos.")

    registrados, sha_reg = {}, {}
    for mid in baselines(config) + [m for m in llm(config) if modelo_config(config, m)["tipo"] == "afinado"]:
        ruta = modelo_config(config, mid)["resultado_validacion"]
        datos = (Path(raiz) / ruta).read_bytes()
        registrados[mid], sha_reg[mid] = json.loads(datos.decode("utf-8")), sha256_bytes(datos)
    avance(f"  Predicciones de {len(predicciones)} modelos con su SHA-256 y exactamente los ids de cada conjunto; "
           f"catálogo SCIAN {manifest['scian_version']} con {len(codigos):,} clases.")
    return Entradas(config=config, sha_config=sha_config, manifest=manifest, codigos=codigos, titulos=titulos,
                    salida=salida, predicciones=predicciones, al_tope=al_tope, fuentes=fuentes,
                    registro_baselines=registro, sha_registro_baselines=sha256_bytes(datos_reg),
                    validacion=validacion, referencias_nombres=refs, registrados=registrados, sha_registrados=sha_reg,
                    raiz=Path(raiz))


def cargar_etiquetas(manifest: Mapping, config: Mapping, copias: Mapping[str, Sequence[Mapping]],
                     raiz: Path = REPO_RAIZ) -> dict[str, list[dict]]:
    """ABRIR TEST: las filas de test y hard con codigo_act y marcas. Solo lo llama --test."""
    res = {}
    for conj in RESERVADOS:
        e = manifest["splits"][ENTRADA_SPLIT[conj]]
        if e["sha256"] != config["conjuntos"][conj]["sha256"]:
            raise ErrorFinal(f"El SHA-256 de {conj} del manifest no es el del config.")
        regs = parsear_jsonl(leer_verificado(Path(raiz) / e["ruta"], e["sha256"]), e["campos"])
        if len(regs) != e["filas"] or len(regs) != config["conjuntos"][conj]["filas"]:
            raise ErrorFinal(f"Las filas de {conj} no son las del manifest y del config.")
        copia = {r["id"]: r["nom_estab"] for r in copias[conj]}
        if {r["id"]: r["nom_estab"] for r in regs} != copia:
            raise ErrorFinal(f"Las filas de {conj} no son las de su copia sin etiquetas.")
        res[conj] = regs
    tipos = Counter(r["tipo"] for r in res["hard"])
    if dict(tipos) != config["conjuntos"]["hard"]["por_tipo"]:
        raise ErrorFinal("Los tipos del hard set no son los del config.")
    return res


# --- Cálculo ---


def _sin(d: Mapping, claves: Sequence[str]) -> dict:
    return {k: v for k, v in d.items() if k not in claves}


def codigos_mas_predichos(pred: Mapping[str, str], ids: Sequence[str], k: int = TOP_CODIGOS) -> list[dict]:
    """Los k códigos de formato válido más respondidos (en empate, el menor); lo inválido solo se cuenta aparte."""
    conteo = Counter(c for c in (limpiar(pred[i]) for i in ids) if RE_CODIGO.fullmatch(c))
    return [{"codigo": c, "filas": n} for c, n in sorted(conteo.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def metricas_modelo(pred: Mapping[str, str], filas: Sequence[Mapping], codigos, al_tope: Mapping[str, bool] | None,
                    es_llm: bool) -> dict:
    ids = [f["id"] for f in filas]
    sub = {i: pred[i] for i in ids}
    m = _sin(evaluar_validacion(sub, filas, codigos), ("definiciones", "implementacion", "semilla_bootstrap",
                                                       "remuestreos_bootstrap"))
    m["codigos_validos_mas_predichos"] = codigos_mas_predichos(sub, ids)
    if es_llm:
        m["tolerante"] = evaluar_tolerante(sub, filas, codigos)
        k = sum(bool(al_tope[i]) for i in ids)
        m["al_tope"] = {"filas": len(ids), "al_tope": k, "pct_al_tope": round(100 * k / len(ids), 2)}
    return m


def pareadas(preds: Mapping[str, Mapping[str, str]], filas: Sequence[Mapping], codigos, config: Mapping) -> dict:
    """Las pareadas de los grupos del config sobre las filas (en orden de int(id)), cada una con su métrica."""
    filas = sorted(filas, key=lambda f: int(f["id"]))
    aciertos = {m: aciertos_por_nivel(p, filas, codigos) for m, p in preds.items()}
    res = {}
    for grupo in GRUPOS:
        g = config["comparaciones"][grupo]
        nivel = NIVEL_DE_METRICA[g["metrica"]]
        res[grupo] = [{"a": par["a"], "b": par["b"], "metrica": g["metrica"], "signo": f"{par['b']} − {par['a']}",
                       **diferencia_pareada_bootstrap(aciertos[par["a"]][nivel], aciertos[par["b"]][nivel])}
                      for par in g["pares"]]
    return res


def bloque(preds: Mapping[str, Mapping[str, str]], al_tope: Mapping[str, Mapping[str, bool]], filas: Sequence[Mapping],
           codigos, config: Mapping) -> dict:
    filas = sorted(filas, key=lambda f: int(f["id"]))
    llms = set(llm(config))
    return {
        "filas": len(filas),
        "clases_en_la_referencia": len({f["codigo_act"] for f in filas}),
        "modelos": {m: metricas_modelo(preds[m], filas, codigos, al_tope.get(m), m in llms) for m in ids_modelos(config)},
        "pareadas": pareadas(preds, filas, codigos, config),
    }


def _compacto(m: Mapping) -> dict:
    sector = m["por_nivel"]["sector"]
    return {"aciertos": m["regla5"]["aciertos"], "accuracy": m["regla5"]["accuracy"], "ic95": m["ic95_accuracy"],
            "accuracy_sector": sector["accuracy"], "ic95_sector": sector["ic95"],
            "macro_f1_sector": m["macro_f1_sector"]["valor"]}


def fila_tabla(mid: str, tipo: str, m: Mapping) -> dict:
    r5 = m["regla5"]
    return {
        "modelo": mid, "tipo_modelo": tipo, "n": r5["total"], **_compacto(m),
        "pct_formato_invalido": r5["porcentajes"]["formato_invalido"],
        "pct_codigos_inexistentes": r5["pct_codigos_inexistentes"],
        "conteos_regla5": r5["conteos"],
        "clases_existentes_distintas_predichas": m["concentracion"]["clases_existentes_distintas_predichas"],
        "pct_filas_con_el_codigo_mas_predicho": m["concentracion"]["pct_filas_con_ese_codigo"],
        "tolerante_accuracy": m["tolerante"]["accuracy"] if "tolerante" in m else None,
        "tolerante_ic95": m["tolerante"]["ic95"] if "tolerante" in m else None,
        "pct_al_tope": m["al_tope"]["pct_al_tope"] if "al_tope" in m else None,
    }


def errores_4b(pred: Mapping[str, str], filas: Sequence[Mapping], codigos, titulos: Mapping[str, str],
               config: Mapping) -> tuple[list[dict], dict]:
    """Los errores de clase del modelo del config en orden de int(id) y el sorteo de n con la convención del config."""
    ea = config["errores_para_analisis"]
    filas = sorted(filas, key=lambda f: int(f["id"]))
    ac = aciertos_por_nivel({f["id"]: pred[f["id"]] for f in filas}, filas, codigos)
    errores = [i for i, a in enumerate(ac["clase"]) if not a]
    posiciones = np.sort(np.random.default_rng(SEMILLA).permutation(len(errores))[: ea["n"]])
    sel = []
    for p in posiciones:
        i = errores[int(p)]
        f = filas[i]
        resp = pred[f["id"]]
        cat = categoria(resp, codigos)
        limpia = limpiar(resp)
        fino = next((n for n, _ in reversed(NIVELES[:-1]) if ac[n][i]), "ninguno")
        sel.append({
            "id": f["id"], "nom_estab": f["nom_estab"], "codigo_act": f["codigo_act"], "titulo_act": titulos[f["codigo_act"]],
            "respuesta": resp, "categoria_regla5": cat, "codigo_respondido": limpia if RE_CODIGO.fullmatch(limpia) else "",
            "titulo_respondido": titulos.get(limpia, "") if cat == CODIGO_EXISTENTE else "",
            "nivel_mas_fino_acertado": fino, "acierta_sector": ac["sector"][i],
            "sin_nombre": f["sin_nombre"], "generico": f["generico"], "igual_titulo": f["igual_titulo"],
        })
    info = {"modelo": ea["modelo"], "conjunto": ea["conjunto"], "n_errores": len(errores), "n": len(sel),
            "sorteo": ea["sorteo"], "columnas": list(COLUMNAS_ERRORES),
            "nota": "Las filas llevan nombres: solo en data/fase4/ (ignorado). Quién las clasifica: paso 18."}
    return sel, info


def bytes_errores(filas: Sequence[Mapping]) -> dict[str, bytes]:
    """JSONL (UTF-8, \\n) y CSV (UTF-8 con BOM, para Google Sheets) con las mismas columnas y el mismo orden."""
    for f in filas:
        if tuple(f) != COLUMNAS_ERRORES:
            raise ErrorFinal("Una fila de errores no tiene las columnas esperadas en orden.")
    jsonl = "".join(json.dumps(f, ensure_ascii=False) + "\n" for f in filas).encode("utf-8")
    texto = io.StringIO()
    w = csv.DictWriter(texto, fieldnames=COLUMNAS_ERRORES, lineterminator="\n")
    w.writeheader()
    w.writerows(filas)
    return {"jsonl": jsonl, "csv": texto.getvalue().encode("utf-8-sig")}


def comparar(recalculado, registrado, ruta: str = "$") -> list[str]:
    """Rutas (nunca valores de texto) donde dos objetos JSON difieren."""
    if isinstance(recalculado, Mapping) and isinstance(registrado, Mapping):
        res = []
        for k in sorted(set(recalculado) | set(registrado), key=str):
            if k not in recalculado or k not in registrado:
                res.append(f"{ruta}.{k}")
            else:
                res += comparar(recalculado[k], registrado[k], f"{ruta}.{k}")
        return res
    if isinstance(recalculado, list) and isinstance(registrado, list) and len(recalculado) == len(registrado):
        return [r for i, (a, b) in enumerate(zip(recalculado, registrado)) for r in comparar(a, b, f"{ruta}[{i}]")]
    return [] if recalculado == registrado else [ruta]


def _ruta(obj: Mapping, claves: Sequence[str]):
    for k in claves:
        obj = obj[k]
    return obj


def validar(ent: Entradas) -> dict:
    """Ensayo en validación: recálculo de los 4 modelos con predicciones de validación contra lo registrado."""
    config, val, codigos = ent.config, sorted(ent.validacion, key=lambda f: int(f["id"])), ent.codigos
    modelos = [m for m in ids_modelos(config) if "validacion" in ent.predicciones.get(m, {})]
    res: dict = {"filas": len(val), "modelos": {}, "pareadas": []}
    for mid in modelos:
        m = metricas_modelo(ent.predicciones[mid]["validacion"], val, codigos, None, False)
        reg = ent.registrados[mid]["metricas"]
        dif = comparar({k: m[k] for k in CLAVES_METRICAS}, {k: reg[k] for k in CLAVES_METRICAS})
        item = {"recalculado": _compacto(m), "registrado": _compacto(reg), "identico": not dif, "diferencias": dif,
                "por_nivel": {n: m["por_nivel"][n] for n, _ in NIVELES},
                "resultado_registrado": {"ruta": modelo_config(config, mid)["resultado_validacion"],
                                         "sha256": ent.sha_registrados[mid]}}
        mc = modelo_config(config, mid)
        if mc["tipo"] == "afinado":
            ref = sf.leer_referencia_validacion(config, mid, ent.raiz)
            kern = ent.predicciones[mid]["validacion"]
            m_ref = metricas_modelo(ref, val, codigos, None, False)
            item["respuestas_del_kernel_identicas_a_la_referencia"] = sum(kern[i] == ref[i] for i in ref)
            item["referencia_identica_a_lo_registrado"] = not comparar({k: m_ref[k] for k in CLAVES_METRICAS},
                                                                      {k: reg[k] for k in CLAVES_METRICAS})
            item["fuente_recalculo"] = "respuestas de validación del kernel de la Fase 4"
        else:
            item["fuente_recalculo"] = "predicciones de validación de data/fase4/baselines (paso 16)"
        res["modelos"][mid] = item
    aciertos = {mid: aciertos_por_nivel(ent.predicciones[mid]["validacion"], val, codigos) for mid in modelos}
    for a, b, dueño, claves in PAREADAS_VALIDACION:
        reg_obj = _ruta(ent.registrados[dueño], claves)
        for nivel in ("clase", "sector"):
            nuevo = diferencia_pareada_bootstrap(aciertos[a][nivel], aciertos[b][nivel])
            registrado = {k: reg_obj[nivel][k] for k in nuevo}
            dif = comparar(nuevo, registrado)
            res["pareadas"].append({
                "a": a, "b": b, "nivel": nivel, "signo": f"{b} − {a}", "recalculada": nuevo, "registrada": registrado,
                "identica": not dif, "diferencias": dif,
                "registrada_en": {"ruta": modelo_config(config, dueño)["resultado_validacion"], "campo": ".".join(claves),
                                  "sha256": ent.sha_registrados[dueño]},
                "alineacion": "fila por fila por id: las dos listas de aciertos salen de las predicciones de cada modelo "
                              "contra data/splits/val.jsonl, en orden de int(id)",
            })
    res["todo_identico"] = (all(x["identico"] for x in res["modelos"].values())
                            and all(p["identica"] for p in res["pareadas"]))
    return res


def calcular(ent: Entradas, etiquetas: Mapping[str, Sequence[Mapping]], validacion: Mapping) -> tuple[dict, list[dict]]:
    """Todos los números del resultado (sin el contexto de git) y las filas de errores. Función pura."""
    config, codigos = ent.config, ent.codigos
    preds = {c: {m: ent.predicciones[m][c] for m in ids_modelos(config)} for c in RESERVADOS}
    tope = {c: {m: ent.al_tope[m][c] for m in llm(config)} for c in RESERVADOS}
    test = bloque(preds["test"], tope["test"], etiquetas["test"], codigos, config)
    hard = {}
    for sub in SUBCONJUNTOS_HARD:
        filas = [f for f in etiquetas["hard"] if sub == "total" or f["tipo"] == sub]
        hard[sub] = bloque(preds["hard"], tope["hard"], filas, codigos, config)
    tipo = {m["id"]: m["tipo"] for m in config["modelos"]}
    tabla_test = [fila_tabla(m, tipo[m], test["modelos"][m]) for m in ids_modelos(config)]
    tabla_hard = [{"modelo": m, **{s: {"n": hard[s]["filas"], **_compacto(hard[s]["modelos"][m])} for s in SUBCONJUNTOS_HARD}}
                  for m in ids_modelos(config)]
    val_test = []
    for m, v in validacion["modelos"].items():
        t = test["modelos"][m]
        ic = v["recalculado"]["ic95"]
        val_test.append({
            "modelo": m, "validacion": v["recalculado"], "test": _compacto(t),
            "test_dentro_del_ic95_de_validacion": ic["inferior"] <= t["regla5"]["accuracy"] <= ic["superior"],
            "por_nivel": {"validacion": {n: v["por_nivel"][n]["accuracy"] for n, _ in NIVELES},
                          "test": {n: t["por_nivel"][n]["accuracy"] for n, _ in NIVELES}},
        })
    ea = config["errores_para_analisis"]
    filas_err, info_err = errores_4b(ent.predicciones[ea["modelo"]][ea["conjunto"]], etiquetas[ea["conjunto"]], codigos,
                                     ent.titulos, config)
    return {
        "test": test,
        "hard": hard,
        "tabla_test": tabla_test,
        "tabla_hard": tabla_hard,
        "validacion": validacion,
        "validacion_contra_test": val_test,
        "errores_para_analisis": info_err,
    }, filas_err


# --- Resultado, tabla y candados ---


def costos(ent: Entradas) -> dict:
    c = ent.salida.costos
    cpu = ent.registro_baselines.get("cpu", {})
    return {
        "kernel": {k: c[k] for k in ("segundos_kernel", "horas_gpu_kernel", "segundos_etapas", "segundos_fuera_de_etapas",
                                     "duraciones_orquestador_s")},
        "llm": c["por_modelo"],
        "baselines": {"horas_gpu": 0, "cpu_tiempos_s": cpu.get("tiempos_s"), "registro": RESULTADO_BASELINES},
        "nota": c["nota"],
    }


def advertencias(config: Mapping) -> list[str]:
    ch = config["comparaciones"]["hard"]
    return [
        f"Hard set: {config['conjuntos']['hard']['papel']}",
        f"Curación del hard set: {ch['advertencia_curacion']}",
        f"Tipo indirecto: {ch['advertencia_indirecto']}",
        f"Comparaciones múltiples: {config['comparaciones']['comparaciones_multiples']}",
        f"Test contra validación: {config['uso_unico']['test_contra_validacion']}",
        f"Diccionario: {modelo_config(config, 'diccionario')['nota']}",
        f"06b-lr2e-4-r32: {modelo_config(config, '06b-lr2e-4-r32')['nota']}",
        "Métrica tolerante: solo secundaria; la principal es la accuracy de clase de la regla 5.",
    ]


def armar_resultado(*, fecha: str, git: EstadoGit, ent: Entradas, calc: Mapping, archivos_errores: Mapping) -> dict:
    config, sal = ent.config, ent.salida
    return {
        "fecha": fecha,
        "experimento": EXPERIMENTO,
        "commit": git.commit,
        "arbol_limpio": git.arbol_limpio,
        "head_sin_cambios": True,
        "modelo": ids_modelos(config),
        "configuracion": {
            "script": SCRIPT,
            "preregistro": {"ruta": RUTA_CONFIG, "sha256": ent.sha_config},
            "evaluador": "src/evaluador_clasificador.py (evaluar_validacion, intervalo_bootstrap, "
                         "diferencia_pareada_bootstrap) y src/metricas_clasificador.py (regla 5), sin cambios",
            "metrica_tolerante": "src/metrica_tolerante.py",
            "salida_del_kernel": "src/salida_fase4.py",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "semilla_bootstrap": SEMILLA,
            "remuestreos_bootstrap": REMUESTREOS,
            "scian_version": ent.manifest["scian_version"],
            # La regla 5 va como referencia: su texto (DEFINICION de src/metricas_clasificador.py) tiene dos palabras
            # seguidas que, por casualidad, son la llave de un nombre de test (lo encontró la maqueta del ensayo).
            "definiciones": {**DEFINICIONES, "regla5": REFERENCIA_REGLA5, "pareada": DEFINICION_PAREADA,
                             "tolerante": DEFINICION_TOLERANTE,
                             "al_tope": DEFINICION_AL_TOPE, "codigos_validos_mas_predichos":
                             "Los 5 códigos de formato válido (regla 5) más respondidos; en empate, el menor."},
        },
        "semilla": SEMILLA,
        "split": "test congelado con su distribución natural y hard set congelado, evaluados una sola vez",
        "hash_split": {
            "test": config["conjuntos"]["test"]["sha256"],
            "hard": config["conjuntos"]["hard"]["sha256"],
            "validacion": config["conjuntos"]["validacion"]["sha256"],
            "test_copia": ent.manifest["kaggle_datasets"]["denue_scian_evaluacion"]["archivos"]["test.jsonl"]["sha256"],
            "hard_copia": ent.manifest["kaggle_datasets"]["denue_scian_evaluacion"]["archivos"]["hard.jsonl"]["sha256"],
        },
        "hash_split_origen": "data/manifest.json (splits y kaggle_datasets.denue_scian_evaluacion) y "
                             f"{RUTA_CONFIG} (conjuntos)",
        "tamano_muestra": {"test": calc["test"]["filas"], **{f"hard_{s}": calc["hard"][s]["filas"] for s in SUBCONJUNTOS_HARD},
                           "validacion": calc["validacion"]["filas"]},
        "horas_gpu": sal.costos["horas_gpu_kernel"],
        "kernel": {
            "id": sf.ESPEC.id_kernel, "version": sf.OBSERVACIONES["version_kaggle"], "estado_cli": sf.OBSERVACIONES["estado_cli"],
            "estado": sal.mediciones.get("estado"), "etapas_completas": sum(e["estado"] == "completa" for e in sal.etapas),
            "etapas": len(sal.etapas),
            "compuertas": {m: {"identicas": g["local_identicas"], "filas": g["filas"], "minimo": g["minimo"]}
                           for m, g in sal.compuertas.items()},
            "imagen": {"misma_que_el_4b": sal.imagen["misma_imagen_que_el_4b"], "docker_image": sal.imagen["docker_image_metadata"],
                       "pip_freeze_antes_identico_al_del_4b": sal.imagen["pip_freeze_antes_identico_al_del_4b"],
                       "entorno": sal.imagen["entorno"], "controlador": sal.imagen["controlador"]},
            "versiones_no_fijables": (sal.versiones or {}).get("no_fijables"),
            "versiones_fijadas_distintas": (sal.versiones or {}).get("fijadas_distintas"),
            "descarga": {k: sal.descarga[k] for k in ("archivos_bajados", "coinciden_con_kaggle", "con_sha256_de_kaggle")},
            "revision_de_nombres": {"archivos_revisados": sal.nombres["archivos_revisados"],
                                    "posible_fuga": sal.nombres["posible_fuga"]},
        },
        "predicciones": ent.fuentes,
        "registro_baselines": {"ruta": RESULTADO_BASELINES, "sha256": ent.sha_registro_baselines},
        **{k: calc[k] for k in ("tabla_test", "tabla_hard", "validacion_contra_test", "test", "hard", "validacion")},
        "costos": costos(ent),
        "errores_para_analisis": {**calc["errores_para_analisis"], "archivos": dict(archivos_errores)},
        "advertencias": advertencias(config),
    }


def _p(x) -> str:
    return "—" if x is None else f"{100 * x:.1f}"


def _ic(ic) -> str:
    return "—" if not ic else f"{100 * ic['inferior']:.1f}–{100 * ic['superior']:.1f}"


def _acc(x, ic) -> str:
    return "—" if x is None else f"{_p(x)} ({_ic(ic)})"


def _tope(x) -> str:
    return "—" if x is None else f"{x:.1f}"


def _pp(d: Mapping) -> str:
    ic = d["ic95"]
    return f"{100 * d['diferencia']:+.1f} ({100 * ic['inferior']:+.1f} a {100 * ic['superior']:+.1f})"


def tabla_markdown(r: Mapping) -> str:
    """La tabla para las diapositivas, solo desde el JSON del resultado: números e ids de modelo."""
    t = r["tamano_muestra"]
    L = [
        "# Evaluación final del clasificador (Fase 4): test y hard set",
        "",
        f"Fecha {r['fecha']} · commit `{r['commit'][:7]}` · pre-registro `{r['configuracion']['preregistro']['sha256'][:8]}` · "
        f"semilla {r['semilla']} · IC 95 % por bootstrap de {r['configuracion']['remuestreos_bootstrap']:,} remuestreos. "
        "Fuente: INEGI, DENUE (CDMX, 05/2026) y SCIAN 2023. Porcentajes; pareadas en puntos porcentuales.",
        "",
        f"## Test ({t['test']:,} filas, distribución natural)",
        "",
        "| Modelo | Clase (IC 95 %) | Sector (IC 95 %) | Macro-F1 de sector | % formato inválido | % código inexistente "
        "| Clases distintas | Tolerante (IC 95 %) | % al tope |",
        "|---|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for f in r["tabla_test"]:
        L.append(f"| {f['modelo']} | {_acc(f['accuracy'], f['ic95'])} | {_acc(f['accuracy_sector'], f['ic95_sector'])} | "
                 f"{f['macro_f1_sector']:.3f} | {f['pct_formato_invalido']:.1f} | {f['pct_codigos_inexistentes']:.1f} | "
                 f"{f['clases_existentes_distintas_predichas']} | {_acc(f['tolerante_accuracy'], f['tolerante_ic95'])} | "
                 f"{_tope(f['pct_al_tope'])} |")
    L += ["", "## Pareadas en test (b − a, mismas filas)", "",
          "| Grupo | a | b | Métrica | Diferencia (IC 95 %) | Solo a acierta | Solo b acierta |",
          "|---|---|---|---|---|---:|---:|"]
    for g in GRUPOS:
        for p in r["test"]["pareadas"][g]:
            L.append(f"| {g} | {p['a']} | {p['b']} | {NIVEL_DE_METRICA[p['metrica']]} | {_pp(p)} | "
                     f"{p['filas_solo_a_acierta']} | {p['filas_solo_b_acierta']} |")
    L += ["", f"## Hard set (exploratorio: {t['hard_total']} filas; ambiguo {t['hard_ambiguo']}, indirecto "
          f"{t['hard_indirecto']})", "",
          "| Modelo | Total (IC 95 %) | Ambiguo (IC 95 %) | Indirecto (IC 95 %) |", "|---|---|---|---|"]
    for f in r["tabla_hard"]:
        L.append(f"| {f['modelo']} | " + " | ".join(_acc(f[s]["accuracy"], f[s]["ic95"]) for s in SUBCONJUNTOS_HARD) + " |")
    L += ["", "### Pareadas en el hard set (b − a)", "",
          "| Subconjunto | Grupo | a | b | Métrica | Diferencia (IC 95 %) |", "|---|---|---|---|---|---|"]
    for s in SUBCONJUNTOS_HARD:
        for g in GRUPOS:
            for p in r["hard"][s]["pareadas"][g]:
                L.append(f"| {s} | {g} | {p['a']} | {p['b']} | {NIVEL_DE_METRICA[p['metrica']]} | {_pp(p)} |")
    L += ["", "## Validación contra test (accuracy de clase)", "",
          "| Modelo | Validación (IC 95 %) | Test (IC 95 %) | Test dentro del IC de validación |", "|---|---|---|---|"]
    for f in r["validacion_contra_test"]:
        L.append(f"| {f['modelo']} | {_acc(f['validacion']['accuracy'], f['validacion']['ic95'])} | "
                 f"{_acc(f['test']['accuracy'], f['test']['ic95'])} | {'sí' if f['test_dentro_del_ic95_de_validacion'] else 'no'} |")
    L += ["", "### Accuracy por nivel del SCIAN", "",
          "| Modelo | Conjunto | Sector | Subsector | Rama | Subrama | Clase |", "|---|---|---:|---:|---:|---:|---:|"]
    for f in r["validacion_contra_test"]:
        if f["modelo"] in ("4b-lr2e-4-r32", "tfidf-lr-C10"):
            for c in ("validacion", "test"):
                L.append(f"| {f['modelo']} | {c} | " + " | ".join(_p(f["por_nivel"][c][n]) for n, _ in NIVELES) + " |")
    k = r["costos"]
    L += ["", "## Costos", "",
          f"Kernel de inferencia en una T4: {k['kernel']['horas_gpu_kernel']:.2f} h de GPU en total "
          f"({k['kernel']['segundos_fuera_de_etapas']:.0f} s fuera de las etapas). Baselines: CPU de la laptop, 0 h de GPU "
          f"(ajuste de TF-IDF + LR: {k['baselines']['cpu_tiempos_s']['ajuste_s'] / 60:.0f} min).", "",
          "| Modelo | Horas de GPU (sus etapas) | s/fila en test | % al tope en test | % al tope en hard |",
          "|---|---:|---:|---:|---:|"]
    for m, c in k["llm"].items():
        L.append(f"| {m} | {c['horas_gpu_procesos']:.3f} | {c['conjuntos']['test']['s_por_fila']:.3f} | "
                 f"{c['conjuntos']['test']['pct_al_tope']:.1f} | {c['conjuntos']['hard']['pct_al_tope']:.1f} |")
    L += ["", "## Notas", ""] + [f"- {a}" for a in r["advertencias"]] + [""]
    return "\n".join(L)


def revisar_nombres_salida(textos: Mapping[str, str], filas: Sequence[Mapping]) -> dict:
    """Regla acotada sobre los textos a escribir, contra los nombres de `filas`; solo conteos."""
    idx = indice_de_nombres(referencias_de_nombres(filas))
    res = {}
    for nombre, texto in textos.items():
        _, encontrados = buscar_nombres_indexada(texto, idx)
        res[nombre] = clasificar(encontrados, "", set())
    return {"archivos": res, "posible_fuga": sorted(n for n, c in res.items() if posible_fuga(c))}


def sin_codigos_scian(obj, clave: str | None = None):
    """Copia del objeto sin los valores de los campos que, por construcción, son códigos SCIAN respondidos (6 dígitos
    ASCII): un código puede coincidir con un id de 6 dígitos sin serlo. Cualquier otro texto queda igual."""
    if isinstance(obj, str):
        return None if clave in CAMPOS_CODIGO and RE_CODIGO.fullmatch(obj) else obj
    if isinstance(obj, Mapping):
        return {k: sin_codigos_scian(v, k) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sin_codigos_scian(v, clave) for v in obj]
    return obj


def candados(resultado: Mapping, textos: Mapping[str, str], llaves: set[str], ids: set[str],
             filas_nombres: Sequence[Mapping]) -> dict:
    """verificar_privacidad, sin ids y regla acotada; ErrorFinal (o ErrorPrivacidad) si algo falla."""
    verificar_privacidad(resultado, llaves, set())
    try:
        verificar_sin_ids(sin_codigos_scian(resultado), ids)
    except ValueError as e:
        raise ErrorFinal(str(e)) from e
    # Los ids son números chicos: un conteo o un decimal de la tabla puede coincidir con uno sin ser un id, así que
    # no se buscan como números sueltos. Ningún texto del JSON es un id (verificar_sin_ids) y la tabla sale solo del
    # JSON (correr_test la regenera desde él y exige los mismos bytes).
    rev = revisar_nombres_salida(textos, filas_nombres)
    if rev["posible_fuga"]:
        raise ErrorFinal(f"Posible fuga de nombres (regla acotada) en {rev['posible_fuga']}: no se escribe nada.")
    return rev


def llaves_e_ids(ent: Entradas, etiquetas: Mapping[str, Sequence[Mapping]] | None) -> tuple[set[str], set[str]]:
    """Llaves e ids de test, hard, validación y train final (lo que vio el kernel)."""
    llaves, ids = set(), set()
    for regs in [*ent.referencias_nombres.values(), *(etiquetas or {}).values(), ent.validacion]:
        for r in regs:
            ids.add(r["id"])
            k = r.get("llave") or normalizar_nombre(r["nom_estab"])
            if k:
                llaves.add(k)
    return llaves, ids


# --- Maqueta (en --validacion): el resultado completo con números de validación, para probar los candados ---


def etiquetas_maqueta(ent: Entradas) -> tuple[Entradas, dict[str, list[dict]]]:
    """Validación en lugar de test y de hard (tipos alternados) y, para los base, las respuestas de validación del
    afinado de su tamaño. Solo para pasar los candados a los textos fijos; nunca se escribe."""
    val = sorted(ent.validacion, key=lambda f: int(f["id"]))
    hard = [{**f, "tipo": ("ambiguo", "indirecto")[i % 2]} for i, f in enumerate(val)]
    preds, tope = {}, {}
    for m in ids_modelos(ent.config):
        mc = modelo_config(ent.config, m)
        fuente = m if "validacion" in ent.predicciones[m] else ("06b-lr2e-4-r32" if mc["tamano"] == "0.6B" else "4b-lr2e-4-r32")
        preds[m] = {c: ent.predicciones[fuente]["validacion"] for c in RESERVADOS}
        if m in llm(ent.config):
            tope[m] = {c: ent.al_tope[fuente]["validacion"] for c in RESERVADOS}
    maqueta = Entradas(**{**ent.__dict__, "predicciones": preds, "al_tope": tope})
    return maqueta, {"test": val, "hard": hard}


# --- Corridas ---


def estado_git() -> EstadoGit:
    return EstadoGit(commit=_git("rev-parse", "HEAD"),
                     arbol_limpio=_git("status", "--porcelain", "--untracked-files=no") == "",
                     versionado=codigo_versionado(SCRIPT))


def existentes(raiz: Path) -> list[str]:
    res = [p.name for pat in (PATRON_RESULTADO, PATRON_TABLA) for p in sorted((Path(raiz) / "results").glob(pat))]
    res += [p.name for p in sorted((Path(raiz) / CARPETA_ERRORES).glob(f"{BASE_ERRORES}.*"))]
    return res


def preparar(ent: Entradas, etiquetas, validacion, git: EstadoGit, fecha: str) -> dict[str, bytes]:
    """Resultado, tabla y archivos de errores en bytes (sin escribir)."""
    calc, filas_err = calcular(ent, etiquetas, validacion)
    err = bytes_errores(filas_err)
    archivos = {f: {"ruta": f"{CARPETA_ERRORES}/{BASE_ERRORES}.{f}", "bytes": len(b), "sha256": _sha(b)} for f, b in err.items()}
    resultado = armar_resultado(fecha=fecha, git=git, ent=ent, calc=calc, archivos_errores=archivos)
    return {"resultado": bl.a_bytes(resultado), "tabla": tabla_markdown(resultado).encode("utf-8"),
            "errores_jsonl": err["jsonl"], "errores_csv": err["csv"]}


def _sha(b: bytes) -> str:
    return sha256_bytes(b)


def dos_veces(f: Callable[[], dict[str, bytes]]) -> dict[str, bytes]:
    """Calcula dos veces en memoria; exige bytes idénticos."""
    a, b = f(), f()
    distintos = sorted(k for k in a if a[k] != b.get(k))
    if distintos or set(a) != set(b):
        raise ErrorFinal(f"El doble cálculo no da los mismos bytes en {distintos}; no se escribe nada.")
    return a


def correr_validacion(ent: Entradas, avance: Callable[[str], None] = print) -> dict:
    """Ensayo en validación y maqueta con candados. No escribe nada."""
    v1, v2 = validar(ent), validar(ent)
    if bl.a_bytes(v1) != bl.a_bytes(v2):
        raise ErrorFinal("El doble cálculo de validación no da los mismos bytes.")
    maqueta, etq = etiquetas_maqueta(ent)
    git = EstadoGit(commit="0" * 40, arbol_limpio=True, versionado=True)
    datos = dos_veces(lambda: preparar(maqueta, etq, v1, git, "2026-01-01"))
    llaves, ids = llaves_e_ids(ent, None)
    filas_nombres = [r for c in RESERVADOS for r in ent.referencias_nombres[c]]
    rev = candados(json.loads(datos["resultado"]), {"resultado": datos["resultado"].decode("utf-8"),
                                                    "tabla": datos["tabla"].decode("utf-8")}, llaves, ids, filas_nombres)
    avance(f"Maqueta del resultado con números de validación: {len(datos['resultado']):,} bytes de JSON y "
           f"{len(datos['tabla']):,} de tabla, idénticos en el doble cálculo; verificar_privacidad ({len(llaves):,} llaves), "
           f"sin ids ({len(ids):,}) y regla acotada contra test y hard: limpios "
           f"({sum(c['llaves_una_palabra_con_letras'] for c in rev['archivos'].values())} coincidencias de una palabra, "
           "solo contadas).")
    return v1


def correr_test(ent: Entradas, *, raiz: Path, fecha: str, git: Callable[[], EstadoGit] = estado_git,
                etiquetas_de: Callable[..., dict] = cargar_etiquetas, ignorado: Callable[[Path], bool] = esta_ignorado,
                avance: Callable[[str], None] = print) -> dict:
    """La evaluación final, una sola vez."""
    hay = existentes(raiz)
    if hay:
        raise ErrorFinal(f"Ya existen {hay}: la evaluación final ya se hizo; no se repite ni se sobrescribe.")
    inicial = git()
    if not (inicial.arbol_limpio and inicial.versionado):
        raise ErrorFinal(f"{SCRIPT} debe estar en git y el árbol limpio: el commit registrado no sería el código exacto.")
    carpeta_err = Path(raiz) / CARPETA_ERRORES
    if not ignorado(carpeta_err / f"{BASE_ERRORES}.jsonl"):
        raise ErrorFinal(f"{CARPETA_ERRORES}/ no está ignorado por git; me detengo.")
    validacion = validar(ent)
    avance("ABRO TEST: leo codigo_act de data/splits/test.jsonl y hard.jsonl (única vez).")
    etiquetas = etiquetas_de(ent.manifest, ent.config, ent.referencias_nombres, raiz)
    datos = dos_veces(lambda: preparar(ent, etiquetas, validacion, inicial, fecha))
    llaves, ids = llaves_e_ids(ent, etiquetas)
    filas_nombres = [r for c in RESERVADOS for r in etiquetas[c]]
    resultado = json.loads(datos["resultado"])
    if tabla_markdown(resultado).encode("utf-8") != datos["tabla"]:
        raise ErrorFinal("La tabla no se regenera igual desde el JSON.")
    candados(resultado, {"resultado": datos["resultado"].decode("utf-8"), "tabla": datos["tabla"].decode("utf-8")},
             llaves, ids, filas_nombres)
    comprobar_head(git, inicial, avance)
    rutas = {
        "resultado": Path(raiz) / "results" / f"{fecha}_fase4_evaluacion-final.json",
        "tabla": Path(raiz) / "results" / f"{fecha}_fase4_tabla-final.md",
        "errores_jsonl": carpeta_err / f"{BASE_ERRORES}.jsonl",
        "errores_csv": carpeta_err / f"{BASE_ERRORES}.csv",
    }
    for k in ("errores_jsonl", "errores_csv", "tabla", "resultado"):  # el resultado al final: su existencia cierra
        rutas[k].parent.mkdir(parents=True, exist_ok=True)
        bl.escribir_atomico(datos[k], rutas[k], temporal_en=carpeta_err)
        avance(f"Escrito {rutas[k].relative_to(Path(raiz)).as_posix()}: {len(datos[k]):,} bytes, SHA-256 {_sha(datos[k])}.")
    return resultado


# --- Terminal ---


def imprimir_validacion(v: Mapping) -> None:
    print(f"Validación ({v['filas']:,} filas): recalculado contra lo registrado")
    for m, x in v["modelos"].items():
        r, g = x["recalculado"], x["registrado"]
        extra = ""
        if "respuestas_del_kernel_identicas_a_la_referencia" in x:
            extra = (f"; respuestas del kernel idénticas a la referencia: {x['respuestas_del_kernel_identicas_a_la_referencia']:,}"
                     f"; referencia idéntica a lo registrado: {x['referencia_identica_a_lo_registrado']}")
        print(f"  {m}: clase {r['accuracy']} ({r['ic95']['inferior']}–{r['ic95']['superior']}) vs {g['accuracy']} "
              f"({g['ic95']['inferior']}–{g['ic95']['superior']}); sector {r['accuracy_sector']} vs {g['accuracy_sector']}; "
              f"macro-F1 {r['macro_f1_sector']} vs {g['macro_f1_sector']}; idéntico en todo: {x['identico']} "
              f"{x['diferencias'] or ''}{extra}")
    for p in v["pareadas"]:
        n, g = p["recalculada"], p["registrada"]
        print(f"  {p['signo']} ({p['nivel']}): {n['diferencia']:+} ({n['ic95']['inferior']} a {n['ic95']['superior']}; solo a "
              f"{n['filas_solo_a_acierta']}, solo b {n['filas_solo_b_acierta']}) vs registrada {g['diferencia']:+} "
              f"({g['ic95']['inferior']} a {g['ic95']['superior']}); idéntica: {p['identica']} {p['diferencias'] or ''}")
    print(f"Todo idéntico a lo registrado: {v['todo_identico']}")


def imprimir_test(r: Mapping) -> None:
    print(tabla_markdown(r))


def main(argv: Sequence[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--validacion", action="store_true", help="ensayo en validación (solo imprime)")
    grupo.add_argument("--test", action="store_true", help="la evaluación final, una sola vez")
    args = parser.parse_args(argv)
    try:
        ent = cargar_entradas()
        if args.validacion:
            imprimir_validacion(correr_validacion(ent))
        elif args.test:
            imprimir_test(correr_test(ent, raiz=REPO_RAIZ, fecha=datetime.date.today().isoformat()))
        else:
            print(f"PUNTO DE CONTROL: entradas verificadas; no se abrió test ni se escribió nada. Ya existen: "
                  f"{existentes(REPO_RAIZ) or 'ninguno'}.")
    except ERRORES_PROPIOS as e:
        sys.exit(f"ME DETENGO: {e}")
    except Exception as e:  # noqa: BLE001 (sin su mensaje, que podría traer un id, un nombre o una respuesta)
        print(f"ME DETENGO por un error inesperado {type(e).__name__} (sin su mensaje). Marcos:")
        for m in traceback.extract_tb(e.__traceback__):
            print(f"  {Path(m.filename).name}:{m.lineno} en {m.name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
