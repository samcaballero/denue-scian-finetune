"""Kernel de Kaggle de la evaluación final del clasificador: inferencia en test y hard set (Fase 4, paso 15).

Genera las respuestas de los 6 LLM del pre-registro (`evals/fase4_preregistro.json`, incrustado en el
kernel con su SHA-256): los dos afinados (con su adaptador LoRA) y los cuatro modelos base (0-shot y
few-shot), en el orden de `orden_en_el_kernel`. El kernel nunca ve etiquetas de test ni de hard (las
copias del dataset `denue-scian-evaluacion` solo traen id y nom_estab) y no calcula ninguna métrica:
las respuestas se evalúan una sola vez, en la laptop.

Este módulo es el orquestador (el proceso principal). Cada etapa (un modelo en un conjunto) corre en
su propio proceso de Python (`src/etapa_fase4_kaggle.py`), como las configuraciones del barrido: si
una falla, se cuelga o pasa de su tiempo máximo, se marca incompleta, sus salidas no se evalúan y
el kernel sigue con la siguiente. Etapas del orquestador (lo que puede fallar barato va primero):
 1. Config incrustado: SHA-256 y plan de las 14 etapas (`validar_config_kernel`).
 2. nvidia-smi: toda GPU debe ser T4.
 3. Entrada: cada insumo localizado por su ruta y con su SHA-256 (las dos copias, train final,
    validación, adaptadores, adapter_config y predicciones de validación de referencia); las
    carpetas de los datasets solo con sus archivos; ningún archivo con el SHA-256 de test o de hard
    con etiquetas.
 4. Instalación con las versiones fijadas del barrido; si una fijada no queda instalada, aborta.
 5. Comprobaciones en un proceso propio: selección few-shot rehecha, descarga verificada de las dos
    bases y plantilla y largos con los tokenizadores reales en todas las filas de cada modelo. Si
    algo falla, el kernel sale con 1 antes de cargar el primer modelo.
 6. Etapas, en orden. Afinados: validación, test y hard; base: test y hard. Antes de cada una, un
    candado de tiempo; después de validación de cada afinado, la compuerta de reproducción
    (`criterio_de_paro`): si no pasa, ese modelo no corre en test ni en hard.
 7. Cierre: mediciones.json con el SHA-256 de cada archivo de salida.

Salidas en /kaggle/working/fase4/: predicciones/<modelo>/<conjunto>.jsonl (id, respuesta cruda,
fichas del prompt, fichas generadas, si llegó al tope y segundos), etapas/<etapa>.json,
comprobaciones.json, mediciones.json y logs/<etapa>.log. Los JSON y el log solo llevan estados,
conteos, tiempos y hashes: nunca nombres, ids ni respuestas.

El orquestador solo importa la biblioteca estándar y los módulos incrustados del barrido; nunca
importa torch, numpy ni el preregistro, y nunca lee nom_estab. Sale con 0 aunque una etapa falle,
para que Kaggle guarde la salida de las demás.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import src.barrido_kaggle as bk
import src.clasificador_kaggle as kc
import src.humo_kaggle as hk

RUTA_CONFIG = "evals/fase4_preregistro.json"
RAIZ_FUENTES = Path(__file__).resolve().parents[1]  # en Kaggle, /tmp/denue_src
DIR_SALIDA = Path("/kaggle/working/fase4")
DIR_TRABAJO = Path("/tmp/denue_trabajo/fase4")
ARCHIVO_MEDICIONES = "mediciones.json"
ARCHIVO_COMPROBACIONES = "comprobaciones.json"
CARPETA_PREDICCIONES = "predicciones"
CARPETA_ETAPAS = "etapas"
CARPETA_LOGS = "logs"
CARPETA_ENTRADAS = "entradas"  # JSON de entrada de cada proceso, en DIR_TRABAJO
CARPETA_DESCARTADAS = "descartadas"  # salidas de etapas incompletas, en DIR_TRABAJO (no se descargan)

ARCHIVOS_EVALUACION = ("test.jsonl", "hard.jsonl")
ARCHIVOS_CLASIFICADOR = ("train_final.jsonl", "val.jsonl")
ARCHIVOS_DATOS = ARCHIVOS_EVALUACION + ARCHIVOS_CLASIFICADOR
ARCHIVO_DE_CONJUNTO = {"validacion": "val.jsonl", "test": "test.jsonl", "hard": "hard.jsonl", "train_final": "train_final.jsonl"}
CAMPOS_PREDICCION = ("id", "respuesta", "fichas_prompt", "fichas_generadas", "al_tope", "segundos")
CAMPOS_REFERENCIA = ("id", "respuesta")

# Caminos de carga del adaptador, en el orden en que se prueban en validación (src/etapa_fase4_kaggle.py).
METODOS_ADAPTADOR = ("estado_lora", "peft_from_pretrained")
CONJUNTOS_AFINADO = ("validacion", "test", "hard")
CONJUNTOS_BASE = ("test", "hard")
MODO_DE_BASE = {"0-shot": "base-0shot", "few-shot": "base-fewshot"}
TOKENS_DE_FIN = ["<|im_end|>", "<|endoftext|>"]
RE_ID = re.compile(r"[0-9]+")

# --- Tiempos (segundos) ---

# Inferencia de validación de los afinados, de una fila a la vez, con 7 fichas generadas como máximo
# (results/2026-11-24_fase2_4b-lr2e-4-r32.json, para_la_fase_4).
S_FILA_7 = {"0.6B": 0.2239, "4B": 0.4302}
FICHAS_AFINADOS = 7
# Por etapa: importar Unsloth, cargar la base (y el adaptador) y revisar plantilla y largos.
# Medido en la Fase 2: imports de 15 a 54 s y carga de 8 a 14 s; se deja holgura.
S_ARRANQUE = {"0.6B": 120.0, "4B": 180.0}
# Prefill (procesar el prompt) del few-shot, de unas 400 fichas contra unas 63 del 0-shot (paso 15, punto 2).
S_PREFILL_FEWSHOT = {"0.6B": {"mejor": 0.02, "peor": 0.10}, "4B": {"mejor": 0.15, "peor": 0.50}}
S_ORQUESTADOR = 150.0  # GPU, hashes de la entrada e instalación (barrido: 41 s de pip)
S_COMPROBACIONES = 300.0  # descarga verificada de las dos bases (4B: 67 s) y tokenizadores en todas las filas

TIEMPO_KAGGLE_S = 28_800  # el -t de `kaggle kernels push`: 8 h, unas 1.5 veces el peor caso
MARGEN_CIERRE_S = 900
PRESUPUESTO_S = TIEMPO_KAGGLE_S - MARGEN_CIERRE_S
TIEMPO_MAXIMO_COMPROBACIONES_S = 1800
GRACIA_S = 120  # la etapa deja de generar sola este tiempo antes de que el orquestador la mate
FACTOR_CANDADO = bk.FACTOR_CANDADO
FACTOR_TIEMPO_MAXIMO = bk.FACTOR_TIEMPO_MAXIMO


class AbortoFase4(hk.AbortoHumo):
    """Condición que detiene el kernel antes de la primera etapa; el mensaje nunca incluye un nom_estab."""


def informar(mensaje: str) -> None:
    print(f"[fase4 {time.strftime('%H:%M:%S')}] {mensaje}", flush=True)


# --- Config ---


def leer_config(raiz: str | Path, sha256_esperado: str | None) -> tuple[dict, dict]:
    """El config incrustado y su huella; aborta si su SHA-256 no es el incrustado o no es UTF-8 con \\n."""
    datos = (Path(raiz) / RUTA_CONFIG).read_bytes()
    sha = hk.sha256_bytes(datos)
    if sha256_esperado is not None and sha != sha256_esperado:
        raise AbortoFase4(f"El SHA-256 del config es {sha}; el incrustado es {sha256_esperado}.")
    if datos.startswith(b"\xef\xbb\xbf") or b"\r" in datos or not datos.endswith(b"\n"):
        raise AbortoFase4("El config debe ser UTF-8 sin BOM, con saltos \\n y uno al final.")
    return json.loads(datos.decode("utf-8")), {"ruta": RUTA_CONFIG, "sha256": sha, "bytes": len(datos)}


def _entero(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def criterio_de_paro(config: Mapping) -> dict:
    """Modelos, mínimo de respuestas idénticas y filas de la reproducción de validación."""
    rv = config["uso_unico"]["reproduccion_validacion_en_el_kernel"]
    c = rv["criterio_de_paro"]
    if not isinstance(c, Mapping):
        raise AbortoFase4("uso_unico.reproduccion_validacion_en_el_kernel.criterio_de_paro no está fijado.")
    minimo, filas = c["minimo_identicas"], c["filas"]
    if not (_entero(minimo) and _entero(filas) and 0 < minimo <= filas):
        raise AbortoFase4("criterio_de_paro: minimo_identicas y filas deben ser enteros con 0 < mínimo <= filas.")
    return {"modelos": list(rv["modelos"]), "minimo_identicas": minimo, "filas": filas}


def insumos_afinado(m: Mapping) -> dict:
    """Ruta relativa (sufijo dentro de la salida del kernel de origen) y SHA-256 de cada insumo del afinado."""
    carpeta = str(m["origen_kaggle"]["carpeta_en_la_salida"]).strip("/")
    raiz = carpeta.rsplit("/", 1)[0] if "/" in carpeta else ""
    sufijos = {
        "adaptador": f"{carpeta}/adapter_model.safetensors",
        "adaptador_config": f"{carpeta}/adapter_config.json",
        "predicciones_validacion": f"{raiz}/predicciones.jsonl" if raiz else "predicciones.jsonl",
    }
    res = {}
    for clave, sufijo in sufijos.items():
        if not str(m[clave]["ruta"]).endswith("/" + sufijo):
            raise AbortoFase4(f"modelos.{m['id']}.{clave}.ruta no termina en {sufijo}.")
        if not re.fullmatch(r"[0-9a-f]{64}", str(m[clave]["sha256"])):
            raise AbortoFase4(f"modelos.{m['id']}.{clave}.sha256 no es un SHA-256.")
        res[clave] = {"sufijo": sufijo, "sha256": m[clave]["sha256"]}
    return res


def validar_config_kernel(config: Mapping) -> dict:
    """Plan del kernel desde el config: las etapas en orden y el criterio de paro; aborta si algo no cuadra."""
    problemas: list[str] = []
    try:
        modelos = {m["id"]: m for m in config["modelos"]}
        orden = list(config["orden_en_el_kernel"])
        dec, lm = config["decodificacion"], config["largo_maximo"]
        filas = {c: config["conjuntos"][c]["filas"] for c in ("validacion", "test", "hard")}
        crit = criterio_de_paro(config)
    except (KeyError, TypeError) as e:
        raise AbortoFase4(f"Al config le falta una clave del kernel: {e}.") from e
    if len(orden) != 6 or len(set(orden)) != 6:
        problemas.append("orden_en_el_kernel debe nombrar 6 modelos distintos.")
    for grupo, nuevas in (("afinados", "nuevas_afinados"), ("base", "nuevas_base")):
        d = dec[grupo]
        if d.get("do_sample") is not False or d.get("num_beams") != 1 or d.get("lote") != 1:
            problemas.append(f"decodificacion.{grupo}: greedy, num_beams 1 y lote 1.")
        if list(d.get("tokens_de_fin", [])) != TOKENS_DE_FIN:
            problemas.append(f"decodificacion.{grupo}.tokens_de_fin debe ser {TOKENS_DE_FIN}.")
        if d.get("max_new_tokens") != lm.get(nuevas) or not _entero(d.get("max_new_tokens")):
            problemas.append(f"decodificacion.{grupo}.max_new_tokens no es largo_maximo.{nuevas}.")
    if dec["afinados"].get("max_seq_length") != lm.get("general"):
        problemas.append("decodificacion.afinados.max_seq_length no es largo_maximo.general.")
    if dec["base"].get("max_seq_length") != {"0-shot": lm.get("general"), "few-shot": lm.get("few_shot")}:
        problemas.append("decodificacion.base.max_seq_length no es {0-shot: general, few-shot: few_shot}.")

    etapas: list[dict] = []
    afinados: list[str] = []
    for mid in orden:
        m = modelos.get(mid)
        if m is None:
            problemas.append(f"orden_en_el_kernel nombra {mid}, que no está en modelos.")
            continue
        tam = m.get("tamano")
        if tam not in S_FILA_7 or "Kaggle" not in str(m.get("corre_en")):
            problemas.append(f"modelos.{mid}: tamaño sin tiempos medidos o no corre en Kaggle.")
            continue
        base = m.get("base", {})
        if set(base) != {"repo", "revision"} or not re.fullmatch(r"[0-9a-f]{40}", str(base.get("revision"))):
            problemas.append(f"modelos.{mid}.base debe tener repo y una revisión de 40 hexadecimales.")
            continue
        insumos = None
        if m.get("tipo") == "afinado":
            modo, conjs = "afinado", CONJUNTOS_AFINADO
            max_seq, nuevas = dec["afinados"]["max_seq_length"], dec["afinados"]["max_new_tokens"]
            try:
                insumos = insumos_afinado(m)
            except (AbortoFase4, KeyError, TypeError) as e:
                problemas.append(f"modelos.{mid}: insumos del afinado: {e}")
                continue
            afinados.append(mid)
        elif m.get("tipo") == "base" and m.get("modo") in MODO_DE_BASE:
            modo, conjs = MODO_DE_BASE[m["modo"]], CONJUNTOS_BASE
            max_seq, nuevas = dec["base"]["max_seq_length"][m["modo"]], dec["base"]["max_new_tokens"]
        else:
            problemas.append(f"modelos.{mid}: tipo o modo desconocido.")
            continue
        for conj in conjs:
            etapas.append({
                "id": f"{mid}__{conj}",
                "modelo": mid,
                "conjunto": conj,
                "tipo": m["tipo"],
                "tamano": tam,
                "modo": modo,
                "base": {"repo": base["repo"], "revision": base["revision"]},
                "max_seq_length": max_seq,
                "max_new_tokens": nuevas,
                "tokens_de_fin": list(TOKENS_DE_FIN),
                "filas": filas[conj],
                "conjuntos_del_modelo": list(conjs),
                "compuerta": m["tipo"] == "afinado" and conj == "validacion",
                "insumos_afinado": insumos,
            })
    if crit["modelos"] != afinados:
        problemas.append("reproduccion_validacion_en_el_kernel.modelos deben ser los afinados, en el orden del kernel.")
    if crit["filas"] != filas["validacion"]:
        problemas.append("criterio_de_paro.filas no son las filas de validación.")
    if problemas:
        raise AbortoFase4("El config no sirve para el kernel: " + " ".join(problemas))
    return {"etapas": etapas, "criterio": crit}


# --- Tiempos ---


def s_por_fila(etapa: Mapping, caso: str) -> float:
    """Segundos por fila: los de validación de los afinados; el base, en el peor caso, siempre al tope de fichas."""
    if caso not in ("mejor", "peor"):
        raise ValueError("caso debe ser 'mejor' o 'peor'.")
    s = S_FILA_7[etapa["tamano"]]
    if etapa["tipo"] == "base" and caso == "peor":
        s *= etapa["max_new_tokens"] / FICHAS_AFINADOS
    if etapa["modo"] == "base-fewshot":
        s += S_PREFILL_FEWSHOT[etapa["tamano"]][caso]
    return s


def estimacion_s(etapa: Mapping, caso: str = "peor") -> float:
    return round(S_ARRANQUE[etapa["tamano"]] + etapa["filas"] * s_por_fila(etapa, caso), 1)


def estimaciones(plan: Mapping) -> dict:
    """Arranque y comprobaciones, cada etapa en mejor y peor caso, y el total (si todas corren)."""
    etapas = {e["id"]: {"mejor_s": estimacion_s(e, "mejor"), "peor_s": estimacion_s(e, "peor"),
                        "tiempo_maximo_s": round(FACTOR_TIEMPO_MAXIMO * estimacion_s(e, "peor"), 1)}
              for e in plan["etapas"]}
    fijo = S_ORQUESTADOR + S_COMPROBACIONES
    return {
        "arranque_y_comprobaciones_s": fijo,
        "etapas": etapas,
        "total_mejor_s": round(fijo + sum(e["mejor_s"] for e in etapas.values()), 1),
        "total_peor_s": round(fijo + sum(e["peor_s"] for e in etapas.values()), 1),
        "tiempo_kaggle_s": TIEMPO_KAGGLE_S,
        "presupuesto_s": PRESUPUESTO_S,
    }


# --- Entrada ---


def _padre(ruta: str) -> str:
    return ruta.rsplit("/", 1)[0] if "/" in ruta else ""


def revisar_entrada(listado: Sequence[Mapping], hashes: Mapping[str, str], plan: Mapping, esperados: Mapping[str, str],
                    config: Mapping) -> dict:
    """Ruta de cada insumo con su SHA-256 verificado; aborta con todos los problemas juntos.

    `listado`: archivos bajo /kaggle/input (kc.listar_entrada); `hashes`: SHA-256 de cada uno.
    """
    problemas: list[str] = []
    por_nombre: dict[str, list[str]] = {}
    for a in listado:
        por_nombre.setdefault(a["ruta"].rsplit("/", 1)[-1], []).append(a["ruta"])
    rutas: dict[str, str] = {}
    for nombre in ARCHIVOS_DATOS:
        encontrados = por_nombre.get(nombre, [])
        if len(encontrados) != 1:
            problemas.append(f"{nombre}: {len(encontrados)} copias en la entrada.")
        else:
            rutas[nombre] = encontrados[0]
    for grupo in (ARCHIVOS_EVALUACION, ARCHIVOS_CLASIFICADOR):
        if not all(n in rutas for n in grupo):
            continue
        padres = {_padre(rutas[n]) for n in grupo}
        if len(padres) != 1 or "" in padres:
            problemas.append(f"{list(grupo)} no están en una misma carpeta de dataset.")
            continue
        padre = padres.pop()
        dentro = sorted(a["ruta"] for a in listado if a["ruta"].startswith(padre + "/"))
        if dentro != sorted(rutas[n] for n in grupo):
            problemas.append(f"La carpeta de {list(grupo)} tiene {len(dentro) - len(grupo)} archivos de más.")
    afinados: dict[str, dict[str, str]] = {}
    for e in plan["etapas"]:
        if not e["insumos_afinado"] or e["modelo"] in afinados:
            continue
        afinados[e["modelo"]] = {}
        for clave, ins in e["insumos_afinado"].items():
            cand = [a["ruta"] for a in listado if a["ruta"] == ins["sufijo"] or a["ruta"].endswith("/" + ins["sufijo"])]
            if len(cand) != 1:
                problemas.append(f"{e['modelo']}: {len(cand)} archivos {ins['sufijo']} en la entrada.")
            elif hashes[cand[0]] != ins["sha256"]:
                problemas.append(f"{e['modelo']}: el SHA-256 de {ins['sufijo']} no es el del config.")
            else:
                afinados[e["modelo"]][clave] = cand[0]
    for nombre, ruta in rutas.items():
        if hashes[ruta] != esperados[nombre]:
            problemas.append(f"El SHA-256 de {nombre} no es el del manifest.")
    con_etiquetas = {config["conjuntos"]["test"]["sha256"], config["conjuntos"]["hard"]["sha256"]}
    prohibidos = sum(h in con_etiquetas for h in hashes.values())
    if prohibidos:
        problemas.append(f"{prohibidos} archivos de la entrada tienen el SHA-256 de test o de hard con etiquetas.")
    if problemas:
        raise AbortoFase4("Entrada inválida: " + " ".join(problemas))
    return {"rutas": rutas, "afinados": afinados, "archivos_revisados": len(hashes), "con_sha256_de_etiquetas": 0}


# --- Compuerta de reproducción ---


def leer_jsonl(datos: bytes, campos: Sequence[str]) -> list[dict]:
    """Registros con las claves `campos` en orden, ids de dígitos sin repetir; corta solo en "\\n"."""
    texto = datos.decode("utf-8")
    if not texto.endswith("\n") or texto.startswith("﻿"):
        raise ValueError("El JSONL está vacío, tiene BOM o no termina en salto de línea.")
    regs = []
    for linea in texto[:-1].split("\n"):
        r = json.loads(linea)
        if not isinstance(r, dict) or tuple(r) != tuple(campos) or not isinstance(r["id"], str) or not RE_ID.fullmatch(r["id"]):
            raise ValueError("Una línea no tiene las claves esperadas o su id no es de dígitos.")
        regs.append(r)
    if len({r["id"] for r in regs}) != len(regs):
        raise ValueError("El JSONL tiene ids repetidos.")
    return regs


def comparar_con_referencia(nuevas: bytes | None, referencia: bytes, criterio: Mapping) -> dict:
    """¿El afinado reproduce su validación? Solo conteos: filas, mismos ids e idénticas."""
    ref = {r["id"]: r["respuesta"] for r in leer_jsonl(referencia, CAMPOS_REFERENCIA)}
    res = {"filas_referencia": len(ref), "filas_criterio": criterio["filas"], "minimo_identicas": criterio["minimo_identicas"]}
    if nuevas is None:
        return {**res, "filas_nuevas": 0, "mismos_ids": False, "identicas": 0, "cumple": False,
                "motivo": "La etapa de validación no terminó completa."}
    try:
        nue = {r["id"]: r["respuesta"] for r in leer_jsonl(nuevas, CAMPOS_PREDICCION)}
    except ValueError as e:
        return {**res, "filas_nuevas": 0, "mismos_ids": False, "identicas": 0, "cumple": False, "motivo": str(e)}
    mismos = set(nue) == set(ref) and len(ref) == criterio["filas"]
    identicas = sum(nue.get(i) == v for i, v in ref.items())
    cumple = mismos and identicas >= criterio["minimo_identicas"]
    motivo = None if cumple else ("Los ids no son los de la referencia." if not mismos else "Menos idénticas que el mínimo.")
    return {**res, "filas_nuevas": len(nue), "mismos_ids": mismos, "identicas": identicas, "cumple": cumple, "motivo": motivo}


# --- Procesos ---


@dataclass
class Dependencias:
    """Lo que el orquestador usa del mundo exterior; las pruebas lo reemplazan."""

    dir_salida: Path
    dir_trabajo: Path
    dir_entrada: Path
    consultar_gpus: Callable[[], list[dict]]
    instalar: Callable[[hk.Registro, Path], None]
    versiones: Callable[[Sequence[str]], dict[str, str | None]]
    memoria_gpu: Callable[[], dict[int, int]]
    estimar: Callable[[Mapping], float] = estimacion_s
    presupuesto_s: float = PRESUPUESTO_S
    tiempo_maximo_comprobaciones_s: float = TIEMPO_MAXIMO_COMPROBACIONES_S
    modulo_etapa: str = "src.etapa_fase4_kaggle"
    raiz_fuentes: Path = RAIZ_FUENTES
    python: str = sys.executable
    reloj: Callable[[], float] = time.time
    espera_s: float = 2.0


def dependencias_kaggle() -> Dependencias:
    return Dependencias(
        dir_salida=DIR_SALIDA,
        dir_trabajo=DIR_TRABAJO,
        dir_entrada=hk.DIR_ENTRADA,
        consultar_gpus=hk.consultar_gpus,
        instalar=bk._instalar,
        versiones=bk._versiones,
        memoria_gpu=bk._memoria_gpu,
    )


def _entorno(dep: Dependencias) -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONPATH": str(dep.raiz_fuentes),
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "TOKENIZERS_PARALLELISM": "false",
    }


def _lanzar(dep: Dependencias, nombre: str, opcion: str, entrada: Mapping, tiempo_maximo_s: float) -> dict:
    ruta = dep.dir_trabajo / CARPETA_ENTRADAS / f"{nombre}.json"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(entrada, ensure_ascii=False), encoding="utf-8")
    return bk.ejecutar(
        [dep.python, "-u", "-m", dep.modulo_etapa, opcion, str(ruta)], env=_entorno(dep), cwd=dep.dir_trabajo,
        log=dep.dir_salida / CARPETA_LOGS / f"{nombre}.log", tiempo_maximo_s=tiempo_maximo_s,
        eco=lambda linea: print(f"[{nombre}] {linea}", flush=True), espera_s=dep.espera_s,
    )


def _leer_json(ruta: Path) -> dict:
    if not ruta.is_file():
        return {}
    try:
        d = json.loads(ruta.read_bytes())
    except (OSError, ValueError):
        return {"legible": False}
    return d if isinstance(d, dict) else {"legible": False}


def _contexto(entrada_rev: Mapping, dep: Dependencias, fuentes: Mapping[str, str]) -> dict:
    """Lo que necesita cada proceso: rutas absolutas de los datos y SHA-256 esperados."""
    return {
        "rutas": {n: str(dep.dir_entrada / r) for n, r in entrada_rev["rutas"].items()},
        "afinados": {m: {k: str(dep.dir_entrada / r) for k, r in d.items()} for m, d in entrada_rev["afinados"].items()},
        "fuentes": dict(fuentes),
        "salida": str(dep.dir_salida),
        "raiz_fuentes": str(dep.raiz_fuentes),
    }


def correr_comprobaciones(reg: hk.Registro, ctx: Mapping, esperados: Mapping[str, str], dep: Dependencias) -> dict:
    ej = _lanzar(dep, "comprobaciones", "--comprobaciones", {**ctx, "esperados": dict(esperados)},
                 dep.tiempo_maximo_comprobaciones_s)
    comp = _leer_json(dep.dir_salida / ARCHIVO_COMPROBACIONES)
    res = {**ej, "ok": comp.get("ok") is True, "log": f"{CARPETA_LOGS}/comprobaciones.log"}
    reg.datos["comprobaciones"] = {"proceso": res, "resultado": comp}
    if ej["matado_por_tiempo"] or ej["codigo_de_salida"] != 0 or not res["ok"]:
        raise AbortoFase4(
            f"Las comprobaciones iniciales no pasaron (código {ej['codigo_de_salida']}, matado por tiempo "
            f"{ej['matado_por_tiempo']}); no se carga ningún modelo. Ver {ARCHIVO_COMPROBACIONES}."
        )
    return comp


def ruta_prediccion(dir_salida: Path, etapa: Mapping) -> Path:
    return Path(dir_salida) / CARPETA_PREDICCIONES / etapa["modelo"] / f"{etapa['conjunto']}.jsonl"


def _filas_validas(ruta: Path) -> int | None:
    """Filas del JSONL de predicciones si todas tienen las claves esperadas e ids distintos; si no, None."""
    try:
        return len(leer_jsonl(ruta.read_bytes(), CAMPOS_PREDICCION))
    except (OSError, ValueError):
        return None


def _descartar(dep: Dependencias, etapa: Mapping) -> bool:
    """Saca de la salida las predicciones de una etapa que no quedó completa (no se evalúan)."""
    pred = ruta_prediccion(dep.dir_salida, etapa)
    if not pred.exists():
        return False
    destino = dep.dir_trabajo / CARPETA_DESCARTADAS / etapa["modelo"] / pred.name
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(pred), str(destino))
    return True


def correr_etapa(reg: hk.Registro, etapa: Mapping, ctx: Mapping, esperados: Mapping[str, str], dep: Dependencias,
                 t_arranque: float, metodo_adaptador: str | None) -> dict:
    """Una etapa en su propio proceso, con candado de tiempo y tiempo máximo; solo conteos en el resultado."""
    nombre = etapa["id"]
    estimado = dep.estimar(etapa)
    transcurrido = dep.reloj() - t_arranque
    decision = bk.decidir(estimado, transcurrido, dep.presupuesto_s)
    res: dict = {"id": nombre, "modelo": etapa["modelo"], "conjunto": etapa["conjunto"], "estimado_peor_s": estimado,
                 "transcurrido_al_empezar_s": round(transcurrido, 1), **decision}
    if not decision["lanzar"]:
        reg.falla(f"etapa_{nombre}", f"Omitida por tiempo: {decision['motivo']}.")
        informar(f"{nombre}: omitida por tiempo")
        return {**res, "estado": "omitida"}
    tmax = decision["tiempo_maximo_s"]
    limite = round(tmax - min(GRACIA_S, tmax / 2), 1)
    entrada = {**ctx, "etapa": dict(etapa), "esperados": dict(esperados), "limite_s": limite,
               "metodo_adaptador": metodo_adaptador}
    informar(f"{nombre}: empieza (estimado peor {estimado / 60:.0f} min; tiempo máximo {tmax / 60:.0f} min)")
    ej = _lanzar(dep, nombre, "--etapa", entrada, tmax)
    memoria = dep.memoria_gpu()
    medida = _leer_json(dep.dir_salida / CARPETA_ETAPAS / f"{nombre}.json")
    pred = ruta_prediccion(dep.dir_salida, etapa)
    archivo = medida.get("archivo") if isinstance(medida.get("archivo"), dict) else {}
    # Solo booleanos y números deciden: los textos del JSON de la etapa pasan por tapar_llaves.
    ok = (
        not ej["matado_por_tiempo"]
        and ej["codigo_de_salida"] == 0
        and medida.get("completa") is True
        and pred.is_file()
        and archivo.get("filas") == etapa["filas"]
        and archivo.get("bytes") == pred.stat().st_size
        and _filas_validas(pred) == etapa["filas"]
    )
    res.update({
        **ej,
        "log": f"{CARPETA_LOGS}/{nombre}.log",
        "limite_de_generacion_s": limite,
        "memoria_gpu0_al_salir_mib": memoria.get(0),
        "memoria_liberada": memoria.get(0) is not None and memoria[0] <= bk.MEMORIA_LIBRE_MAXIMA_MIB,
        "medicion": medida,
    })
    if not ok and _descartar(dep, etapa):
        res["salida_descartada"] = True
    if ok:
        res["predicciones"] = {"ruta": pred.relative_to(dep.dir_salida).as_posix(), "sha256": hk.sha256_archivo(pred),
                               "filas": archivo["filas"], "bytes": archivo["bytes"]}
    elif ej["matado_por_tiempo"]:
        reg.falla(f"etapa_{nombre}", f"Pasó de su tiempo máximo ({tmax:.0f} s); se mató el proceso.")
    elif ej["codigo_de_salida"] != 0:
        reg.falla(f"etapa_{nombre}", f"El proceso salió con código {ej['codigo_de_salida']} ({medida.get('motivo')}).")
    else:
        reg.falla(f"etapa_{nombre}", "La etapa no dejó predicciones completas y verificadas.")
    if not res["memoria_liberada"]:
        reg.falla(f"memoria_{nombre}", f"La GPU 0 tiene {memoria.get(0)} MiB en uso al salir el proceso.")
    res["estado"] = "completa" if ok else "incompleta"
    inf = medida.get("inferencia") if isinstance(medida.get("inferencia"), dict) else {}
    informar(f"{nombre}: {res['estado']} en {ej['segundos'] / 60:.1f} min; {inf.get('filas', 0)} filas; "
             f"{inf.get('s_por_fila')} s/fila; {inf.get('al_tope')} al tope; GPU 0 con {memoria.get(0)} MiB al salir")
    return res


def orquestar(reg: hk.Registro, esperados: Mapping[str, str], fuentes: Mapping[str, str], dep: Dependencias,
              t_arranque: float) -> None:
    with reg.etapa("config"):
        config, huella = leer_config(dep.raiz_fuentes, fuentes.get(RUTA_CONFIG))
        plan = validar_config_kernel(config)
        reg.datos["config"] = huella
        reg.datos["plan"] = [{k: e[k] for k in ("id", "modelo", "conjunto", "tipo", "tamano", "modo", "max_seq_length",
                                                "max_new_tokens", "filas", "compuerta")} for e in plan["etapas"]]
        reg.datos["criterio_de_paro"] = plan["criterio"]
        reg.datos["estimaciones"] = estimaciones(plan)
        informar(f"config {huella['sha256']}: {len(plan['etapas'])} etapas; peor caso "
                 f"{reg.datos['estimaciones']['total_peor_s'] / 3600:.2f} h")

    with reg.etapa("gpu"):
        gpus = dep.consultar_gpus()
        hk.verificar_t4(gpus)
        reg.datos["gpu"] = {"nvidia_smi": gpus, "machine_shape_pedido": hk.MACHINE_SHAPE}

    with reg.etapa("entrada"):
        listado = kc.listar_entrada(dep.dir_entrada)
        hashes = {a["ruta"]: hk.sha256_archivo(dep.dir_entrada / a["ruta"]) for a in listado}
        rev = revisar_entrada(listado, hashes, plan, esperados, config)
        reg.datos["entrada"] = {"listado": listado, **{k: rev[k] for k in ("rutas", "afinados", "archivos_revisados",
                                                                           "con_sha256_de_etiquetas")}}
        informar(f"entrada: {len(listado)} archivos revisados; insumos con su SHA-256; ninguno con el de test o hard con etiquetas")

    with reg.etapa("instalacion"):
        dep.instalar(reg, dep.dir_salida)

    with reg.etapa("versiones"):
        nombres = sorted({p.split("==")[0] for p in bk.PAQUETES_BARRIDO} | set(bk.NO_FIJABLES))
        vers = bk.revisar_versiones(dep.versiones(nombres), gpus[0].get("driver") if gpus else None)
        reg.datos["versiones"] = vers
        reg.datos["advertencias"] = vers["advertencias"]
        if vers["fijadas_distintas"]:
            raise AbortoFase4(f"Versiones fijadas con pip que no quedaron instaladas: {sorted(vers['fijadas_distintas'])}.")
        for k, v in vers["advertencias"].items():
            informar(f"ADVERTENCIA: {k} es {v['instalada']}; en 06b-inicial era {v['06b_inicial']}")

    ctx = _contexto(rev, dep, fuentes)
    with reg.etapa("comprobaciones"):
        correr_comprobaciones(reg, ctx, esperados, dep)
        informar("comprobaciones iniciales: todo pasa")

    referencias = {m: (dep.dir_entrada / d["predicciones_validacion"]).read_bytes() for m, d in rev["afinados"].items()}
    reg.datos["etapas"] = []
    reg.datos["reproduccion_validacion"] = {}
    compuertas: dict[str, dict] = {}
    for etapa in plan["etapas"]:
        mid = etapa["modelo"]
        if etapa["tipo"] == "afinado" and not etapa["compuerta"] and not compuertas.get(mid, {}).get("cumple"):
            reg.datos["etapas"].append({"id": etapa["id"], "modelo": mid, "conjunto": etapa["conjunto"], "estado": "no_corre",
                                        "motivo": "El modelo no pasó la reproducción de validación."})
            informar(f"{etapa['id']}: no corre (el modelo no pasó la reproducción de validación)")
            reg.escribir()
            continue
        metodo = compuertas.get(mid, {}).get("metodo_adaptador")
        with reg.etapa(f"etapa_{etapa['id']}"):
            try:
                r = correr_etapa(reg, etapa, ctx, esperados, dep, t_arranque, metodo)
            except Exception as e:  # el error de una etapa nunca tumba a las demás
                reg.falla(f"etapa_{etapa['id']}", e)
                r = {"id": etapa["id"], "modelo": mid, "conjunto": etapa["conjunto"], "estado": "incompleta",
                     "motivo": f"Error del orquestador en la etapa: {type(e).__name__}.",
                     "salida_descartada": _descartar(dep, etapa)}
            reg.datos["etapas"].append(r)
        if etapa["compuerta"]:
            datos = ruta_prediccion(dep.dir_salida, etapa).read_bytes() if r["estado"] == "completa" else None
            g = comparar_con_referencia(datos, referencias[mid], plan["criterio"])
            adaptador = r.get("medicion", {}).get("adaptador") if isinstance(r.get("medicion"), dict) else None
            indice = adaptador.get("metodo_indice") if isinstance(adaptador, dict) else None
            g["metodo_adaptador"] = METODOS_ADAPTADOR[indice] if _entero(indice) and 0 <= indice < len(METODOS_ADAPTADOR) else None
            compuertas[mid] = g
            reg.datos["reproduccion_validacion"][mid] = g
            if not g["cumple"]:
                reg.falla(f"reproduccion_{mid}", f"{g['identicas']} de {g['filas_referencia']} idénticas; mínimo "
                                                f"{g['minimo_identicas']}. {g['motivo']}")
            informar(f"{mid}: {g['identicas']} de {g['filas_referencia']} respuestas de validación idénticas (mínimo "
                     f"{g['minimo_identicas']}); {'sigue con test y hard' if g['cumple'] else 'NO corre en test ni hard'}")
            reg.escribir()

    with reg.etapa("cierre"):
        reg.datos["archivos_salida"] = kc.archivos_salida(dep.dir_salida, (ARCHIVO_MEDICIONES, ARCHIVO_MEDICIONES + ".tmp"))
        todas = all(e.get("estado") == "completa" for e in reg.datos["etapas"])
        reg.datos["estado"] = "completo" if todas and not reg.datos["fallas"] else "completo_con_fallas"


def main(
    esperados: Mapping[str, str],
    t_arranque: float | None = None,
    fuentes: Mapping[str, str] | None = None,
    dep: Dependencias | None = None,
) -> None:
    """Punto de entrada en Kaggle. `esperados` (SHA-256 de los 4 archivos de datos) y `fuentes` los incrusta el generador."""
    t_arranque = time.time() if t_arranque is None else t_arranque
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(line_buffering=True)
    hk.informar = informar
    if set(esperados) != set(ARCHIVOS_DATOS):
        raise AbortoFase4(f"Los SHA-256 esperados deben ser exactamente los de {list(ARCHIVOS_DATOS)}.")
    dep = dep or dependencias_kaggle()
    dep.dir_salida.mkdir(parents=True, exist_ok=True)
    dep.dir_trabajo.mkdir(parents=True, exist_ok=True)
    reg = hk.Registro(dep.dir_salida / ARCHIVO_MEDICIONES, t_arranque)
    reg.datos["descripcion"] = (
        "Inferencia de la evaluación final del clasificador en test y hard set (Fase 4, paso 15): una etapa por "
        "proceso. Sin etiquetas de test ni de hard y sin métricas."
    )
    reg.datos["configuracion"] = {
        "paquetes_fijados": list(bk.PAQUETES_BARRIDO),
        "no_fijables": list(bk.NO_FIJABLES),
        "tiempo_kaggle_s": TIEMPO_KAGGLE_S,
        "presupuesto_s": dep.presupuesto_s,
        "factor_candado": FACTOR_CANDADO,
        "factor_tiempo_maximo": FACTOR_TIEMPO_MAXIMO,
        "gracia_s": GRACIA_S,
        "tiempo_maximo_comprobaciones_s": dep.tiempo_maximo_comprobaciones_s,
        "memoria_libre_maxima_mib": bk.MEMORIA_LIBRE_MAXIMA_MIB,
    }
    reg.datos["fuentes_incrustadas_sha256"] = dict(fuentes or {})
    reg.datos["datos_sha256_esperados"] = dict(esperados)
    reg.escribir()
    try:
        orquestar(reg, esperados, fuentes or {}, dep, t_arranque)
    except hk.AbortoHumo as e:
        reg.datos["estado"] = "abortado"
        reg.datos["aborto"] = str(e)
        reg.escribir()
        informar(f"ABORTO: {e}")
        sys.exit(1)
    except Exception:
        reg.datos["estado"] = "error"
        reg.datos["error"] = traceback.format_exc()[-4000:]
        reg.escribir()
        print(reg.datos["error"], flush=True)
        sys.exit(1)
    reg.escribir()
    informar(f"estado final: {reg.datos['estado']}; fallas: {len(reg.datos['fallas'])}")


def _cli(argv: Sequence[str] | None = None) -> None:
    """En la laptop: imprime el plan y las estimaciones de tiempo con el config del repo."""
    parser = argparse.ArgumentParser(description="Plan y estimaciones del kernel de la Fase 4.")
    parser.add_argument("--estimar", action="store_true", required=True)
    parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    config, huella = leer_config(RAIZ_FUENTES, None)
    plan = validar_config_kernel(config)
    est = estimaciones(plan)
    print(f"Config {huella['sha256']}; criterio: {plan['criterio']['minimo_identicas']} de {plan['criterio']['filas']}")
    print(f"Arranque y comprobaciones: {est['arranque_y_comprobaciones_s']:.0f} s")
    for e in plan["etapas"]:
        x = est["etapas"][e["id"]]
        print(f"  {e['id']}: {e['filas']} filas; mejor {x['mejor_s']:.0f} s; peor {x['peor_s']:.0f} s; "
              f"tiempo máximo {x['tiempo_maximo_s']:.0f} s")
    print(f"Total: mejor {est['total_mejor_s'] / 3600:.2f} h; peor {est['total_peor_s'] / 3600:.2f} h; "
          f"-t {TIEMPO_KAGGLE_S} s ({TIEMPO_KAGGLE_S / 3600:.1f} h); presupuesto {PRESUPUESTO_S} s")


if __name__ == "__main__":
    _cli()
