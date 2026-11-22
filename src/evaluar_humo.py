"""Evaluación local de la prueba de punta a punta (Fase 2, paso 6c).

Lee la salida que bajó `kaggle kernels output` a `kaggle-output/humo-6b/` (ignorada por git) y:

1. verifica que cada archivo bajado tenga el tamaño y el SHA-256 que calculó el kernel en
   Kaggle, y que ninguno tenga un SHA-256 del manifest (no salieron datos);
2. busca nombres de negocio de la muestra (`nom_estab` original y llaves normalizadas) en el
   log, en `humo_mediciones.json`, en los demás archivos de texto y, aparte, en las
   predicciones; las muestras se leen con su SHA-256 verificado contra el manifest;
3. resume cada etapa del kernel (terminó bien, falló o se omitió);
4. calcula las métricas de la regla 5 con `src/metricas_clasificador.py` sobre
   `predicciones.jsonl`, tras verificar que sus ids son exactamente los de `humo_val.jsonl`;
5. mide el rendimiento por modelo y lote y estima las horas de GPU de una época;
6. arma `results/2026-11-21_fase2_prueba-humo.json` (no reportable en la charla), que pasa por
   la misma revisión de nombres antes de escribirse.

Se detiene (PARADA) si un archivo bajado no coincide con lo que calculó Kaggle, si la revisión
de nombres encuentra una posible fuga o si los ids no son los de `humo_val.jsonl`. Nunca abre
`test.jsonl`. Los archivos bajados solo se leen como datos (JSON y texto); el GGUF y el
adaptador solo se hashean. En la terminal solo imprime conteos, códigos, hashes y tiempos.

Uso:
  .venv/Scripts/python -m src.evaluar_humo              # verifica e imprime el resumen
  .venv/Scripts/python -m src.evaluar_humo --escribir   # además escribe el JSON de results/
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

from src.construir_kernel import ID_KERNEL, hashes_del_manifest
from src.denue import REPO_RAIZ
from src.humo_kaggle import (
    ARCHIVO_MEDICIONES,
    ARCHIVO_PREDICCIONES,
    ARCHIVO_TRAIN,
    ARCHIVO_VAL,
    BASE_16_ESPEJO,
    LOTE,
    MODELO_06B,
    MODELO_4B,
    NOMBRE_INVENTADO,
    PAQUETES,
    PASOS_4B,
    leer_jsonl_verificado,
    sha256_archivo,
)
from src.metricas_clasificador import DEFINICION, ErrorMetricas, evaluar, leer_predicciones, limpiar
from src.normalizacion import normalizar_nombre
from src.prompt_clasificador import FIN_DE_TURNO, RE_CODIGO
from src.scian import CATALOGO, leer_catalogo

CARPETA_RELATIVA = "kaggle-output/humo-6b"
ARCHIVO_LOG = "denue-scian-humo-6b.log"  # lo escribe la CLI de Kaggle 2.2.5 junto a la salida
ARCHIVO_RESULTADO = "results/2026-11-21_fase2_prueba-humo.json"
RUTA_KERNEL = "notebooks/kaggle-humo/humo_6b.py"
VERSION_KERNEL = 1
SECCION = "kaggle_datasets"
ENTRADA = "denue_scian_humo"
SCRIPT = "src/evaluar_humo.py"
SEMILLA = 42

# El kernel no puede hashearse a sí mismo y el log lo escribe Kaggle: son los únicos archivos
# bajados sin SHA-256 calculado en Kaggle.
SIN_HASH_DE_KAGGLE = frozenset({ARCHIVO_MEDICIONES, ARCHIVO_LOG})
EXTENSIONES_BINARIAS = (".gguf", ".safetensors")
MINIMO_CRUDO = 3  # largo mínimo de un nom_estab para buscarlo como subcadena exacta

TAMANOS_TRAIN = (50_000, 75_000, 130_000, 190_000)
FILAS_VALIDACION = 1000  # validación congelada completa (data/manifest.json, splits)
TOPE_TORCH = (2, 15)  # unsloth 2026.11.9 exige torch<2.15.0
REPOS_ESPERADOS = frozenset(
    (m["repo"], m["revision"]) for m in (MODELO_06B, MODELO_4B, BASE_16_ESPEJO)
)
MODELOS = {MODELO_06B["repo"]: "qwen3_0.6b", MODELO_4B["repo"]: "qwen3_4b"}

# Observaciones de la corrida que no están en la salida (sesión del 2026-11-21, paso 6b).
OBSERVACIONES = {
    "estado_cli": "COMPLETE",
    "inicio_cli_utc": "2026-11-21T22:51:41+00:00",  # lastRunTime de `kaggle kernels list --mine`, sin zona; se toma como UTC
    "ultimo_running_utc": "2026-11-21T23:00:57+00:00",  # `kaggle kernels status`
    "complete_observado_utc": "2026-11-21T23:05:59+00:00",  # `kaggle kernels status`
    "pagina_kaggle_min": 16,  # "Successful" en la página del kernel, visto por Samuel
}


class ErrorHumo(RuntimeError):
    """PARADA del paso 6c; el mensaje nunca incluye un nom_estab, una llave ni una respuesta."""


def sha256_bytes(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


# --- Archivos bajados ---


def inventario(carpeta: str | Path) -> list[dict]:
    """Ruta relativa, bytes y SHA-256 de cada archivo de la carpeta, en orden."""
    carpeta = Path(carpeta)
    return [
        {"ruta": p.relative_to(carpeta).as_posix(), "bytes": p.stat().st_size, "sha256": sha256_archivo(p)}
        for p in sorted(carpeta.rglob("*"))
        if p.is_file()
    ]


def verificar_descarga(inv: Sequence[Mapping], mediciones: Mapping, hashes_manifest: set[str]) -> dict:
    """Compara lo bajado con `archivos_salida` del kernel; PARADA si algo no coincide."""
    locales = {a["ruta"]: a for a in inv}
    kaggle = {a["ruta"]: a for a in mediciones.get("archivos_salida", [])}
    faltan = sorted(set(kaggle) - set(locales))
    distintos = sorted(
        r for r in set(kaggle) & set(locales)
        if (locales[r]["bytes"], locales[r]["sha256"]) != (kaggle[r]["bytes"], kaggle[r]["sha256"])
    )
    extra = sorted(set(locales) - set(kaggle) - SIN_HASH_DE_KAGGLE)
    sin_archivo = sorted(SIN_HASH_DE_KAGGLE - set(locales))
    con_hash_del_manifest = sorted(r for r, a in locales.items() if a["sha256"] in hashes_manifest)
    inf = mediciones.get("inferencia", {}).get("archivo", {})
    predicciones_ok = locales.get(ARCHIVO_PREDICCIONES, {}).get("sha256") == inf.get("sha256")
    gguf = mediciones.get("gguf", {})
    gguf_ok = (not gguf.get("exportado")) or locales.get(gguf.get("archivo", ""), {}).get("sha256") == gguf.get("sha256")
    res = {
        "archivos_bajados": len(locales),
        "con_sha256_de_kaggle": len(kaggle),
        "coinciden_con_kaggle": len(set(kaggle) & set(locales)) - len(distintos),
        "sin_sha256_de_kaggle": sorted(SIN_HASH_DE_KAGGLE & set(locales)),
        "faltan": faltan,
        "distintos": distintos,
        "extra": extra,
        "con_sha256_del_manifest": con_hash_del_manifest,
        "predicciones_coinciden_con_la_inferencia": predicciones_ok,
        "gguf_coincide": gguf_ok,
    }
    if faltan or distintos or extra or sin_archivo or con_hash_del_manifest or not predicciones_ok or not gguf_ok:
        raise ErrorHumo(
            f"La descarga no coincide con lo que calculó Kaggle: faltan {len(faltan)}, distintos {len(distintos)}, "
            f"extra {len(extra)}, sin log o mediciones {len(sin_archivo)}, con hash del manifest {len(con_hash_del_manifest)}, "
            f"predicciones {'ok' if predicciones_ok else 'distintas'}, GGUF {'ok' if gguf_ok else 'distinto'}."
        )
    return res


def leer_log(datos: bytes) -> list[dict]:
    """Entradas del log de Kaggle: arreglo JSON de {stream_name, time, data}; si no lo es, texto plano."""
    texto = datos.decode("utf-8")
    try:
        arr = json.loads(texto)
    except json.JSONDecodeError:
        return [{"stream_name": None, "time": None, "data": texto}]
    if not isinstance(arr, list) or not all(isinstance(x, dict) and isinstance(x.get("data"), str) for x in arr):
        raise ErrorHumo("El log no es un arreglo de entradas con texto en 'data'.")
    return [{"stream_name": x.get("stream_name"), "time": x.get("time"), "data": x["data"]} for x in arr]


def texto_log(entradas: Sequence[Mapping]) -> str:
    return "".join(e["data"] for e in entradas)


# --- Revisión de nombres ---


def referencias_de_nombres(filas: Sequence[Mapping]) -> dict:
    """Lo que se busca: nom_estab originales (>= MINIMO_CRUDO caracteres) y llaves no vacías."""
    llaves = {k for k in (normalizar_nombre(r["nom_estab"]) for r in filas) if k}
    return {
        "nombres": sorted({r["nom_estab"].strip() for r in filas if len(r["nom_estab"].strip()) >= MINIMO_CRUDO}),
        "varias": frozenset(k for k in llaves if " " in k),
        "una": frozenset(k for k in llaves if " " not in k),
    }


def buscar_nombres(texto: str, ref: Mapping) -> dict:
    """Conteos y números de línea (nunca el texto) de nombres o llaves que aparecen en `texto`.

    (i) cada nom_estab original como subcadena exacta; (ii) cada llave como palabras completas
    de una línea normalizada con `normalizar_nombre`, separando las llaves de una palabra.
    """
    lineas = texto.split("\n")
    crudos = [n for n in ref["nombres"] if n in texto]
    lineas_crudo = sorted({i for i, linea in enumerate(lineas, 1) for n in crudos if n in linea}) if crudos else []
    varias, una = set(), set()
    lineas_varias, lineas_una, largos_una = set(), set(), []
    for i, linea in enumerate(lineas, 1):
        normal = normalizar_nombre(linea)
        if not normal:
            continue
        rodeada = f" {normal} "
        for k in ref["varias"]:
            if f" {k} " in rodeada:
                varias.add(k)
                lineas_varias.add(i)
        for palabra in set(normal.split()) & ref["una"]:
            una.add(palabra)
            lineas_una.add(i)
    largos_una = sorted(len(k) for k in una)
    return {
        "nombres_crudos": len(crudos),
        "lineas_nombres_crudos": lineas_crudo,
        "llaves_varias_palabras": len(varias),
        "lineas_llaves_varias_palabras": sorted(lineas_varias),
        "llaves_una_palabra": len(una),
        "lineas_llaves_una_palabra": sorted(lineas_una),
        "largos_llaves_una_palabra": largos_una,
    }


def posible_fuga(r: Mapping) -> bool:
    return bool(r["nombres_crudos"] or r["llaves_varias_palabras"])


def alguna_coincidencia(r: Mapping) -> bool:
    return posible_fuga(r) or bool(r["llaves_una_palabra"])


def textos_de_json(obj) -> str:
    """Todas las claves y valores de texto de un objeto JSON, uno por línea."""
    res: list[str] = []

    def recorrer(o):
        if isinstance(o, str):
            res.append(o)
        elif isinstance(o, Mapping):
            for k, v in o.items():
                res.append(str(k))
                recorrer(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                recorrer(v)

    recorrer(obj)
    return "\n".join(res)


def revisar_privacidad(textos: Mapping[str, str], mediciones: Mapping, ref: Mapping, filas: int) -> dict:
    """Revisión de nombres de cada archivo de texto; predicciones.jsonl va aparte."""
    archivos = {nombre: buscar_nombres(t, ref) for nombre, t in sorted(textos.items()) if nombre != ARCHIVO_PREDICCIONES}
    archivos[f"{ARCHIVO_MEDICIONES} (valores)"] = buscar_nombres(textos_de_json(mediciones), ref)
    predicciones = buscar_nombres(textos.get(ARCHIVO_PREDICCIONES, ""), ref)
    fugas = sorted(n for n, r in archivos.items() if posible_fuga(r))
    return {
        "filas_de_referencia": filas,
        "nombres_buscados": len(ref["nombres"]),
        "llaves_varias_palabras_buscadas": len(ref["varias"]),
        "llaves_una_palabra_buscadas": len(ref["una"]),
        "nombre_inventado_es_llave": normalizar_nombre(NOMBRE_INVENTADO) in (ref["varias"] | ref["una"]),
        "tapados_por_el_kernel": len(mediciones.get("tapados", [])),
        "archivos": archivos,
        "predicciones": predicciones,
        "archivos_con_posible_fuga": fugas,
        "posible_fuga": bool(fugas) or posible_fuga(predicciones),
    }


def _sin_largos(r: Mapping) -> dict:
    """Resultado de una búsqueda sin los largos de las llaves (solo van a la terminal)."""
    return {k: v for k, v in r.items() if k != "largos_llaves_una_palabra"}


def privacidad_para_resultado(p: Mapping) -> dict:
    return {
        **{k: v for k, v in p.items() if k not in ("archivos", "predicciones")},
        "archivos": {n: _sin_largos(r) for n, r in p["archivos"].items()},
        "predicciones": _sin_largos(p["predicciones"]),
        "metodo": (
            f"(i) cada nom_estab de humo_train.jsonl y humo_val.jsonl de {MINIMO_CRUDO} caracteres o más como "
            "subcadena exacta; (ii) cada llave (normalizar_nombre) como palabras completas de cada línea normalizada. "
            "Posible fuga: (i) o (ii) con llaves de dos o más palabras. Las coincidencias de una sola palabra se listan por línea."
        ),
    }


def candado_resultado(resultado: Mapping, ref: Mapping) -> None:
    """El JSON de results/ no puede tener ninguna coincidencia, ni de una sola palabra."""
    texto = json.dumps(resultado, ensure_ascii=False, indent=2)
    for r in (buscar_nombres(texto, ref), buscar_nombres(textos_de_json(resultado), ref)):
        if alguna_coincidencia(r):
            raise ErrorHumo(
                "El JSON de resultados coincide con la muestra: "
                f"{r['nombres_crudos']} nombres, {r['llaves_varias_palabras']} llaves de varias palabras y "
                f"{r['llaves_una_palabra']} de una (líneas {r['lineas_llaves_una_palabra'][:20]}); no se escribe."
            )


# --- Etapas del kernel ---


def version_menor(version: str, tope: tuple[int, int]) -> bool:
    m = re.match(r"([0-9]+)\.([0-9]+)", str(version))
    return bool(m) and (int(m.group(1)), int(m.group(2))) < tope


def resumen_perdida(perdida: Sequence[Mapping], k: int = 10) -> dict:
    valores = [h.get("loss") for h in perdida]
    finitos = [v for v in valores if isinstance(v, (int, float)) and math.isfinite(v)]
    res = {"pasos_registrados": len(valores), "no_finitos": len(valores) - len(finitos)}
    if finitos:
        n = max(1, min(k, len(finitos) // 2))  # los primeros y los últimos no se enciman
        res.update(
            {
                "primera": round(finitos[0], 4),
                "ultima": round(finitos[-1], 4),
                f"media_primeros_{n}": round(sum(finitos[:n]) / n, 4),
                f"media_ultimos_{n}": round(sum(finitos[-n:]) / n, 4),
                "minima": round(min(finitos), 4),
                "lr_primer_paso": perdida[0].get("learning_rate"),
                "lr_ultimo_paso": perdida[-1].get("learning_rate"),
            }
        )
    return res


def _baja(p: Mapping) -> bool:
    primeros = next((v for k, v in p.items() if k.startswith("media_primeros_")), None)
    ultimos = next((v for k, v in p.items() if k.startswith("media_ultimos_")), None)
    return primeros is not None and ultimos is not None and ultimos < primeros


def _etapa(comprobaciones: Mapping[str, bool], **datos) -> dict:
    if {"estado", "comprobaciones"} & set(datos):
        raise ValueError("Los datos de una etapa no pueden llamarse estado ni comprobaciones.")
    return {
        "estado": "ok" if all(comprobaciones.values()) else "fallo",
        "comprobaciones": {k: bool(v) for k, v in comprobaciones.items()},
        **datos,
    }


def _tabla_ok(ejemplo: Mapping) -> bool:
    """Etiquetas del ejemplo inventado: -100 en todo el prompt; el código y <|im_end|> copian su id, al final."""
    tabla = ejemplo.get("tabla", [])
    con = [i for i, f in enumerate(tabla) if f["etiqueta"] != -100]
    return (
        bool(con)
        and con == list(range(con[0], len(tabla)))
        and all(tabla[i]["etiqueta"] == tabla[i]["id"] for i in con)
        and [tabla[i]["token"] for i in con] == [*ejemplo.get("codigo", ""), FIN_DE_TURNO]
        and ejemplo.get("fichas_con_etiqueta") == [*ejemplo.get("codigo", ""), FIN_DE_TURNO]
    )


def resumen_etapas(m: Mapping, *, esperados: Mapping[str, str], fuentes_actuales: Mapping[str, str],
                   kernel_igual_al_generado: bool, descarga: Mapping, filas_train: int, filas_val: int) -> dict:
    """Terminó bien, falló o se omitió, etapa por etapa, con su dato clave (sin nombres)."""
    e: dict[str, dict] = {}
    g = m.get("gpu", {})
    nombres_gpu = [x.get("nombre", "") for x in g.get("nvidia_smi", [])]
    e["gpu"] = _etapa(
        {
            "todas_t4": bool(nombres_gpu) and all("T4" in n for n in nombres_gpu),
            "cuda_visible_devices_0": g.get("cuda_visible_devices") == "0",
            "torch_ve_una_sola_gpu": g.get("torch_device_count") == 1,
            "torch_ve_una_t4": "T4" in str(g.get("torch_device_name")),
            "sin_bf16_nativo": g.get("bf16_nativo") is False,
        },
        gpus_en_la_sesion=nombres_gpu,
        memoria_total_mib_torch=g.get("memoria_total_mib"),
        capacidad_de_computo=g.get("capacidad_de_computo"),
    )

    d = m.get("datos", {})
    comp = m.get("comprobaciones", {})
    e["datos"] = _etapa(
        {
            **{f"sha256_{a}": d.get(a, {}).get("sha256_verificado") is True and d.get(a, {}).get("sha256") == h for a, h in esperados.items()},
            "d_sin_nombres_con_marca_especial": comp.get("d_nombres_con_marca_especial") == 0,
            "fuentes_incrustadas_iguales_a_src": dict(m.get("fuentes_incrustadas_sha256", {})) == dict(fuentes_actuales),
            "kernel_escrito_igual_al_generado": kernel_igual_al_generado,
            "filas_train": d.get(ARCHIVO_TRAIN, {}).get("filas") == filas_train,
            "filas_val": d.get(ARCHIVO_VAL, {}).get("filas") == filas_val,
        },
        filas={a: d.get(a, {}).get("filas") for a in esperados},
        clases={a: d.get(a, {}).get("clases") for a in esperados},
    )

    versiones = m.get("entorno", {}).get("versiones", {})
    fijadas = dict(p.split("==") for p in PAQUETES)
    cambios = m.get("instalacion", {}).get("cambios_freeze", {})
    tocados = set(cambios.get("agregados", {})) | set(cambios.get("quitados", {})) | set(cambios.get("cambiados", {}))
    e["instalacion"] = _etapa(
        {
            "versiones_fijadas": all(versiones.get(k) == v for k, v in fijadas.items()),
            "torch_menor_que_2_15": version_menor(versiones.get("torch", ""), TOPE_TORCH),
            "pip_no_toco_torch": "torch" not in tocados,
        },
        torch=versiones.get("torch"),
        python=m.get("entorno", {}).get("python"),
        torch_cuda=m.get("entorno", {}).get("torch_cuda"),
        segundos_pip_install=m.get("instalacion", {}).get("segundos_pip_install"),
        pip_agrego=sorted(cambios.get("agregados", {})),
        pip_cambio=dict(cambios.get("cambiados", {})),
        pip_quito=sorted(cambios.get("quitados", {})),
    )

    m06, m4, b16 = m.get("modelo_06b", {}), m.get("modelo_4b", {}), m.get("base_16_bits", {})
    cache = {(x.get("repo"), x.get("revision")) for x in m.get("descargas_hf", [])}
    rev = {
        "qwen3_0.6b_revision_fijada": m06.get("revision_descargada") == MODELO_06B["revision"] and m06.get("name_or_path") == MODELO_06B["repo"],
        "qwen3_0.6b_nf4": m06.get("cuantizacion", {}).get("bnb_4bit_quant_type") == "nf4",
        "espejo_main_es_la_revision_fijada": b16.get("camino") != "espejo" or b16.get("main_actual_del_espejo") == BASE_16_ESPEJO["revision"],
        "cache_solo_repos_y_revisiones_fijados": cache <= REPOS_ESPERADOS,
    }
    if "omitido" not in m4:
        rev["qwen3_4b_revision_fijada"] = m4.get("revision_descargada") == MODELO_4B["revision"] and m4.get("name_or_path") == MODELO_4B["repo"]
        rev["qwen3_4b_nf4"] = m4.get("cuantizacion", {}).get("bnb_4bit_quant_type") == "nf4"
    e["revisiones"] = _etapa(rev, cache_hf=[{"repo": r, "revision": v} for r, v in sorted(cache)])

    t, ej = m.get("tokens", {}), m.get("ejemplo_etiquetas", {})
    a = comp.get("a_prompt_generacion_igual_a_plantilla", {})
    e["plantilla_y_tokens"] = _etapa(
        {
            "a_prompt_igual_a_la_plantilla": a.get("distintos") == 0 and a.get("textos") == filas_train + filas_val + 1,
            "c_tokens_especiales_un_solo_token": comp.get("c_tokens_especiales_un_solo_token", {}).get("ok") is True,
            "d_sin_marca_especial": comp.get("d_nombres_con_marca_especial") == 0,
            "sin_filas_de_mas_de_128": t.get("mayores_a_max_seq") == 0,
            "prompt_es_prefijo": t.get("prompt_no_es_prefijo") == 0,
            "etiquetas_solo_codigo_y_fin": t.get("etiquetas_distintas_de_codigo_y_fin") == 0,
            "tabla_del_ejemplo_inventado": _tabla_ok(ej),
        },
        a_textos=a.get("textos"),
        b_solo_registro=comp.get("b_texto_entrenamiento_frente_a_plantilla", {}).get("conteo"),
        largo_max=t.get("largo_max"),
        largo_medio=t.get("largo_medio"),
        fichas_etiquetadas_por_fila=t.get("fichas_etiquetadas_por_fila"),
        ejemplo_inventado={"fichas": ej.get("fichas"), "fichas_con_etiqueta": len(ej.get("fichas_con_etiqueta", []))},
    )

    ent = m.get("entrenamiento_06b", {})
    perd = resumen_perdida(ent.get("perdida", []))
    e["entrenamiento_06b"] = _etapa(
        {
            "fp16": ent.get("fp16") is True,
            "pasos_de_una_epoca": ent.get("estadisticas", {}).get("pasos_total") == math.ceil(filas_train / LOTE),
            "perdida_finita": perd["no_finitos"] == 0 and perd["pasos_registrados"] > 0,
            "perdida_baja": _baja(perd),
        },
        lote=ent.get("lote"),
        perdida=perd,
        segundos=ent.get("segundos_train_total"),
    )

    inf = m.get("inferencia", {})
    gen = inf.get("generation_config", {})
    e["inferencia"] = _etapa(
        {
            "filas_de_validacion": inf.get("filas") == filas_val,
            "greedy": gen.get("do_sample") is False and gen.get("num_beams") == 1,
            "todas_terminan_en_token_de_fin": inf.get("terminacion", {}).get("token_de_fin") == inf.get("filas"),
        },
        terminacion=inf.get("terminacion"),
        fichas_generadas_max=inf.get("fichas_generadas_max"),
        respaldo_parametros_de_muestreo=inf.get("respaldo_parametros_de_muestreo"),
        s_por_fila=inf.get("s_por_fila"),
    )

    c = b16.get("comparacion_tensores", {})
    e["espejo_16_bits"] = _etapa(
        {
            "camino_conocido": b16.get("camino") in ("espejo", "exacto", "exacto_local"),
            "tensores_identicos": b16.get("camino") != "espejo" or c.get("identicos") is True,
            "exportable": b16.get("exportar") is True,
        },
        camino=b16.get("camino"),
        base_resuelta_por_unsloth=b16.get("base_resuelta_por_unsloth"),
        tensores={k: c.get(k) for k in ("tensores_oficial", "tensores_espejo", "comunes")},
        distintos=len(c.get("distintos", [])),
        solo_oficial=c.get("solo_oficial"),
        solo_espejo=c.get("solo_espejo"),
        lm_head_atado=c.get("solo_un_lado_aceptado_por_atadura"),
    )

    gg = m.get("gguf", {})
    e["gguf"] = _etapa(
        {
            "exportado": gg.get("exportado") is True,
            "sha256_igual_a_lo_bajado": descarga.get("gguf_coincide") is True,
            "plantilla_igual_al_tokenizador": gg.get("plantilla_gguf_igual_a_tokenizador") is True,
            "un_solo_gguf": gg.get("otros_gguf_generados") == [],
        },
        archivo=gg.get("archivo"),
        bytes=gg.get("bytes"),
        sha256=gg.get("sha256"),
        prompt_inventado_fichas=gg.get("prompt_inventado", {}).get("fichas"),
    )

    lotes = m.get("lotes_06b", {})
    e["lotes_06b"] = _etapa(
        {f"lote_{k}_sin_quedarse_sin_memoria": bool(v) and not v.get("sin_memoria") for k, v in sorted(lotes.items(), key=lambda kv: int(kv[0]))}
        or {"hay_mediciones": False}
    )

    if "omitido" in m4:
        e["modelo_4b"] = {"estado": "omitido", "motivo": m4["omitido"]}
    else:
        l16 = m4.get("lote_16", {})
        p4 = resumen_perdida(l16.get("perdida", []))
        e["modelo_4b"] = _etapa(
            {
                "revision_fijada": m4.get("revision_descargada") == MODELO_4B["revision"],
                "tokenizador_igual_al_06b": m4.get("tokenizador_igual_al_06b") is True,
                f"lote_16_{PASOS_4B}_pasos": l16.get("estadisticas", {}).get("pasos_total") == PASOS_4B,
                "perdida_finita": p4["no_finitos"] == 0 and p4["pasos_registrados"] > 0,
                **{f"{k}_sin_quedarse_sin_memoria": not v.get("sin_memoria") for k, v in sorted(m4.items()) if k.startswith("lote_") and k != "lote_16"},
            },
            perdida_lote_16=p4,
            duraciones_s=m4.get("duraciones_s"),
        )

    fallas = m.get("fallas", [])
    e["estado_final"] = _etapa(
        {"completo": m.get("estado") == "completo", "sin_fallas": not fallas},
        estado_del_kernel=m.get("estado"),
        etapas_con_falla=[f.get("etapa") for f in fallas],
    )
    return {"todas_ok": all(x["estado"] == "ok" for x in e.values()), "etapas": e}


# --- Rendimiento y estimación de horas ---


def corridas(m: Mapping) -> list[tuple[str, int, Mapping, str]]:
    """(modelo, lote, corrida, origen) de cada medición de entrenamiento del kernel."""
    res: list[tuple[str, int, Mapping, str]] = []
    ent = m.get("entrenamiento_06b")
    if ent:
        res.append((MODELO_06B["repo"], int(ent["lote"]), ent, "entrenamiento de la prueba (1 época)"))
    for k, v in sorted(m.get("lotes_06b", {}).items(), key=lambda kv: int(kv[0])):
        res.append((MODELO_06B["repo"], int(k), v, "medición desechable de lote"))
    m4 = m.get("modelo_4b", {})
    for k in sorted((k for k in m4 if re.fullmatch(r"lote_[0-9]+", k)), key=lambda k: int(k[5:])):
        res.append((MODELO_4B["repo"], int(k[5:]), m4[k], f"{PASOS_4B} pasos" if int(k[5:]) == LOTE else "medición desechable de lote"))
    return res


def medir(modelo: str, lote: int, c: Mapping, origen: str) -> dict:
    """Ejemplos/s, fichas/s, s/paso y VRAM pico de una corrida, sin los pasos de calentamiento."""
    base = {"modelo": modelo, "lote": lote, "origen": origen}
    if c.get("sin_memoria"):
        return {**base, "sin_memoria": True}
    est = c["estadisticas"]
    pasos = c.get("pasos", [])[est["pasos_calentamiento"]:]
    segundos = sum(p["s"] for p in pasos)
    ejemplos = sum(p["ejemplos"] for p in pasos)
    reales = sum(p["fichas_reales"] for p in pasos)
    relleno = sum(p["fichas_con_relleno"] for p in pasos)
    recalculado = ejemplos / segundos if segundos else None
    return {
        **base,
        "sin_memoria": False,
        "pasos_total": est["pasos_total"],
        "pasos_medidos": est["pasos_medidos"],
        "pasos_calentamiento_excluidos": est["pasos_calentamiento"],
        "ejemplos_por_s": est["ejemplos_por_s"],
        "fichas_reales_por_s": est["fichas_por_s"],
        "fichas_con_relleno_por_s": round(relleno / segundos, 1) if segundos else None,
        "relleno_pct": round(100 * (1 - reales / relleno), 1) if relleno else None,
        "s_por_paso_media": est["s_por_paso_media"],
        "s_por_paso_mediana": est["s_por_paso_mediana"],
        "s_por_paso_p90": est["s_por_paso_p90"],
        "vram_pico_mib": c.get("vram_pico_mib"),
        "recalculo_desde_los_pasos_coincide": recalculado is not None and abs(recalculado - est["ejemplos_por_s"]) <= 0.005 * est["ejemplos_por_s"],
    }


def rendimiento(m: Mapping) -> list[dict]:
    return [medir(*c) for c in corridas(m)]


def costo_fijo(m: Mapping) -> dict[str, dict]:
    """Segundos medidos de lo que se paga en cada corrida antes de entrenar, por modelo."""
    d = m.get("duraciones_s", {})
    comun = {"instalacion": d.get("instalacion"), "imports": d.get("imports")}
    res = {MODELO_06B["repo"]: {**comun, "descarga": d.get("descarga_06b"), "carga": d.get("carga_06b")}}
    m4 = m.get("modelo_4b", {}).get("duraciones_s", {})
    if m4:
        res[MODELO_4B["repo"]] = {**comun, "descarga": m4.get("descarga"), "carga": m4.get("carga")}
    for v in res.values():
        v["total"] = round(sum(x for x in v.values() if isinstance(x, (int, float))), 1)
    return res


def estimar_horas(rend: Sequence[Mapping], tamanos: Sequence[int], fijo: Mapping[str, Mapping], s_por_fila: Mapping[str, float | None]) -> dict:
    """Horas de GPU de 1 época por modelo, con el lote más rápido que no se quedó sin memoria."""
    res: dict[str, dict] = {}
    for modelo in sorted({r["modelo"] for r in rend}):
        validos = [r for r in rend if r["modelo"] == modelo and not r.get("sin_memoria")]
        if not validos:
            res[MODELOS.get(modelo, modelo)] = {"modelo": modelo, "sin_lote_valido": True}
            continue
        mejor = max(validos, key=lambda r: (r["ejemplos_por_s"], -r["lote"]))
        f = fijo.get(modelo, {}).get("total")
        tabla = []
        for n in tamanos:
            s = n / mejor["ejemplos_por_s"]
            pasos = math.ceil(n / mejor["lote"])
            tabla.append(
                {
                    "filas_train": n,
                    "pasos": pasos,
                    "horas_entrenamiento": round(s / 3600, 2),
                    "horas_contraste_mediana_s_por_paso": round(pasos * mejor["s_por_paso_mediana"] / 3600, 2),
                    "horas_con_costo_fijo": round((s + f) / 3600, 2) if f is not None else None,
                }
            )
        sf = s_por_fila.get(modelo)
        res[MODELOS.get(modelo, modelo)] = {
            "modelo": modelo,
            "lote": mejor["lote"],
            "ejemplos_por_s": mejor["ejemplos_por_s"],
            "lotes_considerados": sorted(r["lote"] for r in validos),
            "costo_fijo_s": dict(fijo.get(modelo, {})),
            f"inferencia_validacion_{FILAS_VALIDACION}_filas_s": round(sf * FILAS_VALIDACION, 1) if sf else None,
            "tabla": tabla,
            "horas_entrenamiento_por_lote": {
                str(r["lote"]): {str(n): round(n / r["ejemplos_por_s"] / 3600, 2) for n in tamanos} for r in validos
            },
        }
    return res


ADVERTENCIAS_ESTIMACION = [
    "Solo estimaciones; la decisión del lote y del tope por clase es del chat de Claude del proyecto y de Samuel.",
    "Horas = filas / (ejemplos/s) / 3600, con ejemplos/s sin los 5 pasos de calentamiento; contraste: ceil(filas/lote) x mediana de s/paso.",
    "Pocos pasos medidos (20 en las mediciones de lote, 45 en el 4B con lote 16 y 120 en el 0.6B con lote 16).",
    "Los largos salen de una muestra de 2,000 filas de train; el train completo puede tener filas más largas.",
    "La VRAM pico sale de pocos pasos; no se midieron lotes mayores que 64 en el 0.6B ni que 32 en el 4B.",
    "Una sola T4 (CUDA_VISIBLE_DEVICES=0) aunque la sesión tiene dos.",
    "El costo fijo no incluye el arranque del contenedor de Kaggle ni la evaluación del test; la inferencia del 4B no se midió.",
]


# --- Tiempo ---


def _utc(texto: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(texto)


def duracion(entradas: Sequence[Mapping], m: Mapping, obs: Mapping = OBSERVACIONES) -> dict:
    """Duración de la corrida según el script, el log, la CLI y la página de Kaggle."""
    tiempos = [e["time"] for e in entradas if isinstance(e.get("time"), (int, float))]
    inicio = _utc(m["inicio_utc"])
    script_s = m.get("segundos_desde_arranque")
    log_s = round(max(tiempos), 1) if tiempos else None
    cota_cli_s = (_utc(obs["complete_observado_utc"]) - _utc(obs["inicio_cli_utc"])).total_seconds()
    pagina_s = obs["pagina_kaggle_min"] * 60
    return {
        "inicio_script_utc": inicio.isoformat(),
        "fin_script_utc": (inicio + datetime.timedelta(seconds=script_s)).isoformat(timespec="seconds") if script_s else None,
        "script_s": script_s,
        "log_ultima_marca_s": log_s,
        "cli": dict(obs),
        "cota_superior_cli_s": cota_cli_s,
        "pagina_kaggle_s": pagina_s,
        "horas": {
            "script": round(script_s / 3600, 3) if script_s else None,
            "log": round(log_s / 3600, 3) if log_s else None,
            "cota_superior_cli": round(cota_cli_s / 3600, 3),
            "pagina_kaggle": round(pagina_s / 3600, 3),
        },
    }


# --- Concentración de las respuestas ---


def _mas_frecuente(valores) -> tuple[str | None, int]:
    """El valor más frecuente y su conteo; en empate, el menor (determinista)."""
    conteo: dict[str, int] = {}
    for v in valores:
        conteo[v] = conteo.get(v, 0) + 1
    if not conteo:
        return None, 0
    return min(conteo.items(), key=lambda kv: (-kv[1], kv[0]))


def concentracion(predicciones: Mapping[str, str], referencias: Mapping[str, str], train: Sequence[Mapping]) -> dict:
    """¿El modelo responde casi siempre lo mismo? Solo conteos y códigos de formato válido, nunca respuestas crudas."""
    limpias = [limpiar(r) for r in predicciones.values()]
    codigo, filas = _mas_frecuente(c for c in limpias if RE_CODIGO.fullmatch(c))
    clase_train, filas_train = _mas_frecuente(r["codigo_act"] for r in train)
    return {
        "respuestas_distintas_tras_limpiar": len(set(limpias)),
        "codigo_valido_mas_predicho": codigo,
        "filas_con_ese_codigo": filas,
        "filas_de_la_referencia_con_ese_codigo": sum(1 for c in referencias.values() if c == codigo),
        "clase_mas_frecuente_de_humo_train": clase_train,
        "filas_de_humo_train_con_esa_clase": filas_train,
    }


# --- Flujo ---


def cargar_entradas(carpeta: str | Path, manifest: Mapping, raiz: Path = REPO_RAIZ) -> SimpleNamespace:
    """Todo lo que el paso lee: la salida bajada, las dos muestras y el catálogo (con hashes verificados)."""
    carpeta = Path(carpeta)
    inv = inventario(carpeta)
    mediciones = json.loads((carpeta / ARCHIVO_MEDICIONES).read_bytes())
    entradas_log = leer_log((carpeta / ARCHIVO_LOG).read_bytes())
    textos = {
        a["ruta"]: (carpeta / a["ruta"]).read_bytes().decode("utf-8")
        for a in inv
        if not a["ruta"].endswith(EXTENSIONES_BINARIAS) and a["ruta"] != ARCHIVO_LOG
    }
    textos[ARCHIVO_LOG] = texto_log(entradas_log)
    archivos = manifest[SECCION][ENTRADA]["archivos"]
    muestras = {a: leer_jsonl_verificado(Path(raiz) / archivos[a]["ruta"], archivos[a]["sha256"]) for a in (ARCHIVO_TRAIN, ARCHIVO_VAL)}
    cat = manifest["catalogos"][CATALOGO[manifest["scian_version"]]]
    ruta_cat = Path(raiz) / cat["ruta"]
    if sha256_bytes(ruta_cat.read_bytes()) != cat["sha256"]:
        raise ErrorHumo("El SHA-256 del catálogo no coincide con el manifest.")
    return SimpleNamespace(
        inventario=inv,
        mediciones=mediciones,
        log=entradas_log,
        textos=textos,
        predicciones=textos[ARCHIVO_PREDICCIONES].encode("utf-8"),
        train=muestras[ARCHIVO_TRAIN],
        val=muestras[ARCHIVO_VAL],
        codigos=frozenset(leer_catalogo(ruta_cat)),
        catalogo={"ruta": cat["ruta"], "sha256": cat["sha256"]},
    )


def evaluar_corrida(ent: SimpleNamespace, manifest: Mapping, contexto: Mapping) -> dict:
    """Verifica, revisa nombres, resume etapas, mide y arma el resultado; PARADA ante cualquier problema."""
    m = ent.mediciones
    descarga = verificar_descarga(ent.inventario, m, hashes_del_manifest(manifest))

    ref = referencias_de_nombres(ent.train + ent.val)
    privacidad = revisar_privacidad(ent.textos, m, ref, len(ent.train) + len(ent.val))
    if privacidad["posible_fuga"]:
        raise ErrorHumo(
            f"Posible fuga de nombres en {privacidad['archivos_con_posible_fuga'] or ['predicciones.jsonl']}; decide Samuel."
        )

    referencias = {r["id"]: r["codigo_act"] for r in ent.val}
    predicciones = leer_predicciones(ent.predicciones)
    metricas = evaluar(predicciones, referencias, ent.codigos)  # falla si los ids no son los de humo_val
    codigos_train = {r["codigo_act"] for r in ent.train}

    archivos = manifest[SECCION][ENTRADA]["archivos"]
    esperados = {a: archivos[a]["sha256"] for a in (ARCHIVO_TRAIN, ARCHIVO_VAL)}
    etapas = resumen_etapas(
        m, esperados=esperados, fuentes_actuales=contexto["fuentes_actuales_sha256"],
        kernel_igual_al_generado=contexto["kernel_igual_al_generado"], descarga=descarga,
        filas_train=len(ent.train), filas_val=len(ent.val),
    )
    rend = rendimiento(m)
    fijo = costo_fijo(m)
    estimacion = estimar_horas(rend, TAMANOS_TRAIN, fijo, {MODELO_06B["repo"]: m.get("inferencia", {}).get("s_por_fila")})
    tiempo = duracion(ent.log, m)

    resultado = armar_resultado(
        manifest=manifest, contexto=contexto, ent=ent, mediciones=m, descarga=descarga, privacidad=privacidad,
        metricas={
            **metricas,
            "definicion": DEFINICION,
            "implementacion": "src/metricas_clasificador.py",
            "catalogo": ent.catalogo,
            "scian_version": manifest["scian_version"],
            "codigos_existentes_distintos_predichos": len({limpiar(r) for r in predicciones.values()} & ent.codigos),
            "clases_distintas_en_la_referencia": len(set(referencias.values())),
            "filas_de_validacion_con_clase_ausente_de_humo_train": sum(1 for c in referencias.values() if c not in codigos_train),
            "concentracion": concentracion(predicciones, referencias, ent.train),
        },
        etapas=etapas, rend=rend, estimacion=estimacion, tiempo=tiempo,
    )
    candado_resultado(resultado, ref)
    return resultado


def armar_resultado(*, manifest, contexto, ent, mediciones, descarga, privacidad, metricas, etapas, rend, estimacion, tiempo) -> dict:
    """JSON de results/ con los campos de la regla 4; sin nombres, llaves ni respuestas crudas."""
    archivos = manifest[SECCION][ENTRADA]["archivos"]
    t = tiempo["horas"]
    return {
        "fecha": contexto["fecha"],
        "experimento": "fase2_paso6c_prueba_humo",
        "reportable": False,
        "nota": "Prueba de punta a punta desechable (pasos 6a a 6c); no se reporta en la charla.",
        "commit": contexto["commit"],
        "arbol_limpio": contexto["arbol_limpio"],
        "modelo": {
            "repo": MODELO_06B["repo"],
            "revision": MODELO_06B["revision"],
            "metodo": "QLoRA (NF4 de bitsandbytes) con Unsloth; enable_thinking=False",
            "adaptador_sha256": next((a["sha256"] for a in ent.inventario if a["ruta"] == "adaptador/adapter_model.safetensors"), None),
        },
        "configuracion": {
            "script": SCRIPT,
            "metricas": "src/metricas_clasificador.py",
            "kernel": RUTA_KERNEL,
            "kernel_generador": "src/construir_kernel.py",
            "commit_kernel": contexto["commit_kernel"],
            "kernel_kaggle": {"id": ID_KERNEL, "version": VERSION_KERNEL},
            "catalogo": ent.catalogo,
            "scian_version": manifest["scian_version"],
            "limpieza_de_respuestas": "strip(' \\r\\n'): solo espacio ASCII, \\r y \\n en las orillas (regla 5)",
            "del_kernel": mediciones.get("configuracion", {}),
        },
        "semilla": SEMILLA,
        "split": "humo_val.jsonl: 200 filas al azar (semilla 42) de la validación congelada; muestra de prueba desechable",
        "hash_split": {a: archivos[a]["sha256"] for a in (ARCHIVO_VAL, ARCHIVO_TRAIN)},
        "hash_split_origen": f"data/manifest.json, {SECCION}.{ENTRADA}",
        "tamano_muestra": {"evaluacion": len(ent.val), "entrenamiento": len(ent.train)},
        "horas_gpu": {
            "valor": t["pagina_kaggle"],
            "metodo": "Duración de la página del kernel en Kaggle (16 min, vista por Samuel), la mayor de las mediciones; la cuota que descontó Kaggle la confirma Samuel.",
            "script": t["script"],
            "log": t["log"],
            "cota_superior_cli": t["cota_superior_cli"],
            "cuota_kaggle": None,
        },
        "corrida_kaggle": {
            "estado_cli": OBSERVACIONES["estado_cli"],
            "estado_kernel": mediciones.get("estado"),
            "fallas": len(mediciones.get("fallas", [])),
            "duraciones_etapas_s": mediciones.get("duraciones_s"),
            "tiempo": tiempo,
            "archivos_bajados": ent.inventario,
            "descarga": descarga,
        },
        "etapas": etapas,
        "privacidad": privacidad_para_resultado(privacidad),
        "metricas_regla5": metricas,
        "rendimiento": rend,
        "estimacion_horas_gpu": {
            "tamanos_train": list(TAMANOS_TRAIN),
            "criterio_de_lote": "el de más ejemplos/s entre los que no se quedaron sin memoria",
            "por_modelo": estimacion,
            "advertencias": ADVERTENCIAS_ESTIMACION,
        },
    }


def serializar(resultado: Mapping) -> bytes:
    return (json.dumps(resultado, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


# --- Programa principal ---


def _git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=REPO_RAIZ, capture_output=True, text=True, check=True)
    return r.stdout.strip()


def contexto_git(manifest: Mapping) -> dict:
    from src import construir_kernel as ck

    generados = ck.generar(manifest)
    return {
        "fecha": datetime.date.today().isoformat(),
        "commit": _git("rev-parse", "HEAD"),
        "arbol_limpio": _git("status", "--porcelain", "--untracked-files=no") == "",
        "commit_kernel": _git("log", "-1", "--format=%H", "--", RUTA_KERNEL),
        "kernel_igual_al_generado": ck.diferencias(generados, ck.CARPETA) == [],
        "fuentes_actuales_sha256": {f: sha256_bytes(t.encode("utf-8")) for f, t in ck.leer_fuentes().items()},
    }


def imprimir(res: Mapping, privacidad_terminal: Mapping | None = None) -> None:
    d = res["corrida_kaggle"]
    print(f"Archivos bajados ({len(d['archivos_bajados'])}):")
    for a in d["archivos_bajados"]:
        print(f"  {a['ruta']}: {a['bytes']:,} bytes, SHA-256 {a['sha256']}")
    x = d["descarga"]
    print(f"Coinciden con Kaggle: {x['coinciden_con_kaggle']} de {x['con_sha256_de_kaggle']}; sin hash de Kaggle: {x['sin_sha256_de_kaggle']}; "
          f"con hash del manifest: {x['con_sha256_del_manifest']}")
    p = res["privacidad"]
    print(f"Privacidad: posible fuga = {p['posible_fuga']}; tapados por el kernel = {p['tapados_por_el_kernel']}; "
          f"nombre inventado es llave = {p['nombre_inventado_es_llave']}")
    for nombre, r in {**p["archivos"], ARCHIVO_PREDICCIONES: p["predicciones"]}.items():
        largos = (privacidad_terminal or {}).get(nombre, {}).get("largos_llaves_una_palabra")
        print(f"  {nombre}: crudos {r['nombres_crudos']}, llaves de varias palabras {r['llaves_varias_palabras']}, "
              f"de una palabra {r['llaves_una_palabra']} (líneas {r['lineas_llaves_una_palabra']}"
              + (f", largos {largos})" if largos else ")"))
    print("Etapas:")
    for nombre, e in res["etapas"]["etapas"].items():
        malas = [k for k, v in e.get("comprobaciones", {}).items() if not v]
        print(f"  {nombre}: {e['estado']}" + (f" (fallan: {malas})" if malas else ""))
    mt = res["metricas_regla5"]
    print(f"Regla 5 (n = {mt['total']}): {mt['conteos']} -> {mt['porcentajes']} %; formato válido {mt['pct_formato_valido']} %, "
          f"inexistentes {mt['pct_codigos_inexistentes']} %; aciertos {mt['aciertos']}, accuracy {mt['pct_accuracy']} %")
    c = mt["concentracion"]
    print(f"Concentración: {c['respuestas_distintas_tras_limpiar']} respuestas distintas; {c['codigo_valido_mas_predicho']} en {c['filas_con_ese_codigo']} filas "
          f"(en la referencia: {c['filas_de_la_referencia_con_ese_codigo']}); clase más frecuente de humo_train: "
          f"{c['clase_mas_frecuente_de_humo_train']} ({c['filas_de_humo_train_con_esa_clase']} filas)")
    print("Rendimiento:")
    for r in res["rendimiento"]:
        if r.get("sin_memoria"):
            print(f"  {r['modelo']} lote {r['lote']}: sin memoria")
            continue
        print(f"  {r['modelo']} lote {r['lote']}: {r['ejemplos_por_s']} ej/s, {r['fichas_reales_por_s']} fichas/s, "
              f"{r['s_por_paso_mediana']} s/paso (mediana), VRAM pico {r['vram_pico_mib']} MiB, {r['pasos_medidos']} pasos medidos")
    print("Horas de GPU estimadas (1 época):")
    for clave, e in res["estimacion_horas_gpu"]["por_modelo"].items():
        print(f"  {clave} (lote {e.get('lote')}, costo fijo {e.get('costo_fijo_s', {}).get('total')} s):")
        for fila in e.get("tabla", []):
            print(f"    {fila['filas_train']:>7,} filas: {fila['horas_entrenamiento']} h (contraste {fila['horas_contraste_mediana_s_por_paso']} h; "
                  f"con costo fijo {fila['horas_con_costo_fijo']} h)")
    print(f"Horas de GPU de la corrida: {res['horas_gpu']}")


def main(argv: Sequence[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from src.scian import leer_manifest
    from src.splits import esta_ignorado

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--escribir", action="store_true", help=f"escribe {ARCHIVO_RESULTADO}")
    args = parser.parse_args(argv)

    carpeta = REPO_RAIZ / CARPETA_RELATIVA
    manifest = leer_manifest()
    try:
        if not esta_ignorado(carpeta) or not all(esta_ignorado(carpeta / a["ruta"]) for a in inventario(carpeta)):
            raise ErrorHumo("La carpeta de la salida o alguno de sus archivos no está ignorado por git.")
        contexto = contexto_git(manifest)
        ent = cargar_entradas(carpeta, manifest)
        resultado = evaluar_corrida(ent, manifest, contexto)
    except (ErrorHumo, ErrorMetricas) as e:
        sys.exit(f"PARADA: {e}")
    except RuntimeError as e:  # AbortoHumo de leer_jsonl_verificado (hash de una muestra)
        sys.exit(f"PARADA: {e}")
    ref = referencias_de_nombres(ent.train + ent.val)
    terminal = revisar_privacidad(ent.textos, ent.mediciones, ref, len(ent.train) + len(ent.val))
    imprimir(resultado, {**terminal["archivos"], ARCHIVO_PREDICCIONES: terminal["predicciones"]})
    print(f"commit {contexto['commit'][:7]} | árbol limpio: {contexto['arbol_limpio']} | kernel de {contexto['commit_kernel'][:7]}")
    if args.escribir:
        ruta = REPO_RAIZ / ARCHIVO_RESULTADO
        ruta.write_bytes(serializar(resultado))
        print(f"Escrito {ARCHIVO_RESULTADO} ({ruta.stat().st_size:,} bytes, SHA-256 {sha256_archivo(ruta)}).")


if __name__ == "__main__":
    main()
