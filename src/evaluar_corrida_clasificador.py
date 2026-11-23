"""Evaluación local de una corrida del clasificador en Kaggle (Fase 2, paso 11).

Lee la salida que bajó `kaggle kernels output` a `kaggle-output/<corrida>/` (ignorada por git) y:

1. verifica que la carpeta y cada archivo estén ignorados por git, que cada archivo bajado tenga
   los bytes y el SHA-256 que calculó el kernel y que ninguno tenga un SHA-256 del manifest;
2. revisa nombres de negocio antes de mostrar nada del log o de las mediciones, con el método
   del paso 6c: (i) cada `nom_estab` de 3 caracteres o más como subcadena exacta y (ii) cada
   llave como palabras completas de cada línea normalizada. La referencia son `train_final.jsonl`
   y `val.jsonl` del paquete de Kaggle (lo que vio el kernel), con su SHA-256 verificado. La
   búsqueda está indexada (`buscar_nombres_indexada`) y da lo mismo que `buscar_nombres` del 6c;
3. resume cada etapa del kernel con `mediciones.json` y el log;
4. confirma que `predicciones.jsonl` tiene 1,000 filas con los ids de `data/splits/val.jsonl`
   (SHA-256 verificado) y la evalúa con `src/evaluador_clasificador.py`;
5. compara con los baselines (sus JSON de `results/`) y calcula la diferencia pareada por
   bootstrap contra TF-IDF + LR (corrida − TF-IDF), con los aciertos fila por fila del registro
   de `data/baselines/<commit>/`, después de comprobar que están alineados;
6. arma `results/<fecha>_fase2_<corrida>.json` (solo agregados: sin nombres, llaves, ids ni
   respuestas crudas) y la figura de las curvas de pérdida, hecha desde ese JSON.

Se detiene (PARADA) ante cualquier problema. Nunca abre `test.jsonl` ni `hard.jsonl`. Los
archivos bajados solo se leen como datos (JSON y texto); el adaptador solo se hashea. En la
terminal solo imprime conteos, códigos SCIAN, hashes y tiempos.

Uso:
  .venv/Scripts/python -m src.evaluar_corrida_clasificador --corrida 06b-inicial              # verifica e imprime
  .venv/Scripts/python -m src.evaluar_corrida_clasificador --corrida 06b-inicial --escribir   # además escribe
"""

from __future__ import annotations

import argparse
import bisect
import datetime
import hashlib
import json
import math
import platform
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

from src import construir_kernel as ck
from src.clasificador_kaggle import (
    ARCHIVO_MEDICIONES,
    ARCHIVO_TRAIN,
    ARCHIVO_VAL,
    ARCHIVOS_ENTRADA,
    CARPETA_ADAPTADOR,
    argumentos_entrenamiento,
    pasos_por_epoca,
    pasos_warmup,
)
from src.denue import REPO_RAIZ
from src.evaluador_clasificador import (
    DEFINICION_PAREADA,
    DEFINICIONES,
    MARCAS_REPORTADAS,
    ErrorEvaluador,
    _bloque,
    aciertos_por_nivel,
    cargar_codigos,
    cargar_referencias,
    diferencia_pareada_bootstrap,
    evaluar_validacion,
    intervalo_bootstrap,
    validar_referencias,
)
from src.evaluar_humo import (
    MINIMO_CRUDO,
    TOPE_TORCH,
    _baja,
    _etapa,
    _git,
    inventario,
    leer_log,
    referencias_de_nombres,
    resumen_perdida,
    texto_log,
    textos_de_json,
    version_menor,
)
from src.humo_kaggle import (
    ARCHIVO_PREDICCIONES,
    DIR_TRABAJO,
    PAQUETES,
    TAPADO,
    VERSIONES_A_REGISTRAR,
    AbortoHumo,
    leer_jsonl_verificado,
    sha256_archivo,
)
from src.metricas_clasificador import ErrorMetricas, categoria, leer_predicciones, limpiar
from src.metricas_clasificador import CODIGO_EXISTENTE
from src.normalizacion import normalizar_nombre
from src.prompt_clasificador import FIN_DE_TURNO, RE_CODIGO
from src.scian import CATALOGO

SCRIPT = "src/evaluar_corrida_clasificador.py"
SEMILLA = 42
SECCION = "kaggle_datasets"
ENTRADA_PAQUETE = "denue_scian_clasificador"
ENTRADA_VAL = "clasificador_validacion"
ENTRADA_TRAIN_FINAL = "clasificador_train_final"
EXTENSIONES_BINARIAS = (".safetensors",)
VENTANA_CURVA = 50  # pasos por punto de la curva de train guardada en results/
LINEAS_EN_TERMINAL = 30
# Precisión del método del 6c (paso 11): un nombre crudo de hasta tantos caracteres que solo aparece
# dentro de un token más largo (letras, dígitos, _ o -) no es posible fuga; se cuenta aparte. Uno de
# 5 caracteres o más dentro de cualquier token sigue siendo PARADA. Propuesta por Claude Code,
# acotada por el chat de Claude del proyecto y aprobada por Samuel el 2026-11-23.
LARGO_MAXIMO_DENTRO_DE_TOKEN = 4

RESULTADO_TFIDF = "results/2026-11-23_fase1_baseline-tfidf-lr_split-nombre.json"
RESULTADO_DICCIONARIO = "results/2026-11-23_fase1_baseline-diccionario_split-nombre.json"
CARPETA_REGISTROS_BASELINES = "data/baselines"

# Observaciones de cada corrida que no están en su salida.
CORRIDAS = {
    "06b-inicial": {
        "kernel": "06b-inicial",
        "carpeta": "kaggle-output/06b-inicial",
        "log": "denue-scian-06b-inicial.log",  # lo escribe la CLI de Kaggle 2.2.5 junto a la salida
        "resultado": "results/2026-11-23_fase2_06b-inicial.json",
        "figura": "docs/figuras/2026-11-23_fase2_curvas-perdida-06b-inicial.png",
        "observaciones": {
            "estado_cli": "COMPLETE",  # kaggle kernels status, 2026-11-23
            "version_kaggle": 1,  # página del kernel: "Version 1 of 1"
            "last_run_time_cli": "2026-11-22 22:19:31.317000",  # kaggle kernels list --mine (sin zona)
            "duracion_pagina_kaggle": "1h 12m 43s",  # página del kernel (Runtime), vista por Samuel
            "duracion_pagina_kaggle_s": 4363,
            "cuota_antes": "29 h 47 min de 30 h",  # bitácora, paso 7
            "cuota_despues": "28 h 35 min de 30 h",  # instrucción del paso 11
            "cuota_descontada_min": 72,
        },
    },
}


class ErrorCorrida(RuntimeError):
    """PARADA del paso; el mensaje nunca incluye un nom_estab, una llave, un id ni una respuesta."""


def sha256_bytes(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


# --- Archivos bajados ---


def verificar_descarga(inv: Sequence[Mapping], m: Mapping, hashes_manifest: set[str], archivo_log: str) -> dict:
    """Compara lo bajado con `archivos_salida` del kernel; PARADA si algo no coincide."""
    locales = {a["ruta"]: a for a in inv}
    kaggle = {a["ruta"]: a for a in m.get("archivos_salida", [])}
    sin_hash = {ARCHIVO_MEDICIONES, archivo_log}
    faltan = sorted(set(kaggle) - set(locales))
    distintos = sorted(
        r for r in set(kaggle) & set(locales)
        if (locales[r]["bytes"], locales[r]["sha256"]) != (kaggle[r]["bytes"], kaggle[r]["sha256"])
    )
    extra = sorted(set(locales) - set(kaggle) - sin_hash)
    sin_archivo = sorted(sin_hash - set(locales))
    con_hash_del_manifest = sorted(r for r, a in locales.items() if a["sha256"] in hashes_manifest)
    sha_pred = locales.get(ARCHIVO_PREDICCIONES, {}).get("sha256")
    predicciones_ok = sha_pred is not None and sha_pred == m.get("inferencia", {}).get("archivo", {}).get("sha256")
    res = {
        "archivos_bajados": len(locales),
        "con_sha256_de_kaggle": len(kaggle),
        "coinciden_con_kaggle": len(set(kaggle) & set(locales)) - len(distintos),
        "sin_sha256_de_kaggle": sorted(sin_hash & set(locales)),
        "faltan": faltan,
        "distintos": distintos,
        "extra": extra,
        "con_sha256_del_manifest": con_hash_del_manifest,
        "predicciones_coinciden_con_la_inferencia": predicciones_ok,
    }
    if faltan or distintos or extra or sin_archivo or con_hash_del_manifest or not predicciones_ok:
        raise ErrorCorrida(
            f"La descarga no coincide con lo que calculó Kaggle: faltan {len(faltan)}, distintos {len(distintos)}, "
            f"extra {len(extra)}, sin log o mediciones {len(sin_archivo)}, con hash del manifest "
            f"{len(con_hash_del_manifest)}, predicciones {'ok' if predicciones_ok else 'distintas'}."
        )
    return res


# --- Revisión de nombres (método del paso 6c, indexado) ---


def indice_de_nombres(ref: Mapping) -> dict:
    """Índice de `referencias_de_nombres`: nombres por sus primeros MINIMO_CRUDO caracteres."""
    por_prefijo: dict[str, list[str]] = {}
    for n in ref["nombres"]:
        por_prefijo.setdefault(n[:MINIMO_CRUDO], []).append(n)
    return {
        "nombres": ref["nombres"],
        "varias": ref["varias"],
        "una": ref["una"],
        "por_prefijo": por_prefijo,
        "max_palabras": max((k.count(" ") + 1 for k in ref["varias"]), default=1),
    }


def buscar_nombres_indexada(texto: str, idx: Mapping) -> tuple[dict, dict]:
    """Lo mismo que `evaluar_humo.buscar_nombres`, sin recorrer todas las llaves en cada línea.

    (i) cada nom_estab como subcadena exacta: se busca en cada posición del texto solo entre los
    nombres que empiezan con los mismos MINIMO_CRUDO caracteres; (ii) cada llave como palabras
    completas: se buscan en conjuntos los n-gramas de palabras contiguas de cada línea normalizada.
    Devuelve el resumen (igual al del 6c) y lo encontrado (solo para uso interno; nunca se imprime).
    """
    lineas = texto.split("\n")
    inicios = [0]
    for linea in lineas[:-1]:
        inicios.append(inicios[-1] + len(linea) + 1)
    crudos: set[str] = set()
    aislados: set[str] = set()  # con alguna aparición que no está pegada a otro carácter de token
    lineas_crudo: set[int] = set()
    por_prefijo = idx["por_prefijo"]
    for i in range(len(texto) - MINIMO_CRUDO + 1):
        candidatos = por_prefijo.get(texto[i:i + MINIMO_CRUDO])
        if not candidatos:
            continue
        for n in candidatos:
            if texto.startswith(n, i):
                crudos.add(n)
                fin = i + len(n)
                if not ((i > 0 and _de_token(texto[i - 1])) or (fin < len(texto) and _de_token(texto[fin]))):
                    aislados.add(n)
                if "\n" not in n:
                    lineas_crudo.add(bisect.bisect_right(inicios, i))
    varias: set[str] = set()
    una: set[str] = set()
    lineas_varias: set[int] = set()
    lineas_una: set[int] = set()
    maximo = idx["max_palabras"]
    for num, linea in enumerate(lineas, 1):
        normal = normalizar_nombre(linea)
        if not normal:
            continue
        palabras = normal.split(" ")
        for p in set(palabras) & idx["una"]:
            una.add(p)
            lineas_una.add(num)
        for a in range(len(palabras)):
            for b in range(a + 2, min(len(palabras), a + maximo) + 1):
                g = " ".join(palabras[a:b])
                if g in idx["varias"]:
                    varias.add(g)
                    lineas_varias.add(num)
    resumen = {
        "nombres_crudos": len(crudos),
        "lineas_nombres_crudos": sorted(lineas_crudo),
        "llaves_varias_palabras": len(varias),
        "lineas_llaves_varias_palabras": sorted(lineas_varias),
        "llaves_una_palabra": len(una),
        "lineas_llaves_una_palabra": sorted(lineas_una),
        "largos_llaves_una_palabra": sorted(len(k) for k in una),
    }
    return resumen, {"crudos": crudos, "crudos_aislados": aislados, "varias": varias, "una": una}


def _de_token(c: str) -> bool:
    """Carácter que forma parte de un token (identificador, paquete o palabra)."""
    return c.isalnum() or c in "_-"


def solo_digitos(texto: str) -> bool:
    """El texto, normalizado, es vacío o solo dígitos: como en tapar_llaves y verificar_privacidad."""
    k = normalizar_nombre(texto)
    return not k or k.replace(" ", "").isdigit()


def palabras_del_texto(texto: str) -> set[str]:
    return {p for linea in texto.split("\n") for p in normalizar_nombre(linea).split(" ") if p}


def clasificar(encontrados: Mapping[str, set[str]], texto_kernel: str, palabras_kernel: set[str]) -> dict:
    """Conteos de lo encontrado: solo dígitos (se aceptan) y presente en el texto del kernel (código)."""
    normal_kernel = " " + " ".join(normalizar_nombre(x) for x in texto_kernel.split("\n")) + " "
    crudos = encontrados["crudos"]
    crudos_texto = {n for n in crudos if not solo_digitos(n)}
    cortos_dentro = {
        n for n in crudos_texto if len(n) <= LARGO_MAXIMO_DENTRO_DE_TOKEN and n not in encontrados["crudos_aislados"]
    }
    varias_texto = {k for k in encontrados["varias"] if not solo_digitos(k)}
    una_texto = {k for k in encontrados["una"] if not solo_digitos(k)}
    return {
        "nombres_crudos_solo_digitos": len(crudos) - len(crudos_texto),
        "nombres_crudos_con_letras": len(crudos_texto),
        "nombres_crudos_con_letras_en_el_kernel": sum(1 for n in crudos_texto if n in texto_kernel),
        "nombres_crudos_cortos_solo_dentro_de_un_token": len(cortos_dentro),
        "nombres_crudos_posible_fuga": len(crudos_texto - cortos_dentro),
        "llaves_varias_palabras_solo_digitos": len(encontrados["varias"]) - len(varias_texto),
        "llaves_varias_palabras_con_letras": len(varias_texto),
        "llaves_varias_palabras_con_letras_en_el_kernel": sum(1 for k in varias_texto if f" {k} " in normal_kernel),
        "llaves_una_palabra_solo_digitos": len(encontrados["una"]) - len(una_texto),
        "llaves_una_palabra_con_letras": len(una_texto),
        "llaves_una_palabra_con_letras_en_el_kernel": len(una_texto & palabras_kernel),
    }


def posible_fuga(c: Mapping) -> bool:
    """Nombre crudo con letras fuera de la excepción de token, o llave de varias palabras con letras.

    Lo que normalizado es solo dígitos se acepta (como en tapar_llaves); también un nombre crudo de
    hasta LARGO_MAXIMO_DENTRO_DE_TOKEN caracteres que solo aparece dentro de un token más largo.
    """
    return bool(c["nombres_crudos_posible_fuga"] or c["llaves_varias_palabras_con_letras"])


def revisar_privacidad(textos: Mapping[str, str], mediciones: Mapping, idx: Mapping, filas: int, texto_kernel: str) -> dict:
    """Revisión de nombres de cada archivo de texto; predicciones.jsonl va aparte."""
    palabras_kernel = palabras_del_texto(texto_kernel)
    archivos: dict[str, dict] = {}
    entradas = {n: t for n, t in sorted(textos.items()) if n != ARCHIVO_PREDICCIONES}
    entradas[f"{ARCHIVO_MEDICIONES} (valores)"] = textos_de_json(mediciones)
    entradas[ARCHIVO_PREDICCIONES] = textos.get(ARCHIVO_PREDICCIONES, "")
    for nombre, texto in entradas.items():
        resumen, encontrados = buscar_nombres_indexada(texto, idx)
        archivos[nombre] = {**resumen, "clasificacion": clasificar(encontrados, texto_kernel, palabras_kernel)}
    fugas = sorted(n for n, r in archivos.items() if posible_fuga(r["clasificacion"]))
    return {
        "filas_de_referencia": filas,
        "nombres_buscados": len(idx["nombres"]),
        "llaves_varias_palabras_buscadas": len(idx["varias"]),
        "llaves_una_palabra_buscadas": len(idx["una"]),
        "tapados_por_el_kernel": len(mediciones.get("tapados", [])),
        "archivos": archivos,
        "archivos_con_posible_fuga": fugas,
        "posible_fuga": bool(fugas),
    }


def privacidad_para_resultado(p: Mapping) -> dict:
    """Solo conteos: sin largos de llaves y sin listas de líneas (pueden ser miles)."""
    archivos = {}
    for n, r in p["archivos"].items():
        archivos[n] = {
            "nombres_crudos": r["nombres_crudos"],
            "lineas_nombres_crudos": len(r["lineas_nombres_crudos"]),
            "llaves_varias_palabras": r["llaves_varias_palabras"],
            "lineas_llaves_varias_palabras": len(r["lineas_llaves_varias_palabras"]),
            "llaves_una_palabra": r["llaves_una_palabra"],
            "lineas_llaves_una_palabra": len(r["lineas_llaves_una_palabra"]),
            "clasificacion": dict(r["clasificacion"]),
        }
    return {
        **{k: v for k, v in p.items() if k != "archivos"},
        "archivos": archivos,
        "metodo": (
            f"Método del paso 6c con búsqueda indexada: (i) cada nom_estab de train_final.jsonl y val.jsonl del paquete "
            f"de Kaggle de {MINIMO_CRUDO} caracteres o más como subcadena exacta; (ii) cada llave (normalizar_nombre) como "
            "palabras completas de cada línea normalizada. Posible fuga: (i) o (ii) con llaves de dos o más palabras, sin "
            "contar lo que normalizado es solo dígitos (se acepta, como en tapar_llaves y verificar_privacidad) ni los "
            f"nombres crudos de {LARGO_MAXIMO_DENTRO_DE_TOKEN} caracteres o menos que solo aparecen dentro de un token más "
            "largo (letras, dígitos, _ o -); un nombre crudo de 5 caracteres o más dentro de cualquier token sigue siendo "
            "posible fuga (precisión propuesta por Claude Code, acotada por el chat y aprobada por Samuel el 2026-11-23). "
            "'en_el_kernel' cuenta lo encontrado que también está en el texto del kernel generado (código, no datos). "
            "Las coincidencias de una sola palabra solo se cuentan."
        ),
    }


# --- Etapas del kernel ---


def leer_freeze(texto: str) -> dict[str, str]:
    """Versiones de un `pip freeze` (nombres en minúsculas y con guion, como diferencias_freeze)."""
    res = {}
    for linea in texto.splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#"):
            continue
        nombre, sep, resto = linea.partition("==")
        if not sep:
            nombre, sep, resto = linea.partition(" @ ")
        res[nombre.strip().lower().replace("_", "-")] = resto.strip() if sep else linea
    return res


def _finito(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def curva_de_train_agregada(train: Sequence[Mapping], ventana: int = VENTANA_CURVA) -> list[dict]:
    """Media de la pérdida de train por ventanas de `ventana` pasos (1-50, 51-100, ...)."""
    grupos: dict[int, list[tuple[int, float]]] = {}
    for h in train:
        paso, loss = h.get("paso"), h.get("loss")
        if isinstance(paso, int) and paso >= 1 and _finito(loss):
            grupos.setdefault((paso - 1) // ventana, []).append((paso, loss))
    res = []
    for g in sorted(grupos):
        xs = grupos[g]
        res.append({
            "paso_inicial": min(p for p, _ in xs),
            "paso_final": max(p for p, _ in xs),
            "pasos": len(xs),
            "media": round(sum(v for _, v in xs) / len(xs), 4),
        })
    return res


RE_PERDIDA_VALIDACION = re.compile(r"pérdida de validación en el paso ([0-9]+): ([^\s]+)")


def validacion_del_log(texto: str) -> dict[int, float | None]:
    """Pérdida de validación por paso, de las líneas que imprime el callback del kernel (on_evaluate).

    Sirve para el paso 0: `trainer.train()` crea un estado nuevo y descarta el `log_history` de la
    evaluación previa, así que ese punto no queda en `curvas` de mediciones.json.
    """
    res: dict[int, float | None] = {}
    for paso, valor in RE_PERDIDA_VALIDACION.findall(texto):
        try:
            v: float | None = float(valor)
        except ValueError:
            v = None
        if int(paso) in res and res[int(paso)] != v:
            raise ErrorCorrida(f"El log trae dos pérdidas de validación distintas en el paso {paso}.")
        res[int(paso)] = v
    return res


def curva_de_validacion(val: Sequence[Mapping], del_log: Mapping[int, float | None]) -> list[dict]:
    """Puntos de validación de mediciones.json más los que solo están en el log (el paso 0), en orden de paso."""
    puntos = {v["paso"]: {"paso": v["paso"], "eval_loss": v.get("eval_loss"), "fuente": "mediciones.json"} for v in val}
    for paso, valor in del_log.items():
        if paso not in puntos:
            puntos[paso] = {"paso": paso, "eval_loss": valor, "fuente": "log (on_evaluate)"}
    return [puntos[p] for p in sorted(puntos)]


def argumentos_esperados(corrida: Mapping, pasos: int) -> dict:
    """Los argumentos del Trainer que arma el kernel para la corrida (mismo código, en fp16)."""
    return argumentos_entrenamiento(corrida, pasos, False, f"{DIR_TRABAJO.as_posix()}/trainer_{corrida['nombre']}")


def revisar_argumentos(args: Mapping, esperados_args: Mapping, tapados: Sequence[str], llaves: set[str]) -> dict:
    """Argumentos registrados contra los esperados; un valor tapado vale si el esperado, normalizado, es una llave."""
    prefijo = "$.entrenamiento.argumentos."
    tapados_args = {t[len(prefijo):] for t in tapados if t.startswith(prefijo)}
    distintos, tapados_explicados, tapados_sin_explicar = [], [], []
    for k, v in esperados_args.items():
        if k in tapados_args:
            ok = args.get(k) == TAPADO and isinstance(v, str) and normalizar_nombre(v) in llaves
            (tapados_explicados if ok else tapados_sin_explicar).append(k)
        elif args.get(k) != v:
            distintos.append(k)
    return {
        "mismas_claves": set(args) == set(esperados_args),
        "distintos": sorted(distintos),
        "tapados_explicados": sorted(tapados_explicados),
        "tapados_sin_explicar": sorted(tapados_sin_explicar),
        "tapados_fuera_de_los_argumentos": sorted(t for t in tapados if not t.startswith(prefijo)),
    }


def resumen_etapas(m: Mapping, *, espec: ck.EspecKernel, esperados: Mapping[str, str], fuentes_actuales: Mapping[str, str],
                   kernel_igual_al_generado: bool, filas_train: int, filas_val: int, adapter_config: Mapping,
                   archivos_bajados: set[str], llaves: set[str], texto_del_log: str) -> dict:
    """Terminó bien o falló, etapa por etapa, con su dato clave (sin nombres)."""
    corrida = dict(espec.corrida)
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
    entrada = sorted(a.get("ruta", "").rsplit("/", 1)[-1] for a in d.get("entrada", []))
    e["datos"] = _etapa(
        {
            "entrada_solo_los_dos_archivos": entrada == sorted(ARCHIVOS_ENTRADA),
            **{f"sha256_{a}": d.get(a, {}).get("sha256_verificado") is True and d.get(a, {}).get("sha256") == h for a, h in esperados.items()},
            "sha256_esperados_del_kernel": dict(m.get("datos_sha256_esperados", {})) == dict(esperados),
            "filas_train": d.get(ARCHIVO_TRAIN, {}).get("filas") == filas_train,
            "filas_val": d.get(ARCHIVO_VAL, {}).get("filas") == filas_val,
            "d_sin_nombres_con_marca_especial": comp.get("d_nombres_con_marca_especial") == 0,
            "fuentes_incrustadas_iguales_a_src": dict(m.get("fuentes_incrustadas_sha256", {})) == dict(fuentes_actuales),
            "kernel_escrito_igual_al_generado": kernel_igual_al_generado,
            "corrida_igual_a_la_especificacion": m.get("configuracion", {}).get("corrida") == corrida,
        },
        archivos_en_la_entrada=len(d.get("entrada", [])),
        filas={a: d.get(a, {}).get("filas") for a in esperados},
        clases={a: d.get(a, {}).get("clases") for a in esperados},
    )

    entorno = m.get("entorno", {})
    versiones = entorno.get("versiones", {})
    fijadas = dict(p.split("==") for p in PAQUETES)
    cambios = m.get("instalacion", {}).get("cambios_freeze", {})
    tocados = set(cambios.get("agregados", {})) | set(cambios.get("quitados", {})) | set(cambios.get("cambiados", {}))
    e["instalacion"] = _etapa(
        {
            "versiones_fijadas": all(versiones.get(k) == v for k, v in fijadas.items()),
            "torch_menor_que_2_15": version_menor(versiones.get("torch", ""), TOPE_TORCH),
            "pip_no_toco_torch": "torch" not in tocados,
        },
        python=entorno.get("python"),
        torch_cuda=entorno.get("torch_cuda"),
        cudnn=entorno.get("cudnn"),
        versiones={p: versiones.get(p) for p in VERSIONES_A_REGISTRAR},
        segundos_pip_install=m.get("instalacion", {}).get("segundos_pip_install"),
        pip_agrego=dict(cambios.get("agregados", {})),
        pip_cambio=dict(cambios.get("cambiados", {})),
        pip_quito=dict(cambios.get("quitados", {})),
    )

    mo = m.get("modelo", {})
    modelo = corrida["modelo"]
    cache = {(x.get("repo"), x.get("revision")) for x in m.get("descargas_hf", [])}
    cuant = mo.get("cuantizacion", {})
    e["modelo"] = _etapa(
        {
            "revision_fijada": mo.get("revision_descargada") == modelo["revision"] == mo.get("revision_fijada"),
            "repo_exacto": mo.get("name_or_path") == modelo["repo"] == mo.get("repo_pedido"),
            "lfs_verificado": bool(mo.get("lfs_verificado")),
            "nf4_de_4_bits": cuant.get("load_in_4bit") is True and cuant.get("bnb_4bit_quant_type") == "nf4",
            "cache_solo_la_revision_fijada": cache == {(modelo["repo"], modelo["revision"])},
            "largo_maximo_de_la_corrida": mo.get("max_seq_length_pedido") == corrida["max_seq_length"],
        },
        archivos_lfs_verificados=len(mo.get("lfs_verificado", {})),
        capas_linear4bit=mo.get("capas_linear4bit"),
        vram_tras_carga_mib=mo.get("vram_tras_carga_mib"),
        cache_hf=[{"repo": r, "revision": v} for r, v in sorted(cache)],
        lora=mo.get("lora"),
    )

    t, ej = m.get("tokens", {}), m.get("ejemplo_etiquetas", {})
    a = comp.get("a_prompt_generacion_igual_a_plantilla", {})
    pc = t.get("por_conjunto", {})
    fichas = t.get("fichas_etiquetadas_por_fila", {})
    e["plantilla_y_tokens"] = _etapa(
        {
            "a_prompt_igual_a_la_plantilla": a.get("distintos") == 0 and a.get("textos") == filas_train + filas_val + 1,
            "c_tokens_especiales_un_solo_token": comp.get("c_tokens_especiales_un_solo_token", {}).get("ok") is True,
            "sin_filas_de_mas_del_largo_maximo": all(pc.get(c, {}).get("mayores_a_max_seq") == 0 for c in ("train", "val")),
            "prompt_es_prefijo": t.get("prompt_no_es_prefijo") == 0,
            "etiquetas_solo_codigo_y_fin": t.get("etiquetas_distintas_de_codigo_y_fin") == 0,
            "fichas_etiquetadas_de_todas_las_filas": sum(fichas.values()) == filas_train + filas_val,
            "ejemplo_inventado_solo_codigo_y_fin": ej.get("fichas_con_etiqueta") == [*ej.get("codigo", ""), FIN_DE_TURNO],
        },
        b_solo_registro=comp.get("b_texto_entrenamiento_frente_a_plantilla", {}).get("conteo"),
        max_seq_length=t.get("max_seq_length"),
        por_conjunto=pc,
        fichas_etiquetadas_por_fila=fichas,
    )

    ent = m.get("entrenamiento", {})
    pasos = pasos_por_epoca(filas_train, corrida["lote"], corrida["gradient_accumulation_steps"]) * corrida["epocas"]
    curvas = ent.get("curvas", {})
    train = curvas.get("train", [])
    val = curvas.get("validacion", [])
    pasos_val = [v.get("paso") for v in val]
    esperados_val = sorted({0, *range(corrida["eval_cada_pasos"], pasos, corrida["eval_cada_pasos"]), pasos})
    del_log = validacion_del_log(texto_del_log)
    perd = resumen_perdida([{"loss": h.get("loss"), "learning_rate": h.get("learning_rate")} for h in train])
    orden = ent.get("orden_train", {})
    args = ent.get("argumentos", {})
    rev_args = revisar_argumentos(args, argumentos_esperados(corrida, pasos), m.get("tapados", []), llaves)
    curva_val = curva_de_validacion(val, del_log)
    eval_loss = [v["eval_loss"] for v in curva_val]
    e["entrenamiento"] = _etapa(
        {
            "fp16": ent.get("fp16") is True and args.get("fp16") is True and args.get("bf16") is False,
            "pasos_de_una_epoca": ent.get("pasos_esperados") == ent.get("max_steps_del_trainer") == ent.get("pasos_hechos") == pasos,
            "warmup": ent.get("pasos_warmup") == pasos_warmup(pasos, corrida["fraccion_warmup"]) == args.get("warmup_steps"),
            "argumentos_iguales_a_los_del_codigo": (
                rev_args["mismas_claves"] and not rev_args["distintos"] and not rev_args["tapados_sin_explicar"]
            ),
            "train_barajado": orden.get("barajado") is True and orden.get("permutacion_completa") is True,
            "train_completo_una_vez": orden.get("filas_vistas") == filas_train and orden.get("repetidas") == 0 and orden.get("faltantes") == 0,
            "revision_temprana_sin_orden": (ent.get("revision_temprana_del_orden") or {}).get("en_orden_creciente") is False,
            "colador_un_lote_por_paso": ent.get("lotes_registrados_por_el_colador", {}).get("train") == pasos,
            "perdida_de_train_en_cada_paso": [h.get("paso") for h in train] == list(range(1, pasos + 1)) and perd["no_finitos"] == 0,
            "perdida_de_train_baja": _baja(perd),
            "validacion_en_mediciones_salvo_el_paso_0": pasos_val == [p for p in esperados_val if p != 0],
            "validacion_del_log_en_los_pasos_previstos": sorted(del_log) == esperados_val,
            "log_y_mediciones_dan_la_misma_validacion": all(del_log.get(v.get("paso")) == v.get("eval_loss") for v in val),
            "validacion_finita_y_baja": all(_finito(x) for x in eval_loss) and bool(eval_loss) and eval_loss[-1] < eval_loss[0],
        },
        pasos=ent.get("pasos_hechos"),
        pasos_warmup=ent.get("pasos_warmup"),
        perdida_train=perd,
        perdida_validacion=[{**v, "eval_loss": round(v["eval_loss"], 4) if _finito(v["eval_loss"]) else v["eval_loss"]} for v in curva_val],
        argumentos=rev_args,
        orden_train={k: orden.get(k) for k in ("filas_vistas", "repetidas", "faltantes", "en_orden_de_id", "barajado", "posiciones_en_su_lugar", "sha256_orden")},
        segundos_train_total=ent.get("segundos_train_total"),
        segundos_evaluacion_paso_0=ent.get("segundos_evaluacion_paso_0"),
        evaluaciones_s=ent.get("evaluaciones_s"),
        estadisticas=ent.get("estadisticas"),
        vram_pico_mib=ent.get("vram_pico_mib"),
    )

    lora = corrida["lora"]
    e["adaptador"] = _etapa(
        {
            "archivos": {f"{CARPETA_ADAPTADOR}/adapter_model.safetensors", f"{CARPETA_ADAPTADOR}/adapter_config.json"} <= archivos_bajados,
            "rank_y_alpha": adapter_config.get("r") == lora["r"] and adapter_config.get("lora_alpha") == lora["lora_alpha"],
            "modulos_objetivo": sorted(adapter_config.get("target_modules") or []) == sorted(lora["target_modules"]),
            "base": adapter_config.get("base_model_name_or_path") == modelo["repo"],
        },
        r=adapter_config.get("r"),
        lora_alpha=adapter_config.get("lora_alpha"),
        lora_dropout=adapter_config.get("lora_dropout"),
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
        max_new_tokens=gen.get("max_new_tokens"),
        segundos=inf.get("segundos"),
        s_por_fila=inf.get("s_por_fila"),
        vram_pico_mib=inf.get("vram_pico_mib"),
    )

    fallas = m.get("fallas", [])
    e["estado_final"] = _etapa(
        {
            "completo": m.get("estado") == "completo",
            "sin_fallas": not fallas,
            "tapados_solo_argumentos_explicados": not rev_args["tapados_fuera_de_los_argumentos"] and not rev_args["tapados_sin_explicar"],
        },
        estado_del_kernel=m.get("estado"),
        etapas_con_falla=[f.get("etapa") for f in fallas],
        tapados=len(m.get("tapados", [])),
    )
    return {"todas_ok": all(x["estado"] == "ok" for x in e.values()), "etapas": e}


def duracion(entradas_log: Sequence[Mapping], m: Mapping, obs: Mapping) -> dict:
    tiempos = [x["time"] for x in entradas_log if isinstance(x.get("time"), (int, float))]
    script_s = m.get("segundos_desde_arranque")
    return {
        "inicio_script_utc": m.get("inicio_utc"),
        "script_s": script_s,
        "log_ultima_marca_s": round(max(tiempos), 1) if tiempos else None,
        "pagina_kaggle_s": obs["duracion_pagina_kaggle_s"],
        "duraciones_etapas_s": m.get("duraciones_s"),
        "horas": {
            "script": round(script_s / 3600, 3) if script_s else None,
            "pagina_kaggle": round(obs["duracion_pagina_kaggle_s"] / 3600, 3),
        },
    }


# --- Métricas ---


def alinear(predicciones: Mapping[str, str], val_splits: Sequence[Mapping], val_paquete: Sequence[Mapping]) -> list[Mapping]:
    """Filas de referencia en orden de int(id), tras confirmar que las predicciones son exactamente esa validación."""
    ids_pred = list(predicciones)
    ids_splits = [f["id"] for f in val_splits]
    paquete = {r["id"]: r["codigo_act"] for r in val_paquete}
    if len(ids_pred) != len(set(ids_pred)):
        raise ErrorCorrida("Las predicciones tienen ids repetidos.")
    if len(ids_pred) != len(ids_splits):
        raise ErrorCorrida(f"Hay {len(ids_pred)} predicciones y {len(ids_splits)} filas de validación.")
    if set(ids_pred) != set(ids_splits) or set(ids_splits) != set(paquete):
        raise ErrorCorrida("Los ids de las predicciones, de data/splits/val.jsonl y del val.jsonl del paquete no son los mismos.")
    if any(paquete[f["id"]] != f["codigo_act"] for f in val_splits):
        raise ErrorCorrida("El val.jsonl del paquete y data/splits/val.jsonl no tienen las mismas etiquetas.")
    filas = validar_referencias(val_splits)
    if [int(f["id"]) for f in filas] != sorted(int(i) for i in ids_pred):
        raise ErrorCorrida("Las filas no quedaron en orden de int(id).")
    return filas


def aciertos_alineados(predicciones: Mapping[str, str], filas: Sequence[Mapping], codigos) -> dict[str, list[bool]]:
    """Aciertos por nivel en el orden de `filas`, con la clase comprobada fila por fila."""
    niveles = aciertos_por_nivel(predicciones, filas, codigos)
    for i, f in enumerate(filas):
        r = predicciones[f["id"]]
        esperado = categoria(r, codigos) == CODIGO_EXISTENTE and limpiar(r) == f["codigo_act"]
        if niveles["clase"][i] != esperado:
            raise ErrorCorrida(f"El acierto de la posición {i} no coincide con su fila.")
    return niveles


def comprobar_registro_tfidf(reg: Mapping, res_json: Mapping, filas: Sequence[Mapping]) -> dict:
    """¿Los aciertos fila por fila del registro de TF-IDF están en el orden de `filas`?

    El registro solo trae aciertos (sin códigos predichos), así que no se pueden recalcular contra
    las etiquetas. Se comprueban los conteos por nivel y por subgrupo y se reproducen los IC por
    bootstrap, que con semilla fija dependen de la posición de cada acierto.
    """
    m = res_json["metricas"]
    clase, sector = reg["aciertos"]["clase"], reg["aciertos"]["sector"]
    c = {
        "trae_codigos_predichos": any(k in reg for k in ("respuestas", "predicciones")) or any(k in reg.get("aciertos", {}) for k in ("respuestas", "predicciones")),
        "mismo_commit": reg.get("commit") == res_json["commit"],
        "clave_del_c_elegido": reg.get("clave") == f"tfidf_nombre_C{res_json['detalle']['C']:g}",
        "mismas_metricas": reg.get("metricas") == m,
        "largo": len(clase) == len(sector) == len(filas),
        "solo_bool": all(isinstance(x, bool) for x in [*clase, *sector]),
        "conteo_clase": sum(clase) == m["regla5"]["aciertos"] == m["por_nivel"]["clase"]["aciertos"],
        "conteo_sector": sum(sector) == m["por_nivel"]["sector"]["aciertos"],
        "acierto_de_clase_implica_sector": all(s or not k for k, s in zip(clase, sector)),
        "ic_clase_reproducido": intervalo_bootstrap(clase) == m["ic95_accuracy"] == m["por_nivel"]["clase"]["ic95"],
        "ic_sector_reproducido": intervalo_bootstrap(sector) == m["por_nivel"]["sector"]["ic95"],
    }
    for marca in MARCAS_REPORTADAS:
        for lado, valor in (("con", True), ("sin", False)):
            sub = [a for a, f in zip(clase, filas) if f[marca] is valor]
            c[f"subgrupo_{marca}_{lado}_reproducido"] = _bloque(sub) == m["subgrupos"][marca][lado]
    if c["trae_codigos_predichos"]:
        raise ErrorCorrida("El registro de TF-IDF trae códigos predichos: hay que recalcular sus aciertos contra las etiquetas.")
    malas = sorted(k for k, v in c.items() if k != "trae_codigos_predichos" and not v)
    if malas:
        raise ErrorCorrida(f"Los aciertos de TF-IDF no pasan la comprobación de alineación: {malas}.")
    return {
        "comprobaciones": c,
        "fuerza": (
            "El registro solo trae aciertos de clase y de sector, sin códigos predichos ni ids, así que no se pudieron "
            "recalcular contra las etiquetas. Se comprobó que es del mismo commit y con las mismas métricas que su JSON de "
            "results/, que los conteos de clase, de sector y de los cuatro subgrupos (con las marcas de validación en orden "
            "de int(id)) coinciden, y que los IC por bootstrap de clase, de sector y de los cuatro subgrupos se reproducen "
            "exactamente. Como el bootstrap con semilla fija depende de la posición de cada acierto, eso confirma el orden "
            "con alta probabilidad, pero no es una comprobación fila por fila con ids."
        ),
    }


def comparacion(aciertos: Mapping[str, list[bool]], metricas: Mapping, tfidf: Mapping, dicc: Mapping, reg_tfidf: Mapping,
                comprobacion: Mapping, ruta_registro: str, sha_registro: str) -> dict:
    def fila(nombre: str, mt: Mapping) -> dict:
        return {
            "modelo": nombre,
            "accuracy": mt["regla5"]["accuracy"],
            "ic95": mt["ic95_accuracy"],
            "accuracy_sector": mt["por_nivel"]["sector"]["accuracy"],
            "ic95_sector": mt["por_nivel"]["sector"]["ic95"],
            "macro_f1_sector": mt["macro_f1_sector"]["valor"],
            "pct_formato_valido": mt["regla5"]["pct_formato_valido"],
            "pct_codigos_inexistentes": mt["regla5"]["pct_codigos_inexistentes"],
            "clases_existentes_distintas_predichas": mt["concentracion"]["clases_existentes_distintas_predichas"],
            "pct_filas_con_el_codigo_mas_predicho": mt["concentracion"]["pct_filas_con_ese_codigo"],
        }

    return {
        "tabla": [
            fila("corrida", metricas),
            fila(f"baseline_tfidf_lr (C = {tfidf['detalle']['C']:g})", tfidf["metricas"]),
            fila("baseline_diccionario", dicc["metricas"]),
        ],
        "resultados_baselines": {"tfidf_lr": RESULTADO_TFIDF, "diccionario": RESULTADO_DICCIONARIO},
        "diferencia_pareada_contra_tfidf": {
            "signo": "corrida − TF-IDF + LR (positivo: la corrida acierta más)",
            "clase": diferencia_pareada_bootstrap(reg_tfidf["aciertos"]["clase"], aciertos["clase"]),
            "sector": diferencia_pareada_bootstrap(reg_tfidf["aciertos"]["sector"], aciertos["sector"]),
            "definicion": DEFINICION_PAREADA,
            "registro_tfidf": {"ruta": ruta_registro, "sha256": sha_registro, "ignorado_por_git": True},
            "alineacion_corrida": {
                "filas": len(aciertos["clase"]),
                "orden": "int(id) de data/splits/val.jsonl",
                "comprobada_fila_por_fila": True,
                "metodo": (
                    "Los ids de las predicciones son exactamente los de data/splits/val.jsonl y los del val.jsonl del paquete "
                    "(mismas etiquetas); en cada posición, el acierto de clase es el de la respuesta de esa fila contra su "
                    "codigo_act, y los aciertos reproducen el conteo y el IC del evaluador."
                ),
            },
            "alineacion_tfidf": dict(comprobacion),
        },
    }


# --- Flujo ---


def cargar_entradas(corrida: str, manifest: Mapping, raiz: Path = REPO_RAIZ) -> SimpleNamespace:
    """Todo lo que el paso lee, con los hashes verificados. Nunca abre test.jsonl ni hard.jsonl."""
    conf = CORRIDAS[corrida]
    espec = ck.KERNELS[conf["kernel"]]
    carpeta = Path(raiz) / conf["carpeta"]
    inv = inventario(carpeta)
    mediciones = json.loads((carpeta / ARCHIVO_MEDICIONES).read_bytes())
    entradas_log = leer_log((carpeta / conf["log"]).read_bytes())
    textos = {
        a["ruta"]: (carpeta / a["ruta"]).read_bytes().decode("utf-8")
        for a in inv
        if not a["ruta"].endswith(EXTENSIONES_BINARIAS) and a["ruta"] != conf["log"]
    }
    textos[conf["log"]] = texto_log(entradas_log)
    adapter_config = json.loads(textos.get(f"{CARPETA_ADAPTADOR}/adapter_config.json", "{}"))
    archivos = manifest[SECCION][ENTRADA_PAQUETE]["archivos"]
    paquete = {
        a: leer_jsonl_verificado(Path(raiz) / archivos[a]["ruta"], archivos[a]["sha256"]) for a in (ARCHIVO_TRAIN, ARCHIVO_VAL)
    }
    tfidf = json.loads((Path(raiz) / RESULTADO_TFIDF).read_bytes())
    dicc = json.loads((Path(raiz) / RESULTADO_DICCIONARIO).read_bytes())
    ruta_reg = f"{CARPETA_REGISTROS_BASELINES}/{tfidf['commit'][:12]}/tfidf_nombre_C{tfidf['detalle']['C']:g}.json"
    datos_reg = (Path(raiz) / ruta_reg).read_bytes()
    return SimpleNamespace(
        corrida=corrida,
        conf=conf,
        espec=espec,
        carpeta=carpeta,
        inventario=inv,
        mediciones=mediciones,
        log=entradas_log,
        textos=textos,
        adapter_config=adapter_config,
        predicciones=textos[ARCHIVO_PREDICCIONES].encode("utf-8"),
        train=paquete[ARCHIVO_TRAIN],
        val_paquete=paquete[ARCHIVO_VAL],
        val_splits=cargar_referencias(manifest, ENTRADA_VAL, raiz),
        codigos=cargar_codigos(manifest, raiz),
        tfidf=tfidf,
        diccionario=dicc,
        registro_tfidf=json.loads(datos_reg),
        ruta_registro_tfidf=ruta_reg,
        sha_registro_tfidf=sha256_bytes(datos_reg),
    )


def evaluar_corrida(ent: SimpleNamespace, manifest: Mapping, contexto: Mapping) -> tuple[dict, dict]:
    """Verifica, revisa nombres, resume etapas, mide y arma el resultado; PARADA ante cualquier problema.

    Devuelve el resultado y la revisión de privacidad completa (con líneas, solo para la terminal).
    """
    m = ent.mediciones
    descarga = verificar_descarga(ent.inventario, m, ck.hashes_del_manifest(manifest), ent.conf["log"])

    filas_ref = ent.train + ent.val_paquete
    ref = referencias_de_nombres(filas_ref)
    idx = indice_de_nombres(ref)
    privacidad = revisar_privacidad(ent.textos, m, idx, len(filas_ref), contexto["texto_kernel"])
    if privacidad["posible_fuga"]:
        raise ErrorCorrida(f"Posible fuga de nombres en {privacidad['archivos_con_posible_fuga']}; decide Samuel.")

    archivos = manifest[SECCION][ENTRADA_PAQUETE]["archivos"]
    esperados = {a: archivos[a]["sha256"] for a in ARCHIVOS_ENTRADA}
    etapas = resumen_etapas(
        m, espec=ent.espec, esperados=esperados, fuentes_actuales=contexto["fuentes_actuales_sha256"],
        kernel_igual_al_generado=contexto["kernel_igual_al_generado"], filas_train=len(ent.train),
        filas_val=len(ent.val_paquete), adapter_config=ent.adapter_config,
        archivos_bajados={a["ruta"] for a in ent.inventario}, llaves=set(ref["varias"] | ref["una"]),
        texto_del_log=ent.textos[ent.conf["log"]],
    )

    predicciones = leer_predicciones(ent.predicciones)
    filas = alinear(predicciones, ent.val_splits, ent.val_paquete)
    metricas = evaluar_validacion(predicciones, filas, ent.codigos)
    aciertos = aciertos_alineados(predicciones, filas, ent.codigos)
    if intervalo_bootstrap(aciertos["clase"]) != metricas["ic95_accuracy"] or sum(aciertos["clase"]) != metricas["regla5"]["aciertos"]:
        raise ErrorCorrida("Los aciertos alineados no reproducen las métricas del evaluador.")
    codigo = metricas["concentracion"]["codigo_valido_mas_predicho"]
    concentracion_extra = {
        "filas_de_validacion_con_ese_codigo": sum(1 for f in filas if f["codigo_act"] == codigo),
        "clase_mas_frecuente_de_validacion": Counter(f["codigo_act"] for f in filas).most_common(1)[0][0],
        "filas_de_validacion_con_la_clase_mas_frecuente": Counter(f["codigo_act"] for f in filas).most_common(1)[0][1],
        "respuestas_distintas_tras_limpiar": len({limpiar(r) for r in predicciones.values()}),
        "codigos_validos_distintos_predichos": len({c for c in map(limpiar, predicciones.values()) if RE_CODIGO.fullmatch(c)}),
    }
    comprobacion = comprobar_registro_tfidf(ent.registro_tfidf, ent.tfidf, filas)
    comp = comparacion(aciertos, metricas, ent.tfidf, ent.diccionario, ent.registro_tfidf, comprobacion,
                       ent.ruta_registro_tfidf, ent.sha_registro_tfidf)

    obs = ent.conf["observaciones"]
    tiempo = duracion(ent.log, m, obs)
    curvas = m.get("entrenamiento", {}).get("curvas", {})
    resultado = armar_resultado(
        manifest=manifest, contexto=contexto, ent=ent, descarga=descarga, privacidad=privacidad, etapas=etapas,
        metricas={k: v for k, v in metricas.items() if k != "definiciones"}, concentracion_extra=concentracion_extra,
        comparacion=comp, tiempo=tiempo,
        curvas={
            "validacion": etapas["etapas"]["entrenamiento"]["perdida_validacion"],
            "train_por_ventana": curva_de_train_agregada(curvas.get("train", [])),
            "ventana_pasos": VENTANA_CURVA,
            "definicion": (
                "Pérdida = entropía cruzada media por ficha etiquetada (solo el código de 6 dígitos y <|im_end|>). "
                "Validación: las 1,000 filas, en el paso 0 (antes de entrenar), cada 500 pasos y en el último; el punto "
                "del paso 0 sale del log, porque trainer.train() descarta el log_history de la evaluación previa. Train: "
                f"media de la pérdida registrada en cada paso, por ventanas de {VENTANA_CURVA} pasos."
            ),
        },
    )
    candado_resultado(resultado, ref)
    return resultado, privacidad


def armar_resultado(*, manifest, contexto, ent, descarga, privacidad, etapas, metricas, concentracion_extra, comparacion,
                    tiempo, curvas) -> dict:
    """JSON de results/ con los campos de la regla 4; sin nombres, llaves, ids ni respuestas crudas."""
    m = ent.mediciones
    obs = ent.conf["observaciones"]
    corrida = dict(ent.espec.corrida)
    archivos = manifest[SECCION][ENTRADA_PAQUETE]["archivos"]
    sp = manifest["splits"]
    return {
        "fecha": contexto["fecha"],
        "experimento": f"fase2_paso11_{ent.corrida}",
        "commit": contexto["commit"],
        "arbol_limpio": contexto["arbol_limpio"],
        "modelo": {
            "nombre": f"qwen3_0.6b_qlora_{ent.corrida}",
            "repo": corrida["modelo"]["repo"],
            "revision": corrida["modelo"]["revision"],
            "metodo": "QLoRA (NF4 de bitsandbytes) con Unsloth; enable_thinking=False; responde solo el código de 6 dígitos",
            "adaptador_sha256": next((a["sha256"] for a in ent.inventario if a["ruta"] == f"{CARPETA_ADAPTADOR}/adapter_model.safetensors"), None),
        },
        "configuracion": {
            "corrida": corrida,
            "del_kernel": m.get("configuracion", {}),
            "kernel": f"{ent.espec.carpeta_relativa}/{ent.espec.archivo_kernel}",
            "kernel_generador": "src/construir_kernel.py",
            "commit_kernel": contexto["commit_kernel"],
            "kernel_kaggle": {"id": ent.espec.id_kernel, "version": obs["version_kaggle"]},
            "script": SCRIPT,
            "evaluador": "src/evaluador_clasificador.py (regla 5: src/metricas_clasificador.py)",
            "catalogo": {k: manifest["catalogos"][CATALOGO[manifest["scian_version"]]][k] for k in ("ruta", "sha256")},
            "scian_version": manifest["scian_version"],
            "python_local": platform.python_version(),
        },
        "semilla": SEMILLA,
        "split": "validación del split por nombre (congelada); train: train final con tope de 1,000 filas por clase",
        "hash_split": {
            "validacion": sp[ENTRADA_VAL]["sha256"],
            "train": sp[ENTRADA_TRAIN_FINAL]["sha256"],
            "paquete_kaggle": {a: archivos[a]["sha256"] for a in ARCHIVOS_ENTRADA},
        },
        "hash_split_origen": f"data/manifest.json: splits.{ENTRADA_VAL}, splits.{ENTRADA_TRAIN_FINAL} y {SECCION}.{ENTRADA_PAQUETE}",
        "tamano_muestra": {"train": len(ent.train), "validacion": len(ent.val_paquete)},
        "horas_gpu": {
            "valor": tiempo["horas"]["pagina_kaggle"],
            "metodo": f"Duración de la página del kernel en Kaggle ({obs['duracion_pagina_kaggle']}, GPU T4 x2 con una sola T4 en uso).",
            "script": tiempo["horas"]["script"],
            "cuota_kaggle": {"antes": obs["cuota_antes"], "despues": obs["cuota_despues"], "descontado_min": obs["cuota_descontada_min"]},
        },
        "corrida_kaggle": {
            "estado_cli": obs["estado_cli"],
            "last_run_time_cli": obs["last_run_time_cli"],
            "estado_kernel": m.get("estado"),
            "fallas": len(m.get("fallas", [])),
            "tiempo": tiempo,
            "archivos_bajados": [dict(a) for a in ent.inventario],
            "descarga": descarga,
        },
        "privacidad": privacidad_para_resultado(privacidad),
        "etapas": etapas,
        "metricas": metricas,
        "concentracion_extra": concentracion_extra,
        "definiciones": DEFINICIONES,
        "comparacion": comparacion,
        "curvas": curvas,
        "para_el_barrido": {
            "sha256_orden_train": m.get("entrenamiento", {}).get("orden_train", {}).get("sha256_orden"),
            "versiones_entorno": {p: m.get("entorno", {}).get("versiones", {}).get(p) for p in VERSIONES_A_REGISTRAR},
            "pip_freeze_despues": {
                p: leer_freeze(ent.textos.get("pip_freeze_despues.txt", "")).get(p.lower().replace("_", "-"))
                for p in VERSIONES_A_REGISTRAR
            },
            "pip_cambio_al_instalar": m.get("instalacion", {}).get("cambios_freeze", {}),
        },
    }


def candado_resultado(resultado: Mapping, ref: Mapping) -> None:
    """Ningún texto del resultado, normalizado, es una llave de train o validación (salvo solo dígitos)."""
    from src.exploracion_clasificador import ErrorPrivacidad, verificar_privacidad

    try:
        verificar_privacidad(resultado, set(ref["varias"] | ref["una"]), set())
    except ErrorPrivacidad as e:
        raise ErrorCorrida(f"El resultado no pasa el candado de privacidad: {e}") from e


def serializar(resultado: Mapping) -> bytes:
    return (json.dumps(resultado, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


# --- Figura ---


def figura_curvas(resultado: Mapping, ruta: Path) -> None:
    """Curvas de pérdida de train (medias por ventana) y de validación, desde el JSON de results/."""
    from src.exploracion_clasificador import (
        EJE, REJILLA, SERIE_1, SERIE_2, TINTA_2, _figura, _guardar, _pyplot, _titulos,
    )

    c = resultado["curvas"]
    corrida = resultado["configuracion"]["corrida"]
    plt = _pyplot()
    fig = _figura(plt)
    ax = fig.add_axes([0.08, 0.14, 0.88, 0.66])
    xs = [(v["paso_inicial"] + v["paso_final"]) / 2 for v in c["train_por_ventana"]]
    ys = [v["media"] for v in c["train_por_ventana"]]
    ax.plot(xs, ys, color=SERIE_1, linewidth=3, label=f"Train (media de cada {c['ventana_pasos']} pasos)")
    vx = [v["paso"] for v in c["validacion"]]
    vy = [v["eval_loss"] for v in c["validacion"]]
    ax.plot(vx, vy, color=SERIE_2, linewidth=3, marker="o", markersize=11, label="Validación (1,000 filas)")
    ax.set_xlabel("Paso de entrenamiento (lote de 32)")
    ax.set_ylabel("Pérdida (entropía cruzada por ficha)")
    ax.set_xlim(0, max(vx + [x for x in xs]) * 1.01)
    ax.set_ylim(0, None)
    ax.grid(axis="y", color=REJILLA, linewidth=1.2)
    ax.tick_params(colors=TINTA_2)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(EJE)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(loc="upper right")
    lora = corrida["lora"]
    pasos = resultado["etapas"]["etapas"]["entrenamiento"]["pasos"]
    _titulos(
        fig,
        f"Curvas de pérdida del Qwen3-0.6B ({resultado['modelo']['nombre'].rsplit('_', 1)[-1]})",
        f"QLoRA rank {lora['r']}, alpha {lora['lora_alpha']}, tasa {corrida['learning_rate']:g}, lote {corrida['lote']}, "
        f"{corrida['epocas']} época ({pasos:,} pasos); validación en el paso 0, cada {corrida['eval_cada_pasos']} pasos y al final.",
    )
    _guardar(fig, ruta)
    plt.close(fig)


# --- Programa principal ---


def contexto_git(manifest: Mapping, corrida: str) -> dict:
    espec = ck.KERNELS[CORRIDAS[corrida]["kernel"]]
    generados = ck.generar(manifest, espec=espec)
    return {
        "fecha": datetime.date.today().isoformat(),
        "commit": _git("rev-parse", "HEAD"),
        "arbol_limpio": _git("status", "--porcelain", "--untracked-files=no") == "",
        "commit_kernel": _git("log", "-1", "--format=%H", "--", f"{espec.carpeta_relativa}/{espec.archivo_kernel}"),
        "kernel_igual_al_generado": ck.diferencias(generados, espec.carpeta) == [],
        "fuentes_actuales_sha256": {f: sha256_bytes(t.encode("utf-8")) for f, t in ck.leer_fuentes(espec=espec).items()},
        "texto_kernel": generados[espec.archivo_kernel].decode("utf-8"),
    }


def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f} %"


def imprimir(res: Mapping, privacidad: Mapping) -> None:
    d = res["corrida_kaggle"]
    print(f"Archivos bajados ({len(d['archivos_bajados'])}):")
    for a in d["archivos_bajados"]:
        print(f"  {a['ruta']}: {a['bytes']:,} bytes, SHA-256 {a['sha256']}")
    x = d["descarga"]
    print(f"Coinciden con Kaggle: {x['coinciden_con_kaggle']} de {x['con_sha256_de_kaggle']}; sin hash de Kaggle: "
          f"{x['sin_sha256_de_kaggle']}; con hash del manifest: {x['con_sha256_del_manifest']}")
    p = privacidad
    print(f"Privacidad: posible fuga = {p['posible_fuga']}; tapados por el kernel = {p['tapados_por_el_kernel']}; "
          f"{p['nombres_buscados']:,} nombres, {p['llaves_varias_palabras_buscadas']:,} llaves de varias palabras y "
          f"{p['llaves_una_palabra_buscadas']:,} de una")
    for nombre, r in p["archivos"].items():
        c = r["clasificacion"]
        lu = r["lineas_llaves_una_palabra"]
        print(f"  {nombre}: crudos {r['nombres_crudos']} (solo dígitos {c['nombres_crudos_solo_digitos']}, cortos dentro de un "
              f"token {c['nombres_crudos_cortos_solo_dentro_de_un_token']}, posible fuga {c['nombres_crudos_posible_fuga']}; "
              f"líneas {r['lineas_nombres_crudos'][:LINEAS_EN_TERMINAL]}), "
              f"varias palabras {r['llaves_varias_palabras']} (solo dígitos {c['llaves_varias_palabras_solo_digitos']}), "
              f"una palabra {r['llaves_una_palabra']} (solo dígitos {c['llaves_una_palabra_solo_digitos']}; con letras "
              f"{c['llaves_una_palabra_con_letras']}, de ellas en el kernel {c['llaves_una_palabra_con_letras_en_el_kernel']}) "
              f"en {len(lu)} líneas {lu[:LINEAS_EN_TERMINAL]}{' ...' if len(lu) > LINEAS_EN_TERMINAL else ''}")
    print("Etapas:")
    for nombre, e in res["etapas"]["etapas"].items():
        malas = [k for k, v in e.get("comprobaciones", {}).items() if not v]
        print(f"  {nombre}: {e['estado']}" + (f" (fallan: {malas})" if malas else ""))
    mt = res["metricas"]
    r5 = mt["regla5"]
    print(f"Regla 5 (n = {r5['total']}): {r5['conteos']}; formato válido {r5['pct_formato_valido']} %, "
          f"inexistentes {r5['pct_codigos_inexistentes']} %; aciertos {r5['aciertos']}, accuracy {r5['accuracy']} "
          f"(IC 95 % {mt['ic95_accuracy']['inferior']}-{mt['ic95_accuracy']['superior']})")
    print("Por nivel: " + ", ".join(f"{k} {v['accuracy']}" for k, v in mt["por_nivel"].items())
          + f"; macro-F1 de sector {mt['macro_f1_sector']['valor']}")
    for marca, g in mt["subgrupos"].items():
        print(f"  {marca}: con {g['con']['accuracy']} ({g['con']['filas']} filas), sin {g['sin']['accuracy']} ({g['sin']['filas']} filas)")
    c = mt["concentracion"]
    ce = res["concentracion_extra"]
    print(f"Concentración: {c['clases_existentes_distintas_predichas']} clases existentes distintas; {c['codigo_valido_mas_predicho']} en "
          f"{c['filas_con_ese_codigo']} filas ({c['pct_filas_con_ese_codigo']} %; en validación: {ce['filas_de_validacion_con_ese_codigo']} filas)")
    print("Comparación:")
    for f in res["comparacion"]["tabla"]:
        print(f"  {f['modelo']}: accuracy {f['accuracy']} ({f['ic95']['inferior']}-{f['ic95']['superior']}), sector "
              f"{f['accuracy_sector']}, macro-F1 sector {f['macro_f1_sector']}, clases predichas {f['clases_existentes_distintas_predichas']}")
    dp = res["comparacion"]["diferencia_pareada_contra_tfidf"]
    for nivel in ("clase", "sector"):
        v = dp[nivel]
        print(f"  pareada {nivel} (corrida − TF-IDF): {v['diferencia']:+.3f} (IC 95 % {v['ic95']['inferior']:+.3f} a {v['ic95']['superior']:+.3f}); "
              f"solo TF-IDF acierta {v['filas_solo_a_acierta']}, solo la corrida {v['filas_solo_b_acierta']}")
    print(f"Horas de GPU: {res['horas_gpu']['valor']} (script {res['horas_gpu']['script']})")


def main(argv: Sequence[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from src.muestra_humo import ErrorMuestra
    from src.scian import leer_manifest
    from src.splits import esta_ignorado

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corrida", required=True, choices=list(CORRIDAS))
    parser.add_argument("--escribir", action="store_true", help="escribe el JSON de results/ y la figura")
    args = parser.parse_args(argv)

    conf = CORRIDAS[args.corrida]
    carpeta = REPO_RAIZ / conf["carpeta"]
    manifest = leer_manifest()
    try:
        if not esta_ignorado(carpeta) or not all(esta_ignorado(carpeta / a["ruta"]) for a in inventario(carpeta)):
            raise ErrorCorrida("La carpeta de la salida o alguno de sus archivos no está ignorado por git.")
        contexto = contexto_git(manifest, args.corrida)
        ent = cargar_entradas(args.corrida, manifest)
        resultado, privacidad = evaluar_corrida(ent, manifest, contexto)
    except (ErrorCorrida, ErrorMetricas, ErrorEvaluador, ErrorMuestra, AbortoHumo) as e:
        sys.exit(f"PARADA: {e}")
    imprimir(resultado, privacidad)
    print(f"commit {contexto['commit'][:7]} | árbol limpio: {contexto['arbol_limpio']} | kernel de {contexto['commit_kernel'][:7]}")
    if args.escribir:
        if not contexto["arbol_limpio"]:
            sys.exit("PARADA: el árbol tiene cambios sin commit; el resultado debe registrar un commit limpio.")
        ruta = REPO_RAIZ / conf["resultado"]
        ruta.write_bytes(serializar(resultado))
        print(f"Escrito {conf['resultado']} ({ruta.stat().st_size:,} bytes, SHA-256 {sha256_archivo(ruta)}).")
        fig = REPO_RAIZ / conf["figura"]
        figura_curvas(json.loads(ruta.read_bytes()), fig)
        print(f"Escrita {conf['figura']} ({fig.stat().st_size:,} bytes, SHA-256 {sha256_archivo(fig)}).")


if __name__ == "__main__":
    main()
