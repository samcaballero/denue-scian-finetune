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

Barrido (Fase 2, paso 12; `--barrido`): la salida de `src/barrido_kaggle.py` trae `barrido.json`
(el registro del orquestador), `logs/<nombre>.log` y una carpeta por configuración. Se aplican las
mismas revisiones a cada configuración, más las del orquestador (procesos, memoria y versiones), el
SHA-256 del orden del train igual al de 06b-inicial y la pareada contra 06b-inicial (configuración −
06b-inicial, alineada fila por fila por int(id)). Arma un JSON de `results/` por configuración, el
resumen del barrido con la regla de elección (`src/eleccion_barrido.py`) y la figura de las curvas.

Corrida única (Fase 2, paso 13; `--barrido 4b`): el kernel del 4B usa el orquestador del barrido con
una sola configuración, así que su salida tiene la misma forma y pasa por las mismas revisiones
(descarga, nombres, orquestador, etapas, orden del train, punto 0 del log y versiones contra
06b-inicial). No hay regla de elección: la pareada es contra la ganadora del barrido del 0.6B
(4B − 06b-lr2e-4-r32, alineada fila por fila por int(id)) y contra TF-IDF + LR. Arma un solo JSON
de `results/` con la tabla de validación de todos los modelos, los tiempos estimados contra los
reales y las horas de GPU, y la figura de las curvas del 4B junto a las de la ganadora.

Se detiene (PARADA) ante cualquier problema. Nunca abre `test.jsonl` ni `hard.jsonl`. Los
archivos bajados solo se leen como datos (JSON y texto); el adaptador solo se hashea. En la
terminal solo imprime conteos, códigos SCIAN, hashes y tiempos.

Uso:
  .venv/Scripts/python -m src.evaluar_corrida_clasificador --corrida 06b-inicial              # verifica e imprime
  .venv/Scripts/python -m src.evaluar_corrida_clasificador --corrida 06b-inicial --escribir   # además escribe
  .venv/Scripts/python -m src.evaluar_corrida_clasificador --barrido 06b-barrido [--escribir]
      [--aceptar-incompletas NOMBRE ...]   # solo con la decisión de Samuel
  .venv/Scripts/python -m src.evaluar_corrida_clasificador --barrido 4b [--escribir]
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

import src.barrido_kaggle as bk
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
from src.eleccion_barrido import ErrorEleccion, candidata_desde_resultado, elegir
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


def argumentos_esperados(corrida: Mapping, pasos: int, dir_trabajo: str = DIR_TRABAJO.as_posix()) -> dict:
    """Los argumentos del Trainer que arma el kernel para la corrida (mismo código, en fp16).

    `dir_trabajo` es la carpeta de trabajo del proceso: /tmp/denue_trabajo en una corrida sola y
    /tmp/denue_trabajo/<nombre> en cada configuración del barrido (`barrido_kaggle.correr_configuracion`).
    """
    return argumentos_entrenamiento(corrida, pasos, False, f"{dir_trabajo}/trainer_{corrida['nombre']}")


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


def resumen_etapas(m: Mapping, *, corrida: Mapping, esperados: Mapping[str, str], fuentes_actuales: Mapping[str, str],
                   kernel_igual_al_generado: bool, filas_train: int, filas_val: int, adapter_config: Mapping,
                   archivos_bajados: set[str], llaves: set[str], texto_del_log: str,
                   dir_trabajo: str = DIR_TRABAJO.as_posix()) -> dict:
    """Terminó bien o falló, etapa por etapa, con su dato clave (sin nombres).

    `corrida` es la configuración que generó el kernel (la de 06b-inicial o una del barrido);
    `archivos_bajados` son rutas relativas a la carpeta de la corrida; `dir_trabajo`, la carpeta de
    trabajo del proceso (ver `argumentos_esperados`).
    """
    corrida = dict(corrida)
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
    rev_args = revisar_argumentos(args, argumentos_esperados(corrida, pasos, dir_trabajo), m.get("tapados", []), llaves)
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
        **_cargar_comunes(manifest, raiz),
    )


def _cargar_comunes(manifest: Mapping, raiz: Path = REPO_RAIZ) -> dict:
    """Paquete de Kaggle, validación, catálogo y baselines, con los hashes verificados (sin test ni hard)."""
    archivos = manifest[SECCION][ENTRADA_PAQUETE]["archivos"]
    paquete = {
        a: leer_jsonl_verificado(Path(raiz) / archivos[a]["ruta"], archivos[a]["sha256"]) for a in (ARCHIVO_TRAIN, ARCHIVO_VAL)
    }
    tfidf = json.loads((Path(raiz) / RESULTADO_TFIDF).read_bytes())
    dicc = json.loads((Path(raiz) / RESULTADO_DICCIONARIO).read_bytes())
    ruta_reg = f"{CARPETA_REGISTROS_BASELINES}/{tfidf['commit'][:12]}/tfidf_nombre_C{tfidf['detalle']['C']:g}.json"
    datos_reg = (Path(raiz) / ruta_reg).read_bytes()
    return {
        "train": paquete[ARCHIVO_TRAIN],
        "val_paquete": paquete[ARCHIVO_VAL],
        "val_splits": cargar_referencias(manifest, ENTRADA_VAL, raiz),
        "codigos": cargar_codigos(manifest, raiz),
        "tfidf": tfidf,
        "diccionario": dicc,
        "registro_tfidf": json.loads(datos_reg),
        "ruta_registro_tfidf": ruta_reg,
        "sha_registro_tfidf": sha256_bytes(datos_reg),
    }


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
        m, corrida=ent.espec.corrida, esperados=esperados, fuentes_actuales=contexto["fuentes_actuales_sha256"],
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
    concentracion_extra = _concentracion_extra(predicciones, filas, metricas)
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


def _concentracion_extra(predicciones: Mapping[str, str], filas: Sequence[Mapping], metricas: Mapping) -> dict:
    """Cuántas filas de validación tienen el código más predicho y cuántas respuestas distintas hubo."""
    codigo = metricas["concentracion"]["codigo_valido_mas_predicho"]
    conteo = Counter(f["codigo_act"] for f in filas).most_common(1)[0]
    return {
        "filas_de_validacion_con_ese_codigo": sum(1 for f in filas if f["codigo_act"] == codigo),
        "clase_mas_frecuente_de_validacion": conteo[0],
        "filas_de_validacion_con_la_clase_mas_frecuente": conteo[1],
        "respuestas_distintas_tras_limpiar": len({limpiar(r) for r in predicciones.values()}),
        "codigos_validos_distintos_predichos": len({c for c in map(limpiar, predicciones.values()) if RE_CODIGO.fullmatch(c)}),
    }


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


# --- Barrido (Fase 2, paso 12) ---

# Observaciones de cada barrido que no están en su salida; `--escribir` se niega mientras falte alguna
# (None). Samuel no abrió la página del kernel: la duración sale de barrido.json (inicio y fin del
# orquestador) y del lastRunTime de la CLI, y la cuota la observó Samuel.
BARRIDOS = {
    "06b-barrido": {
        "kernel": "06b-barrido",
        "inicial": "06b-inicial",  # clave en CORRIDAS de la configuración inicial, que también entra a la regla
        "carpeta": "kaggle-output/06b-barrido",
        "log": "denue-scian-06b-barrido.log",  # lo escribe la CLI de Kaggle 2.2.5 junto a la salida
        "resumen": "results/{fecha}_fase2_06b-barrido.json",
        "resultado_configuracion": "results/{fecha}_fase2_{nombre}.json",
        "figura": "docs/figuras/{fecha}_fase2_curvas-barrido-06b.png",
        "observaciones": {
            "version_kaggle": 1,  # kaggle kernels pull -m, 2026-11-23
            "lanzamiento": "2026-11-23, 6:55 pm (hora de la Ciudad de México), por Samuel",
            "last_run_time_cli": "2026-11-24 00:56:39.433000",  # kaggle kernels list --mine (UTC, sin zona)
            "tiempo_kaggle_s": 27000,  # -t de kaggle kernels push
            "imagen_docker": ck.DOCKER_06B_INICIAL,  # aceptada: es la de la metadata de kaggle kernels pull -m
            "estado_cli": "COMPLETE",  # kaggle kernels status, 2026-11-23 23:55 (hora de la Ciudad de México)
            "pagina_del_kernel": "Version #1 Successful y 0 Active Events, vistos por Samuel; no abrió la página del kernel",
            "cuota_antes": "28 h 35 min de 30 h",  # bitácora, paso 11
            "cuota_despues": "23 h 49 min de 30 h",  # observada por Samuel hacia las 11:45 pm del 2026-11-23
            "cuota_descontada_min": 286,  # 4 h 46 min
            "sesion_interactiva": (
                "Hacia las 7:50 pm, Gemini abrió sin querer una sesión interactiva de GPU desde la vista /edit al "
                "revisar el log, y quedó cancelada. La cuota a las 9:21 pm (26 h 12 min) coincide con el tiempo del "
                "barrido, así que no gastó cuota apreciable."
            ),
        },
    },
    # Corrida única del 4B (paso 13): el orquestador del barrido con una sola configuración. No hay regla
    # de elección; la referencia de la pareada es la ganadora del barrido del 0.6B.
    "4b": {
        "tipo": "corrida_unica",
        "kernel": "4b",
        "inicial": "06b-inicial",  # versiones del entorno y orden del train, como en el barrido
        "barrido_referencia": "06b-barrido",  # clave en BARRIDOS del barrido del que sale la referencia
        "resumen_barrido": "results/2026-11-24_fase2_06b-barrido.json",
        "referencia": "06b-lr2e-4-r32",  # la ganadora de la regla de elección del barrido
        "carpeta": "kaggle-output/4b",
        "log": "denue-scian-4b.log",  # lo escribe la CLI de Kaggle 2.2.5 junto a la salida
        "resultado": "results/{fecha}_fase2_4b-lr2e-4-r32.json",
        "figura": "docs/figuras/{fecha}_fase2_curvas-4b-vs-06b.png",
        "observaciones": {
            "version_kaggle": 1,  # instrucción del paso 13; kaggle kernels pull -m: código y metadata son los del repo
            "lanzamiento": (
                "2026-11-24, hacia la 1:05 am (hora de la Ciudad de México), por Samuel; estuvo más de 30 min en Queued"
            ),
            "last_run_time_cli": "2026-11-24 08:23:28.667000",  # kaggle kernels list --mine (UTC, sin zona)
            "tiempo_kaggle_s": 39600,  # -t de kaggle kernels push
            "imagen_docker": ck.DOCKER_06B_INICIAL,  # aceptada: es la de la metadata de kaggle kernels pull -m
            "estado_cli": "COMPLETE",  # kaggle kernels status, 2026-11-24 12:54 pm (hora de la Ciudad de México)
            "fin": "antes de las 12:10 pm del 2026-11-24 (hora de la Ciudad de México), según Samuel",
            "cuota_antes": "23 h 49 min de 30 h",  # bitácora, paso 12 (Parte B)
            "cuota_despues": "16 h 37 min de 30 h",  # observada por Samuel (instrucción del paso 13)
            "cuota_descontada_min": 432,  # 7 h 12 min
        },
    },
}


def duracion_barrido(b: Mapping, last_run_time_cli: str) -> dict:
    """Duración del barrido: del lastRunTime de la CLI (UTC, sin zona) al fin del orquestador en barrido.json."""
    inicio = datetime.datetime.fromisoformat(b["inicio_utc"])
    script_s = b["segundos_desde_arranque"]
    fin = inicio + datetime.timedelta(seconds=script_s)
    cli = datetime.datetime.fromisoformat(last_run_time_cli).replace(tzinfo=datetime.timezone.utc)
    total_s = (fin - cli).total_seconds()
    return {
        "last_run_time_cli_utc": cli.isoformat(),
        "inicio_orquestador_utc": inicio.isoformat(),
        "fin_orquestador_utc": fin.isoformat(),
        "arranque_del_contenedor_s": round((inicio - cli).total_seconds(), 1),
        "script_s": script_s,
        "desde_last_run_time_s": round(total_s, 1),
        "horas": {"script": round(script_s / 3600, 3), "desde_last_run_time": round(total_s / 3600, 3)},
    }
SIGNO_CONTRA_INICIAL = "configuración − 06b-inicial (positivo: la configuración acierta más)"


def verificar_descarga_barrido(inv: Sequence[Mapping], b: Mapping, mediciones: Mapping[str, Mapping | None],
                               hashes_manifest: set[str], archivo_log: str, completas: Sequence[str]) -> dict:
    """Lo bajado contra `archivos_salida` de barrido.json y de cada mediciones.json; PARADA si algo no coincide.

    barrido.json lista todos los archivos de la salida menos él mismo (el log de la CLI tampoco lleva
    hash). Cada configuración completa debe traer exactamente lo que lista su mediciones.json (más el
    propio mediciones.json), y sus predicciones deben ser las de su inferencia.
    """
    locales = {a["ruta"]: a for a in inv}
    kaggle = {a["ruta"]: a for a in b.get("archivos_salida", [])}
    sin_hash = {bk.ARCHIVO_BARRIDO, archivo_log}

    def distinto(r: str, a: Mapping) -> bool:
        return r not in locales or (locales[r]["bytes"], locales[r]["sha256"]) != (a["bytes"], a["sha256"])

    faltan = sorted(set(kaggle) - set(locales))
    distintos = sorted(r for r in set(kaggle) & set(locales) if distinto(r, kaggle[r]))
    extra = sorted(set(locales) - set(kaggle) - sin_hash)
    sin_archivo = sorted(sin_hash - set(locales))
    con_hash_del_manifest = sorted(r for r, a in locales.items() if a["sha256"] in hashes_manifest)
    por_configuracion, malas = {}, []
    for nombre, m in mediciones.items():
        if m is None:
            por_configuracion[nombre] = {"tiene_mediciones": False}
            if nombre in completas:
                malas.append(nombre)
            continue
        propios = {f"{nombre}/{a['ruta']}": a for a in m.get("archivos_salida", [])}
        en_carpeta = {r for r in locales if r.startswith(f"{nombre}/")}
        dif = sorted(r for r, a in propios.items() if distinto(r, a))
        sin_listar = sorted(en_carpeta - set(propios) - {f"{nombre}/{ARCHIVO_MEDICIONES}"})
        sha_pred = locales.get(f"{nombre}/{ARCHIVO_PREDICCIONES}", {}).get("sha256")
        pred_ok = sha_pred is not None and sha_pred == (m.get("inferencia") or {}).get("archivo", {}).get("sha256")
        por_configuracion[nombre] = {
            "tiene_mediciones": True,
            "con_sha256_de_su_mediciones": len(propios),
            "distintos_o_faltan": dif,
            "sin_listar_en_su_mediciones": sin_listar,
            "predicciones_coinciden_con_la_inferencia": pred_ok,
        }
        if nombre in completas and (dif or sin_listar or not pred_ok or not propios):
            malas.append(nombre)
    res = {
        "archivos_bajados": len(locales),
        "con_sha256_de_kaggle": len(kaggle),
        "coinciden_con_kaggle": len(set(kaggle) & set(locales)) - len(distintos),
        "sin_sha256_de_kaggle": sorted(sin_hash & set(locales)),
        "faltan": faltan,
        "distintos": distintos,
        "extra": extra,
        "con_sha256_del_manifest": con_hash_del_manifest,
        "por_configuracion": por_configuracion,
    }
    if faltan or distintos or extra or sin_archivo or con_hash_del_manifest or malas:
        raise ErrorCorrida(
            f"La descarga del barrido no coincide con lo que calculó Kaggle: faltan {len(faltan)}, distintos "
            f"{len(distintos)}, extra {len(extra)}, sin barrido.json o log {len(sin_archivo)}, con hash del manifest "
            f"{len(con_hash_del_manifest)}, configuraciones con problemas {sorted(malas)}."
        )
    return res


def revisar_privacidad_barrido(textos: Mapping[str, str], jsons: Mapping[str, Mapping], idx: Mapping, filas: int,
                               texto_kernel: str, tapados: Mapping[str, int]) -> dict:
    """Revisión de nombres de cada archivo de texto y de los valores de cada JSON (barrido.json y mediciones)."""
    palabras_kernel = palabras_del_texto(texto_kernel)
    entradas = dict(sorted(textos.items()))
    for nombre, obj in sorted(jsons.items()):
        entradas[f"{nombre} (valores)"] = textos_de_json(obj)
    archivos: dict[str, dict] = {}
    for nombre, texto in entradas.items():
        resumen, encontrados = buscar_nombres_indexada(texto, idx)
        archivos[nombre] = {**resumen, "clasificacion": clasificar(encontrados, texto_kernel, palabras_kernel)}
    fugas = sorted(n for n, r in archivos.items() if posible_fuga(r["clasificacion"]))
    return {
        "filas_de_referencia": filas,
        "nombres_buscados": len(idx["nombres"]),
        "llaves_varias_palabras_buscadas": len(idx["varias"]),
        "llaves_una_palabra_buscadas": len(idx["una"]),
        "tapados_por_el_kernel": dict(tapados),
        "archivos": archivos,
        "archivos_con_posible_fuga": fugas,
        "posible_fuga": bool(fugas),
    }


def privacidad_de_configuracion(p: Mapping, nombre: str) -> dict:
    """La parte de la revisión de nombres que toca a una configuración: su carpeta y su log."""
    propios = (f"{nombre}/", f"{bk.CARPETA_LOGS}/{nombre}.log")
    archivos = {a: r for a, r in p["archivos"].items() if a.startswith(propios)}
    return {
        **{k: v for k, v in p.items() if k not in ("archivos", "archivos_con_posible_fuga", "posible_fuga", "tapados_por_el_kernel")},
        "tapados_por_el_kernel": p["tapados_por_el_kernel"].get(nombre, 0),
        "archivos": archivos,
        "archivos_con_posible_fuga": [a for a in p["archivos_con_posible_fuga"] if a in archivos],
        "posible_fuga": any(a in archivos for a in p["archivos_con_posible_fuga"]),
    }


def resumen_orquestador(b: Mapping, *, corridas: Sequence[Mapping], esperados: Mapping[str, str],
                        fuentes_actuales: Mapping[str, str], kernel_igual_al_generado: bool,
                        incompletas_aceptadas: Sequence[str] = ()) -> dict:
    """La etapa del orquestador (barrido.json): procesos, memoria, versiones y orden del train."""
    configs = list(b.get("configuraciones", []))
    nombres = [c["nombre"] for c in corridas]
    completas = [c for c in configs if c.get("estado") == "completa"]
    incompletas = [n for n in nombres if n not in {c.get("nombre") for c in completas}]
    lanzadas = [c for c in configs if c.get("estado") != "omitida"]
    fallas = [f.get("etapa") for f in b.get("fallas", [])]
    permitidas = {f"configuracion_{n}" for n in incompletas_aceptadas}
    gpus = [g.get("nombre", "") for g in b.get("gpu", {}).get("nvidia_smi", [])]
    v = b.get("versiones", {})
    return _etapa(
        {
            "estado_del_orquestador": b.get("estado") == ("completo" if not incompletas else "completo_con_fallas"),
            "fallas_solo_de_incompletas_aceptadas": set(fallas) <= permitidas,
            "incompletas_aceptadas_por_samuel": set(incompletas) <= set(incompletas_aceptadas),
            "todas_t4": bool(gpus) and all("T4" in g for g in gpus),
            "sha256_de_los_datos_verificados": b.get("datos", {}).get("sha256_verificados") == dict(esperados),
            "sha256_esperados_del_kernel": b.get("datos_sha256_esperados") == dict(esperados),
            "fuentes_incrustadas_iguales_a_src": b.get("fuentes_incrustadas_sha256") == dict(fuentes_actuales),
            "kernel_escrito_igual_al_generado": kernel_igual_al_generado,
            "corridas_iguales_a_la_especificacion": b.get("configuracion", {}).get("corridas") == [dict(c) for c in corridas],
            "configuraciones_en_orden": [c.get("nombre") for c in configs] == nombres,
            "versiones_fijadas_sin_diferencias": (
                v.get("fijadas_distintas") == {} and v.get("fijadas_con_pip") == dict(p.split("==") for p in bk.PAQUETES_BARRIDO)
            ),
            "completas_con_codigo_0_y_sin_matar": all(
                c.get("codigo_de_salida") == 0 and c.get("matado_por_tiempo") is False for c in completas
            ),
            "memoria_liberada_tras_cada_proceso": all(c.get("memoria_liberada") is True for c in lanzadas),
            "orden_del_train_igual_al_de_06b_inicial": all(
                (c.get("mediciones") or {}).get("sha256_orden_train") == bk.SHA256_ORDEN_06B_INICIAL for c in completas
            ),
        },
        estado_del_kernel=b.get("estado"),
        etapas_con_falla=fallas,
        incompletas=incompletas,
        advertencias=b.get("advertencias", {}),
        no_fijables=v.get("no_fijables"),
        controlador_nvidia=v.get("controlador_nvidia"),
        instalacion={k: b.get("instalacion", {}).get(k) for k in ("segundos_pip_install", "cambios_freeze")},
        configuraciones=[
            {k: c.get(k) for k in ("nombre", "estado", "estimado_s", "transcurrido_al_empezar_s", "tiempo_maximo_s", "segundos",
                                   "codigo_de_salida", "matado_por_tiempo", "memoria_gpu0_al_salir_mib", "memoria_liberada")}
            for c in configs
        ],
        segundos_desde_arranque=b.get("segundos_desde_arranque"),
        duraciones_s=b.get("duraciones_s"),
    )


def revisar_configuracion_del_barrido(m: Mapping, versiones_inicial: Mapping[str, str]) -> dict:
    """Etapa propia del barrido: orden del train y versiones del entorno contra 06b-inicial.

    Las versiones que se fijan con pip deben ser iguales; las que van con la imagen (torch,
    torchvision y triton) solo dan una advertencia si cambian.
    """
    versiones = m.get("entorno", {}).get("versiones", {})
    fijadas = [p for p in VERSIONES_A_REGISTRAR if p not in bk.NO_FIJABLES]
    distintas = {p: {"06b_inicial": versiones_inicial.get(p), "instalada": versiones.get(p)}
                 for p in fijadas if versiones.get(p) != versiones_inicial.get(p)}
    advertencias = {p: {"06b_inicial": versiones_inicial.get(p), "instalada": versiones.get(p)}
                    for p in bk.NO_FIJABLES if versiones.get(p) != versiones_inicial.get(p)}
    orden = m.get("entrenamiento", {}).get("orden_train", {}).get("sha256_orden")
    return _etapa(
        {
            "orden_del_train_igual_al_de_06b_inicial": orden == bk.SHA256_ORDEN_06B_INICIAL,
            "versiones_fijadas_iguales_a_06b_inicial": not distintas,
        },
        sha256_orden_train=orden,
        versiones_fijadas_distintas=distintas,
        advertencias_no_fijables=advertencias,
    )


def aciertos_de_la_inicial(datos_pred: bytes, res_inicial: Mapping, val_splits: Sequence[Mapping],
                           val_paquete: Sequence[Mapping], codigos, ruta_pred: str = ARCHIVO_PREDICCIONES,
                           etiqueta: str = "la inicial") -> tuple[dict[str, list[bool]], dict]:
    """Aciertos de 06b-inicial por nivel en orden de int(id), comprobados contra su JSON de results/.

    También sirve para una configuración del barrido (paso 13): `ruta_pred` es la ruta de sus
    predicciones en `archivos_bajados` de su resultado (con el prefijo de su carpeta).
    """
    sha = sha256_bytes(datos_pred)
    esperado = next((a["sha256"] for a in res_inicial["corrida_kaggle"]["archivos_bajados"] if a["ruta"] == ruta_pred), None)
    if sha != esperado:
        raise ErrorCorrida(f"Las predicciones de {etiqueta} no tienen el SHA-256 de las que evaluó su resultado.")
    pred = leer_predicciones(datos_pred)
    filas = alinear(pred, val_splits, val_paquete)
    ac = aciertos_alineados(pred, filas, codigos)
    m = res_inicial["metricas"]
    c = {
        "conteo_clase": sum(ac["clase"]) == m["regla5"]["aciertos"] == m["por_nivel"]["clase"]["aciertos"],
        "ic_clase_reproducido": intervalo_bootstrap(ac["clase"]) == m["ic95_accuracy"],
        "conteo_sector": sum(ac["sector"]) == m["por_nivel"]["sector"]["aciertos"],
        "ic_sector_reproducido": intervalo_bootstrap(ac["sector"]) == m["por_nivel"]["sector"]["ic95"],
    }
    malas = sorted(k for k, v in c.items() if not v)
    if malas:
        raise ErrorCorrida(f"Los aciertos de {etiqueta} no reproducen su resultado: {malas}.")
    return ac, {"predicciones_sha256": sha, "filas": len(filas), "comprobaciones": c}


def contra_la_inicial(aciertos_inicial: Mapping[str, list[bool]], aciertos: Mapping[str, list[bool]], res_inicial: Mapping,
                      info_inicial: Mapping, ruta_inicial: str) -> dict:
    """Diferencia pareada configuración − 06b-inicial, en clase y en sector, sobre las mismas filas."""
    mi = res_inicial["metricas"]
    return {
        "resultado_inicial": ruta_inicial,
        "accuracy_inicial": mi["regla5"]["accuracy"],
        "ic95_inicial": mi["ic95_accuracy"],
        "signo": SIGNO_CONTRA_INICIAL,
        "clase": diferencia_pareada_bootstrap(aciertos_inicial["clase"], aciertos["clase"]),
        "sector": diferencia_pareada_bootstrap(aciertos_inicial["sector"], aciertos["sector"]),
        "definicion": DEFINICION_PAREADA,
        "alineacion": {
            "filas": len(aciertos["clase"]),
            "orden": "int(id) de data/splits/val.jsonl",
            "comprobada_fila_por_fila": True,
            "predicciones_inicial_sha256": info_inicial["predicciones_sha256"],
            "comprobaciones_inicial": dict(info_inicial["comprobaciones"]),
            "metodo": (
                "Las dos listas de aciertos salen de alinear las predicciones de cada modelo con data/splits/val.jsonl "
                "(mismos ids que el val.jsonl del paquete, mismas etiquetas) en orden de int(id); en cada posición, el "
                "acierto es el de la respuesta de esa fila contra su codigo_act. Las de 06b-inicial salen de su "
                "predicciones.jsonl con el SHA-256 de su resultado y reproducen sus conteos e IC."
            ),
        },
    }


def _ruta(plantilla: str, fecha: str, nombre: str = "") -> str:
    return plantilla.format(fecha=fecha, nombre=nombre)


def cargar_barrido(barrido: str, manifest: Mapping, raiz: Path = REPO_RAIZ) -> SimpleNamespace:
    """Todo lo que lee la evaluación del barrido, con los hashes verificados. Nunca abre test.jsonl ni hard.jsonl."""
    conf = BARRIDOS[barrido]
    espec = ck.KERNELS[conf["kernel"]]
    carpeta = Path(raiz) / conf["carpeta"]
    inv = inventario(carpeta)
    textos = {
        a["ruta"]: (carpeta / a["ruta"]).read_bytes().decode("utf-8")
        for a in inv
        if not a["ruta"].endswith(EXTENSIONES_BINARIAS) and a["ruta"] != conf["log"]
    }
    if bk.ARCHIVO_BARRIDO not in textos or not (carpeta / conf["log"]).is_file():
        raise ErrorCorrida(f"Falta {bk.ARCHIVO_BARRIDO} o el log de la CLI en la descarga.")
    entradas_log = leer_log((carpeta / conf["log"]).read_bytes())
    textos[conf["log"]] = texto_log(entradas_log)
    nombres = [c["nombre"] for c in espec.corridas]
    mediciones = {
        n: json.loads(textos[f"{n}/{ARCHIVO_MEDICIONES}"]) if f"{n}/{ARCHIVO_MEDICIONES}" in textos else None for n in nombres
    }
    adapter = {n: json.loads(textos.get(f"{n}/{CARPETA_ADAPTADOR}/adapter_config.json", "{}")) for n in nombres}
    conf_ini = CORRIDAS[conf["inicial"]]
    return SimpleNamespace(
        barrido=barrido,
        conf=conf,
        espec=espec,
        carpeta=carpeta,
        inventario=inv,
        barrido_json=json.loads(textos[bk.ARCHIVO_BARRIDO]),
        log=entradas_log,
        textos=textos,
        mediciones=mediciones,
        adapter_configs=adapter,
        resultado_inicial=json.loads((Path(raiz) / conf_ini["resultado"]).read_bytes()),
        predicciones_inicial=(Path(raiz) / conf_ini["carpeta"] / ARCHIVO_PREDICCIONES).read_bytes(),
        **_cargar_comunes(manifest, raiz),
    )


def armar_resultado_configuracion(*, manifest, contexto, ent, corrida, m, proceso, descarga, privacidad, etapas, metricas,
                                  concentracion_extra, comparacion, contra, curvas, advertencias) -> dict:
    """JSON de results/ de una configuración del barrido (regla 4); sin nombres, llaves, ids ni respuestas crudas."""
    nombre = corrida["nombre"]
    obs = ent.conf["observaciones"]
    archivos = manifest[SECCION][ENTRADA_PAQUETE]["archivos"]
    sp = manifest["splits"]
    proceso_s = proceso.get("segundos")
    script_s = m.get("segundos_desde_arranque")
    propios = (f"{nombre}/", f"{bk.CARPETA_LOGS}/{nombre}.log")
    return {
        "fecha": contexto["fecha"],
        "experimento": f"fase2_paso12_{nombre}",
        "commit": contexto["commit"],
        "arbol_limpio": contexto["arbol_limpio"],
        "modelo": {
            "nombre": f"qwen3_0.6b_qlora_{nombre}",
            "repo": corrida["modelo"]["repo"],
            "revision": corrida["modelo"]["revision"],
            "metodo": "QLoRA (NF4 de bitsandbytes) con Unsloth; enable_thinking=False; responde solo el código de 6 dígitos",
            "adaptador_sha256": next(
                (a["sha256"] for a in ent.inventario if a["ruta"] == f"{nombre}/{CARPETA_ADAPTADOR}/adapter_model.safetensors"), None
            ),
        },
        "configuracion": {
            "corrida": dict(corrida),
            "del_kernel": m.get("configuracion", {}),
            "barrido": ent.barrido,
            "kernel": f"{ent.espec.carpeta_relativa}/{ent.espec.archivo_kernel}",
            "kernel_generador": "src/construir_kernel.py",
            "commit_kernel": contexto["commit_kernel"],
            "kernel_kaggle": {"id": ent.espec.id_kernel, "version": obs["version_kaggle"]},
            "imagen_docker": obs["imagen_docker"],
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
            "valor": round(proceso_s / 3600, 3) if proceso_s else None,
            "metodo": (
                "Duración del proceso de esta configuración en barrido.json (una sola T4 de la sesión GPU T4 x2). Las horas "
                "del barrido completo (página del kernel) y la cuota están en el resumen del barrido."
            ),
            "script": round(script_s / 3600, 3) if script_s else None,
            "resumen_del_barrido": _ruta(ent.conf["resumen"], contexto["fecha"]),
        },
        "corrida_kaggle": {
            "estado_cli": obs["estado_cli"],
            "last_run_time_cli": obs["last_run_time_cli"],
            "proceso": {k: proceso.get(k) for k in ("estado", "codigo_de_salida", "matado_por_tiempo", "estimado_s", "tiempo_maximo_s",
                                                     "transcurrido_al_empezar_s", "segundos", "memoria_gpu0_al_salir_mib")},
            "estado_kernel": m.get("estado"),
            "fallas": len(m.get("fallas", [])),
            "tiempo": {"inicio_script_utc": m.get("inicio_utc"), "script_s": script_s, "duraciones_etapas_s": m.get("duraciones_s")},
            "archivos_bajados": [dict(a) for a in ent.inventario if a["ruta"].startswith(propios)],
            "descarga": descarga["por_configuracion"][nombre],
        },
        "privacidad": privacidad_para_resultado(privacidad),
        "etapas": etapas,
        "metricas": metricas,
        "concentracion_extra": concentracion_extra,
        "definiciones": DEFINICIONES,
        "comparacion": comparacion,
        "contra_06b_inicial": contra,
        "curvas": curvas,
        "advertencias": advertencias,
    }


def fila_de_la_tabla(nombre: str, corrida: Mapping, metricas: Mapping, contra: Mapping | None, estimado_s, proceso_s,
                     resultado: str) -> dict:
    """Una fila de la tabla del barrido (la inicial no tiene pareada contra sí misma)."""
    def pareada(nivel):
        return None if contra is None else {k: contra[nivel][k] for k in ("diferencia", "ic95", "filas_solo_a_acierta", "filas_solo_b_acierta")}

    return {
        "nombre": nombre,
        "rank": corrida["lora"]["r"],
        "lora_alpha": corrida["lora"]["lora_alpha"],
        "learning_rate": corrida["learning_rate"],
        "aciertos": metricas["regla5"]["aciertos"],
        "accuracy": metricas["regla5"]["accuracy"],
        "ic95": metricas["ic95_accuracy"],
        "accuracy_sector": metricas["por_nivel"]["sector"]["accuracy"],
        "ic95_sector": metricas["por_nivel"]["sector"]["ic95"],
        "macro_f1_sector": metricas["macro_f1_sector"]["valor"],
        "pct_formato_valido": metricas["regla5"]["pct_formato_valido"],
        "pct_codigos_inexistentes": metricas["regla5"]["pct_codigos_inexistentes"],
        "clases_existentes_distintas_predichas": metricas["concentracion"]["clases_existentes_distintas_predichas"],
        "pareada_clase_contra_inicial": pareada("clase"),
        "pareada_sector_contra_inicial": pareada("sector"),
        "estimado_s": estimado_s,
        "proceso_s": proceso_s,
        "resultado": resultado,
    }


def evaluar_barrido(ent: SimpleNamespace, manifest: Mapping, contexto: Mapping,
                    aceptar_incompletas: Sequence[str] = ()) -> tuple[dict, dict[str, dict], dict]:
    """Verifica, revisa nombres, resume etapas, mide cada configuración, aplica la regla y arma el resumen.

    Devuelve el resumen, los resultados por configuración (por nombre) y la revisión de privacidad
    completa (con líneas, solo para la terminal). PARADA ante cualquier problema que no sea una etapa
    con comprobaciones fallidas; esas las junta `problemas_del_barrido` y detienen `main`.
    """
    b = ent.barrido_json
    corridas = [dict(c) for c in ent.espec.corridas]
    nombres = [c["nombre"] for c in corridas]
    desconocidas = sorted(set(aceptar_incompletas) - set(nombres))
    if desconocidas:
        raise ErrorCorrida(f"--aceptar-incompletas nombra configuraciones que no son del barrido: {desconocidas}.")
    procesos = {c.get("nombre"): c for c in b.get("configuraciones", [])}
    completas = [n for n in nombres if procesos.get(n, {}).get("estado") == "completa"]
    incompletas = [n for n in nombres if n not in completas]

    descarga = verificar_descarga_barrido(
        ent.inventario, b, ent.mediciones, ck.hashes_del_manifest(manifest), ent.conf["log"], completas
    )
    filas_ref = ent.train + ent.val_paquete
    ref = referencias_de_nombres(filas_ref)
    idx = indice_de_nombres(ref)
    jsons = {bk.ARCHIVO_BARRIDO: b, **{f"{n}/{ARCHIVO_MEDICIONES}": m for n, m in ent.mediciones.items() if m is not None}}
    tapados = {n: len(m.get("tapados", [])) for n, m in ent.mediciones.items() if m is not None}
    privacidad = revisar_privacidad_barrido(ent.textos, jsons, idx, len(filas_ref), contexto["texto_kernel"], tapados)
    if privacidad["posible_fuga"]:
        raise ErrorCorrida(f"Posible fuga de nombres en {privacidad['archivos_con_posible_fuga']}; decide Samuel.")
    no_aceptadas = [n for n in incompletas if n not in aceptar_incompletas]
    if no_aceptadas:
        raise ErrorCorrida(f"Configuraciones incompletas: {no_aceptadas}; decide Samuel (--aceptar-incompletas).")

    archivos = manifest[SECCION][ENTRADA_PAQUETE]["archivos"]
    esperados = {a: archivos[a]["sha256"] for a in ARCHIVOS_ENTRADA}
    orquestador = resumen_orquestador(
        b, corridas=corridas, esperados=esperados, fuentes_actuales=contexto["fuentes_actuales_sha256"],
        kernel_igual_al_generado=contexto["kernel_igual_al_generado"], incompletas_aceptadas=aceptar_incompletas,
    )
    clave_ini = ent.conf["inicial"]
    ruta_ini = CORRIDAS[clave_ini]["resultado"]
    res_ini = ent.resultado_inicial
    aciertos_ini, info_ini = aciertos_de_la_inicial(ent.predicciones_inicial, res_ini, ent.val_splits, ent.val_paquete, ent.codigos)
    versiones_ini = res_ini["para_el_barrido"]["versiones_entorno"]
    llaves = set(ref["varias"] | ref["una"])
    fecha = contexto["fecha"]

    resultados: dict[str, dict] = {}
    comprobacion_tfidf = None
    for corrida in corridas:
        n = corrida["nombre"]
        if n not in completas:
            continue
        m = ent.mediciones[n]
        log = ent.textos.get(f"{bk.CARPETA_LOGS}/{n}.log")
        if log is None:
            raise ErrorCorrida(f"Falta el log de {n}.")
        prefijo = f"{n}/"
        etapas = resumen_etapas(
            m, corrida=corrida, esperados=esperados, fuentes_actuales=contexto["fuentes_actuales_sha256"],
            kernel_igual_al_generado=contexto["kernel_igual_al_generado"], filas_train=len(ent.train),
            filas_val=len(ent.val_paquete), adapter_config=ent.adapter_configs[n],
            archivos_bajados={a["ruta"][len(prefijo):] for a in ent.inventario if a["ruta"].startswith(prefijo)},
            llaves=llaves, texto_del_log=log,
            dir_trabajo=(DIR_TRABAJO / n).as_posix(),  # cada proceso del barrido trabaja en su propia carpeta
        )
        etapas["etapas"]["barrido"] = revisar_configuracion_del_barrido(m, versiones_ini)
        etapas["todas_ok"] = all(x["estado"] == "ok" for x in etapas["etapas"].values())

        predicciones = leer_predicciones(ent.textos[f"{n}/{ARCHIVO_PREDICCIONES}"].encode("utf-8"))
        filas = alinear(predicciones, ent.val_splits, ent.val_paquete)
        metricas = evaluar_validacion(predicciones, filas, ent.codigos)
        aciertos = aciertos_alineados(predicciones, filas, ent.codigos)
        if intervalo_bootstrap(aciertos["clase"]) != metricas["ic95_accuracy"] or sum(aciertos["clase"]) != metricas["regla5"]["aciertos"]:
            raise ErrorCorrida(f"{n}: los aciertos alineados no reproducen las métricas del evaluador.")
        if comprobacion_tfidf is None:
            comprobacion_tfidf = comprobar_registro_tfidf(ent.registro_tfidf, ent.tfidf, filas)
        comp = comparacion(aciertos, metricas, ent.tfidf, ent.diccionario, ent.registro_tfidf, comprobacion_tfidf,
                           ent.ruta_registro_tfidf, ent.sha_registro_tfidf)
        contra = contra_la_inicial(aciertos_ini, aciertos, res_ini, info_ini, ruta_ini)
        advertencias = [
            f"{p} es {v['instalada']}; en 06b-inicial era {v['06b_inicial']} (va con la imagen de Docker, no se fija con pip)."
            for p, v in etapas["etapas"]["barrido"]["advertencias_no_fijables"].items()
        ]
        resultado = armar_resultado_configuracion(
            manifest=manifest, contexto=contexto, ent=ent, corrida=corrida, m=m, proceso=procesos[n], descarga=descarga,
            privacidad=privacidad_de_configuracion(privacidad, n), etapas=etapas,
            metricas={k: v for k, v in metricas.items() if k != "definiciones"},
            concentracion_extra=_concentracion_extra(predicciones, filas, metricas), comparacion=comp, contra=contra,
            curvas={
                "validacion": etapas["etapas"]["entrenamiento"]["perdida_validacion"],
                "train_por_ventana": curva_de_train_agregada(m.get("entrenamiento", {}).get("curvas", {}).get("train", [])),
                "ventana_pasos": VENTANA_CURVA,
            },
            advertencias=advertencias,
        )
        candado_resultado(resultado, ref)
        resultados[n] = resultado

    corrida0 = corridas[0]
    pasos = pasos_por_epoca(len(ent.train), corrida0["lote"], corrida0["gradient_accumulation_steps"]) * corrida0["epocas"]
    candidatas = [candidata_desde_resultado(res_ini, True)]
    for corrida in corridas:
        n = corrida["nombre"]
        if n in resultados:
            candidatas.append(candidata_desde_resultado(resultados[n], False))
        else:
            med = procesos.get(n, {}).get("mediciones") or {}
            candidatas.append({
                "nombre": n, "es_inicial": False, "rank": corrida["lora"]["r"], "learning_rate": corrida["learning_rate"],
                "pasos_hechos": med.get("pasos_hechos"), "filas_inferidas": med.get("filas_inferidas"),
                "aciertos_clase": None, "n": len(ent.val_paquete),
            })
    try:
        eleccion = elegir(candidatas, pasos=pasos, filas=len(ent.val_paquete))
    except ErrorEleccion as e:
        raise ErrorCorrida(f"No se pudo aplicar la regla de elección: {e}") from e

    rutas = {n: _ruta(ent.conf["resultado_configuracion"], fecha, n) for n in resultados}
    shas = {n: sha256_bytes(serializar(r)) for n, r in resultados.items()}
    resumen = armar_resumen_barrido(
        manifest=manifest, contexto=contexto, ent=ent, corridas=corridas, procesos=procesos, descarga=descarga,
        privacidad=privacidad, orquestador=orquestador, resultados=resultados, rutas=rutas, shas=shas,
        eleccion=eleccion, incompletas=incompletas, ruta_inicial=ruta_ini, clave_inicial=clave_ini,
    )
    candado_resultado(resumen, ref)
    return resumen, resultados, privacidad


def armar_resumen_barrido(*, manifest, contexto, ent, corridas, procesos, descarga, privacidad, orquestador, resultados, rutas,
                          shas, eleccion, incompletas, ruta_inicial, clave_inicial) -> dict:
    """Resumen del barrido en results/: tabla de las 5, regla de elección, tiempos, horas, cuota y advertencias."""
    obs = ent.conf["observaciones"]
    b = ent.barrido_json
    res_ini = ent.resultado_inicial
    corrida_ini = res_ini["configuracion"]["corrida"]
    archivos = manifest[SECCION][ENTRADA_PAQUETE]["archivos"]
    sp = manifest["splits"]
    tiempo = duracion_barrido(b, obs["last_run_time_cli"])
    tabla = [fila_de_la_tabla(clave_inicial, corrida_ini, res_ini["metricas"], None, None,
                              res_ini["corrida_kaggle"]["tiempo"]["script_s"], ruta_inicial)]
    for c in corridas:
        n = c["nombre"]
        if n in resultados:
            r = resultados[n]
            tabla.append(fila_de_la_tabla(n, c, r["metricas"], r["contra_06b_inicial"], procesos[n].get("estimado_s"),
                                          procesos[n].get("segundos"), rutas[n]))
    advertencias = [eleccion["advertencia"]]
    for k, v in (b.get("advertencias") or {}).items():
        advertencias.append(f"{k} es {v.get('instalada')}; en 06b-inicial era {v.get('06b_inicial')} (no se fija con pip).")
    for n, r in resultados.items():
        advertencias += [f"{n}: {a}" for a in r["advertencias"]]
    for n in incompletas:
        advertencias.append(f"{n} quedó incompleta: se reporta, no entra a la regla y no se repite antes del 4B.")
    curvas = {clave_inicial: {"validacion": res_ini["curvas"]["validacion"], "train_por_ventana": res_ini["curvas"]["train_por_ventana"]}}
    curvas.update({n: {"validacion": r["curvas"]["validacion"], "train_por_ventana": r["curvas"]["train_por_ventana"]}
                   for n, r in resultados.items()})
    return {
        "fecha": contexto["fecha"],
        "experimento": f"fase2_paso12_{ent.barrido}",
        "commit": contexto["commit"],
        "arbol_limpio": contexto["arbol_limpio"],
        "modelo": {
            "nombre": "qwen3_0.6b_qlora_barrido",
            "repo": corrida_ini["modelo"]["repo"],
            "revision": corrida_ini["modelo"]["revision"],
            "metodo": "QLoRA (NF4 de bitsandbytes) con Unsloth; enable_thinking=False; responde solo el código de 6 dígitos",
        },
        "configuracion": {
            "barrido": ent.barrido,
            "inicial": {"nombre": clave_inicial, "corrida": corrida_ini, "resultado": ruta_inicial},
            "corridas": corridas,
            "kernel": f"{ent.espec.carpeta_relativa}/{ent.espec.archivo_kernel}",
            "kernel_generador": "src/construir_kernel.py",
            "commit_kernel": contexto["commit_kernel"],
            "kernel_kaggle": {"id": ent.espec.id_kernel, "version": obs["version_kaggle"]},
            "imagen_docker": obs["imagen_docker"],
            "tiempo_kaggle_s": obs["tiempo_kaggle_s"],
            "script": SCRIPT,
            "regla_de_eleccion": "src/eleccion_barrido.py",
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
            "valor": tiempo["horas"]["desde_last_run_time"],
            "metodo": (
                "Del lastRunTime de `kaggle kernels list --mine` (UTC) al fin del orquestador (inicio_utc de barrido.json "
                "más segundos_desde_arranque), con el arranque del contenedor; las 4 configuraciones del barrido en una "
                "sesión GPU T4 x2 con una sola T4 en uso. Samuel no abrió la página del kernel."
            ),
            "script": tiempo["horas"]["script"],
            "por_configuracion": {n: round(procesos[n]["segundos"] / 3600, 3) for n in resultados if procesos[n].get("segundos")},
            "cuota_kaggle": {"antes": obs["cuota_antes"], "despues": obs["cuota_despues"], "descontado_min": obs["cuota_descontada_min"]},
        },
        "corrida_kaggle": {
            "estado_cli": obs["estado_cli"],
            "pagina_del_kernel": obs["pagina_del_kernel"],
            "lanzamiento": obs["lanzamiento"],
            "last_run_time_cli": obs["last_run_time_cli"],
            "tiempo": tiempo,
            "sesion_interactiva": obs["sesion_interactiva"],
            "estado_kernel": b.get("estado"),
            "fallas": len(b.get("fallas", [])),
            "archivos_bajados": [dict(a) for a in ent.inventario],
            "descarga": {k: v for k, v in descarga.items() if k != "por_configuracion"},
        },
        "privacidad": privacidad_para_resultado(privacidad),
        "orquestador": orquestador,
        "tabla": tabla,
        "eleccion": eleccion,
        "resultados_por_configuracion": {n: {"ruta": rutas[n], "sha256": shas[n]} for n in resultados},
        "incompletas": incompletas,
        "advertencias": advertencias,
        "curvas": curvas,
        "definiciones": {"pareada": DEFINICION_PAREADA, "signo_contra_inicial": SIGNO_CONTRA_INICIAL, **DEFINICIONES},
    }


def problemas_del_barrido(resumen: Mapping, resultados: Mapping[str, Mapping]) -> list[str]:
    """Comprobaciones que fallaron (en el orquestador o en una etapa de una configuración); cada una es PARADA."""
    res = [f"orquestador: {k}" for k, v in resumen["orquestador"]["comprobaciones"].items() if not v]
    for n, r in resultados.items():
        for etapa, e in r["etapas"]["etapas"].items():
            res += [f"{n}, {etapa}: {k}" for k, v in e.get("comprobaciones", {}).items() if not v]
    return res


def imprimir_barrido(resumen: Mapping, resultados: Mapping[str, Mapping], privacidad: Mapping) -> None:
    d = resumen["corrida_kaggle"]
    x = d["descarga"]
    print(f"Archivos bajados: {len(d['archivos_bajados'])}; coinciden con barrido.json {x['coinciden_con_kaggle']} de "
          f"{x['con_sha256_de_kaggle']}; sin hash de Kaggle: {x['sin_sha256_de_kaggle']}; con hash del manifest: "
          f"{x['con_sha256_del_manifest']}")
    p = privacidad
    print(f"Privacidad: posible fuga = {p['posible_fuga']}; tapados por el kernel = {p['tapados_por_el_kernel']}")
    for nombre, r in p["archivos"].items():
        c = r["clasificacion"]
        print(f"  {nombre}: crudos {r['nombres_crudos']} (cortos dentro de un token {c['nombres_crudos_cortos_solo_dentro_de_un_token']}, "
              f"posible fuga {c['nombres_crudos_posible_fuga']}), varias palabras {r['llaves_varias_palabras']} "
              f"(solo dígitos {c['llaves_varias_palabras_solo_digitos']}), una palabra {r['llaves_una_palabra']}")
    o = resumen["orquestador"]
    malas = [k for k, v in o["comprobaciones"].items() if not v]
    print(f"Orquestador: {o['estado']}" + (f" (fallan: {malas})" if malas else "") + f"; advertencias: {o['advertencias']}")
    for c in o["configuraciones"]:
        print(f"  {c['nombre']}: {c['estado']}, {c['segundos']} s (estimado {c['estimado_s']} s), salida {c['codigo_de_salida']}, "
              f"GPU 0 al salir {c['memoria_gpu0_al_salir_mib']} MiB")
    for n, r in resultados.items():
        estados = {e: v["estado"] for e, v in r["etapas"]["etapas"].items()}
        malas = [f"{e}: {k}" for e, v in r["etapas"]["etapas"].items() for k, ok in v.get("comprobaciones", {}).items() if not ok]
        print(f"Etapas de {n}: " + ", ".join(f"{e} {s}" for e, s in estados.items()) + (f" (fallan: {malas})" if malas else ""))
    print("Tabla (accuracy de clase y de sector con IC 95 %; pareada = configuración − 06b-inicial):")
    for f in resumen["tabla"]:
        pc, ps = f["pareada_clase_contra_inicial"], f["pareada_sector_contra_inicial"]
        texto_p = "" if pc is None else (
            f"; pareada clase {pc['diferencia']:+.3f} ({pc['ic95']['inferior']:+.3f} a {pc['ic95']['superior']:+.3f}), "
            f"sector {ps['diferencia']:+.3f} ({ps['ic95']['inferior']:+.3f} a {ps['ic95']['superior']:+.3f})"
        )
        print(f"  {f['nombre']} (r {f['rank']}, lr {f['learning_rate']:g}): {f['accuracy']} ({f['ic95']['inferior']}-{f['ic95']['superior']}), "
              f"sector {f['accuracy_sector']} ({f['ic95_sector']['inferior']}-{f['ic95_sector']['superior']}), macro-F1 sector "
              f"{f['macro_f1_sector']}, formato válido {f['pct_formato_valido']} %, inexistentes {f['pct_codigos_inexistentes']} %"
              f"{texto_p}; proceso {f['proceso_s']} s")
    e = resumen["eleccion"]
    print(f"Regla de elección: gana {e['ganadora']} ({e['clausula_texto']}); empatadas {e['empatadas']}; "
          f"excluidas por incompletas {e['excluidas_incompletas']}")
    for a in resumen["advertencias"]:
        print(f"  ADVERTENCIA: {a}")
    print(f"Horas de GPU: {resumen['horas_gpu']['valor']} (script {resumen['horas_gpu']['script']})")


# Posiciones 1 a 5 de la paleta categórica de referencia (las mismas SERIE_1 y SERIE_2 de las demás
# figuras), en orden fijo: la inicial y luego las configuraciones en el orden del barrido. Validadas con
# validate_palette.js en modo claro: pares adyacentes con ΔE CVD de 9.1 y normal de 19.6 como mínimo.
# Tres quedan bajo 3:1 de contraste con la superficie: por eso cada serie lleva su propio marcador y la
# leyenda escribe en texto el valor final de validación.
SERIES_BARRIDO = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
MARCADORES_BARRIDO = ("o", "s", "^", "D", "v")
# El panel de validación empieza aquí: en los primeros pasos la pérdida cae de ~3 a ~0.3 y, con el eje
# desde 0, las diferencias finales entre configuraciones no se verían. El paso 0 y el 500 quedan en los JSON.
PASO_INICIAL_PANEL_VALIDACION = 1000


def figura_barrido(resumen: Mapping, ruta: Path) -> None:
    """Curvas de pérdida de las 5 configuraciones (train por ventanas y validación), desde el resumen."""
    curvas = resumen["curvas"]
    if len(curvas) > len(SERIES_BARRIDO):
        raise ErrorCorrida("La figura del barrido admite hasta 5 series.")
    corrida = resumen["configuracion"]["inicial"]["corrida"]
    series = [(nombre, c, SERIES_BARRIDO[i], MARCADORES_BARRIDO[i]) for i, (nombre, c) in enumerate(curvas.items())]
    _figura_dos_paneles(
        series,
        corrida=corrida,
        titulo="Curvas de pérdida del barrido del Qwen3-0.6B",
        subtitulo=(
            f"06b-inicial y {len(curvas) - 1} configuraciones del barrido (alpha = rank); lote {corrida['lote']}, "
            f"{corrida['epocas']} época; validación cada {corrida['eval_cada_pasos']} pasos y al final."
        ),
        ruta=ruta,
    )


def _figura_dos_paneles(series: Sequence[tuple[str, Mapping, str, str]], *, corrida: Mapping, titulo: str, subtitulo: str,
                        ruta: Path, columnas_leyenda: int = 3) -> None:
    """Dos paneles: train por ventanas y validación desde PASO_INICIAL_PANEL_VALIDACION.

    `series` son tuplas (nombre, curvas, color, marcador); `curvas` trae `train_por_ventana` y
    `validacion` como en los JSON de results/.
    """
    from src.exploracion_clasificador import EJE, REJILLA, TINTA_2, _figura, _guardar, _pyplot, _titulos

    plt = _pyplot()
    fig = _figura(plt)
    ax_t = fig.add_axes([0.07, 0.13, 0.41, 0.56])
    ax_v = fig.add_axes([0.56, 0.13, 0.41, 0.56])
    manijas, etiquetas = [], []
    pasos_max = 0
    for nombre, c, color, marca in series:
        xs = [(v["paso_inicial"] + v["paso_final"]) / 2 for v in c["train_por_ventana"]]
        ax_t.plot(xs, [v["media"] for v in c["train_por_ventana"]], color=color, linewidth=2.5)
        cola = [v for v in c["validacion"] if v["paso"] >= PASO_INICIAL_PANEL_VALIDACION]
        (linea,) = ax_v.plot([v["paso"] for v in cola], [v["eval_loss"] for v in cola], color=color, linewidth=2.5,
                             marker=marca, markersize=10, markeredgecolor="#fcfcfb", markeredgewidth=1.5)
        manijas.append(linea)
        etiquetas.append(f"{nombre} (validación final {c['validacion'][-1]['eval_loss']:.3f})")
        pasos_max = max([pasos_max, *(v["paso"] for v in c["validacion"]), *xs])
    for ax, titulo_panel in ((ax_t, f"Train (media de cada {VENTANA_CURVA} pasos)"),
                             (ax_v, f"Validación (1,000 filas), desde el paso {PASO_INICIAL_PANEL_VALIDACION:,}")):
        ax.set_title(titulo_panel, fontsize=26, color=TINTA_2, pad=14)
        ax.set_xlabel(f"Paso de entrenamiento (lote de {corrida['lote']})")
        ax.grid(axis="y", color=REJILLA, linewidth=1.2)
        ax.tick_params(colors=TINTA_2)
        for lado in ("left", "bottom"):
            ax.spines[lado].set_color(EJE)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax_t.set_xlim(0, pasos_max * 1.01)
    ax_t.set_ylim(0, None)
    ax_v.set_xlim(PASO_INICIAL_PANEL_VALIDACION * 0.9, pasos_max * 1.02)
    ax_v.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.3f}"))
    ax_t.set_ylabel("Pérdida (entropía cruzada por ficha)")
    fig.legend(manijas, etiquetas, loc="upper left", bbox_to_anchor=(0.04, 0.84), ncol=columnas_leyenda, fontsize=19,
               handlelength=2.2, columnspacing=1.6)
    _titulos(fig, titulo, subtitulo)
    _guardar(fig, ruta)
    plt.close(fig)


def escribir_barrido(resumen: Mapping, resultados: Mapping[str, Mapping], conf: Mapping, raiz: Path = REPO_RAIZ) -> list[str]:
    """Escribe los JSON de las configuraciones, el resumen y la figura; devuelve las rutas escritas."""
    escritos = []
    for n, r in resultados.items():
        info = resumen["resultados_por_configuracion"][n]
        datos = serializar(r)
        if sha256_bytes(datos) != info["sha256"]:
            raise ErrorCorrida(f"El JSON de {n} no es el que registra el resumen.")
        (Path(raiz) / info["ruta"]).write_bytes(datos)
        escritos.append(info["ruta"])
    ruta_resumen = _ruta(conf["resumen"], resumen["fecha"])
    (Path(raiz) / ruta_resumen).write_bytes(serializar(resumen))
    escritos.append(ruta_resumen)
    ruta_fig = _ruta(conf["figura"], resumen["fecha"])
    figura_barrido(json.loads((Path(raiz) / ruta_resumen).read_bytes()), Path(raiz) / ruta_fig)
    escritos.append(ruta_fig)
    return escritos


# --- Corrida única: el 4B (Fase 2, paso 13) ---

SIGNO_CONTRA_REFERENCIA = "{corrida} − {referencia} (positivo: {corrida} acierta más)"
# Si el contenedor arrancó más de esto después del lastRunTime, esa marca incluye tiempo en cola y las horas
# de GPU son las del orquestador (propuesta de Claude Code en el plan del paso 13, aprobada por Samuel).
UMBRAL_COLA_S = 300
# Etapas de mediciones.json de cada parte de la estimación (`desglose_s` de src/corrida_4b_kaggle.py).
ETAPAS_ANTES_DE_ENTRENAR = ("gpu", "datos", "instalacion", "imports", "descarga", "carga", "tokens_y_plantilla", "lora")
ETAPAS_DESPUES_DE_ENTRENAR = ("adaptador", "inferencia", "cierre")
# Figura del 4B: la referencia conserva el color y el marcador que tiene en la figura del barrido; el 4B
# toma la posición 7 de la paleta categórica de referencia (violeta). El par magenta y violeta pasa
# validate_palette.js en modo claro (ΔE CVD 19.9 y normal 28.1); la magenta queda bajo 3:1 de contraste
# con la superficie, así que la leyenda escribe en texto el valor final de validación.
COLOR_4B = "#6250d6"
MARCADOR_4B = "o"


def _prefijo_modelo(repo: str) -> str:
    """'Qwen/Qwen3-4B' -> 'qwen3_4b' (como 'qwen3_0.6b' en los resultados del 0.6B)."""
    return repo.rsplit("/", 1)[-1].lower().replace("-", "_")


def _tasa(x: float) -> str:
    """2e-4 -> '2e-4', como en los nombres de las configuraciones."""
    mantisa, exponente = f"{x:.0e}".split("e")
    return f"{mantisa}e{int(exponente)}"


def cargar_corrida_unica(clave: str, manifest: Mapping, raiz: Path = REPO_RAIZ) -> SimpleNamespace:
    """Lo de `cargar_barrido` más la referencia (resultado, predicciones y adaptador) y los JSON de la tabla."""
    conf = BARRIDOS[clave]
    ent = cargar_barrido(clave, manifest, raiz)
    datos_resumen = (Path(raiz) / conf["resumen_barrido"]).read_bytes()
    resumen = json.loads(datos_resumen)
    nombre_ref = conf["referencia"]
    por_conf = resumen.get("resultados_por_configuracion", {})
    if nombre_ref not in por_conf:
        raise ErrorCorrida(f"El resumen del barrido no tiene el resultado de {nombre_ref}.")
    rutas = {conf["inicial"]: CORRIDAS[conf["inicial"]]["resultado"], **{n: i["ruta"] for n, i in por_conf.items()}}
    qwen = {}
    for n, ruta in rutas.items():
        datos = (Path(raiz) / ruta).read_bytes()
        qwen[n] = {"ruta": ruta, "sha256": sha256_bytes(datos), "resultado": json.loads(datos)}
    carpeta_ref = Path(raiz) / BARRIDOS[conf["barrido_referencia"]]["carpeta"] / nombre_ref
    adaptador_ref = carpeta_ref / CARPETA_ADAPTADOR / "adapter_model.safetensors"
    ent.referencia = nombre_ref
    ent.resumen_barrido = resumen
    ent.resumen_barrido_sha256 = sha256_bytes(datos_resumen)
    ent.resultados_qwen = qwen
    ent.sha256_baselines = {r: sha256_bytes((Path(raiz) / r).read_bytes()) for r in (RESULTADO_TFIDF, RESULTADO_DICCIONARIO)}
    ent.predicciones_referencia = (carpeta_ref / ARCHIVO_PREDICCIONES).read_bytes()
    ent.adaptador_referencia = {
        "ruta": adaptador_ref.relative_to(raiz).as_posix(),
        "bytes": adaptador_ref.stat().st_size,
        "sha256": sha256_archivo(adaptador_ref),
    }
    return ent


def revisar_referencia(corrida: Mapping, ref: Mapping, nombre_ref: str, resumen_barrido: Mapping,
                       adaptador_ref: Mapping) -> dict:
    """La referencia es la ganadora del barrido, con la misma receta salvo el modelo, y su salida está completa."""
    res_ref = ref["resultado"]
    corrida_ref = res_ref["configuracion"]["corrida"]
    distintas = sorted(k for k in set(corrida) | set(corrida_ref) if corrida.get(k) != corrida_ref.get(k))
    info = resumen_barrido.get("resultados_por_configuracion", {}).get(nombre_ref, {})
    return _etapa(
        {
            "referencia_es_la_ganadora_del_barrido": resumen_barrido.get("eleccion", {}).get("ganadora") == nombre_ref,
            "resultado_de_la_referencia_es_el_del_resumen": info.get("ruta") == ref["ruta"] and info.get("sha256") == ref["sha256"],
            "misma_receta_salvo_nombre_descripcion_y_modelo": set(distintas) <= {"nombre", "descripcion", "modelo"},
            "adaptador_de_la_referencia_igual_al_de_su_resultado": adaptador_ref["sha256"] == res_ref["modelo"]["adaptador_sha256"],
        },
        referencia=nombre_ref,
        claves_distintas=distintas,
        learning_rate=corrida["learning_rate"],
        rank=corrida["lora"]["r"],
        lora_alpha=corrida["lora"]["lora_alpha"],
    )


def contra_referencia(aciertos_ref: Mapping[str, list[bool]], aciertos: Mapping[str, list[bool]], res_ref: Mapping,
                      info_ref: Mapping, ruta_ref: str, nombre_ref: str, nombre: str) -> dict:
    """Diferencia pareada corrida − referencia, en clase y en sector, sobre las mismas filas."""
    m = res_ref["metricas"]
    return {
        "referencia": nombre_ref,
        "resultado_referencia": ruta_ref,
        "accuracy_referencia": m["regla5"]["accuracy"],
        "ic95_referencia": m["ic95_accuracy"],
        "signo": SIGNO_CONTRA_REFERENCIA.format(corrida=nombre, referencia=nombre_ref),
        "clase": diferencia_pareada_bootstrap(aciertos_ref["clase"], aciertos["clase"]),
        "sector": diferencia_pareada_bootstrap(aciertos_ref["sector"], aciertos["sector"]),
        "definicion": DEFINICION_PAREADA,
        "alineacion": {
            "filas": len(aciertos["clase"]),
            "orden": "int(id) de data/splits/val.jsonl",
            "comprobada_fila_por_fila": True,
            "predicciones_referencia_sha256": info_ref["predicciones_sha256"],
            "comprobaciones_referencia": dict(info_ref["comprobaciones"]),
            "metodo": (
                "Las dos listas de aciertos salen de alinear las predicciones de cada modelo con data/splits/val.jsonl "
                "(mismos ids que el val.jsonl del paquete, mismas etiquetas) en orden de int(id); en cada posición, el "
                f"acierto es el de la respuesta de esa fila contra su codigo_act. Las de {nombre_ref} salen de su "
                "predicciones.jsonl con el SHA-256 de su resultado y reproducen sus conteos e IC."
            ),
        },
    }


def _contra(estimado, real) -> dict:
    return {
        "estimado_s": None if estimado is None else round(estimado, 1),
        "real_s": None if real is None else round(real, 1),
        "real_entre_estimado": round(real / estimado, 3) if estimado and real is not None else None,
    }


def tiempos_contra_estimacion(b: Mapping, m: Mapping, nombre: str) -> dict:
    """Las partes de la estimación del kernel (`desglose_estimado_s` de barrido.json) contra los tiempos reales."""
    conf = b.get("configuracion", {})
    est = (conf.get("desglose_estimado_s") or {}).get(nombre)
    proceso = next((c for c in b.get("configuraciones", []) if c.get("nombre") == nombre), None)
    if not est or proceso is None:
        raise ErrorCorrida(f"barrido.json no trae la estimación o el proceso de {nombre}.")
    d = m.get("duraciones_s", {})
    ent = m.get("entrenamiento", {})
    validaciones = (ent.get("segundos_evaluacion_paso_0") or 0) + sum(e.get("s", 0) for e in ent.get("evaluaciones_s", []))
    real = {
        "antes_de_entrenar": sum(d.get(k, 0) for k in ETAPAS_ANTES_DE_ENTRENAR),
        "pasos_de_entrenamiento": d.get("entrenamiento", 0) - validaciones,
        "validaciones": validaciones,
        "adaptador_inferencia_y_cierre": sum(d.get(k, 0) for k in ETAPAS_DESPUES_DE_ENTRENAR),
    }
    if set(est) != set(real):
        raise ErrorCorrida(f"Las partes de la estimación de {nombre} no son las esperadas: {sorted(est)}.")
    orquestador = sum(v for k, v in (b.get("duraciones_s") or {}).items() if not k.startswith("configuracion_"))
    pasos = ent.get("pasos_hechos")
    stats = ent.get("estadisticas") or {}
    return {
        "partes": {k: _contra(est[k], real[k]) for k in real},
        "suma_de_las_partes": _contra(sum(est.values()), sum(real.values())),
        "proceso": _contra(proceso.get("estimado_s"), proceso.get("segundos")),
        "orquestador": _contra(bk.S_ORQUESTADOR, orquestador),
        "total": _contra(bk.S_ORQUESTADOR + proceso["estimado_s"], b.get("segundos_desde_arranque")),
        "s_por_paso": {
            "estimado": round(est["pasos_de_entrenamiento"] / pasos, 4) if pasos else None,
            "real": round(real["pasos_de_entrenamiento"] / pasos, 4) if pasos else None,
            "real_media_medida": stats.get("s_por_paso_media"),
            "real_mediana_medida": stats.get("s_por_paso_mediana"),
        },
        "razon_4b_06b_s_por_paso_usada": conf.get("razon_4b_06b_s_por_paso"),
        "definicion": (
            "Estimado: desglose_estimado_s de barrido.json (src/corrida_4b_kaggle.py: tiempos reales de 06b-inicial × la "
            "razón de s/paso 4B/0.6B de la prueba de humo). Real, de mediciones.json: antes de entrenar = etapas "
            f"{', '.join(ETAPAS_ANTES_DE_ENTRENAR)}; validaciones = la del paso 0 más las 12 de dentro del entrenamiento; "
            "pasos de entrenamiento = la etapa entrenamiento menos esas validaciones; adaptador, inferencia y cierre = "
            f"etapas {', '.join(ETAPAS_DESPUES_DE_ENTRENAR)}. Proceso: segundos del proceso en barrido.json. Orquestador: "
            "etapas propias de barrido.json (sin el proceso). Total: segundos_desde_arranque del orquestador."
        ),
    }


def horas_gpu_corrida_unica(tiempo: Mapping, obs: Mapping) -> dict:
    """Horas de GPU de la corrida, cotejadas con la cuota que descontó Kaggle."""
    con_cola = tiempo["arranque_del_contenedor_s"] > UMBRAL_COLA_S
    valor_s = tiempo["script_s"] if con_cola else tiempo["desde_last_run_time_s"]
    if con_cola:
        metodo = (
            "Duración del orquestador (inicio_utc de barrido.json más segundos_desde_arranque): el contenedor arrancó "
            f"más de {UMBRAL_COLA_S} s después del lastRunTime, así que esa marca incluye tiempo en cola."
        )
    else:
        metodo = (
            "Del lastRunTime de `kaggle kernels list --mine` (UTC) al fin del orquestador (inicio_utc de barrido.json más "
            "segundos_desde_arranque), con el arranque del contenedor."
        )
    return {
        "valor": round(valor_s / 3600, 3),
        "metodo": metodo + " Una sola configuración en una sesión GPU T4 x2 con una sola T4 en uso.",
        "script": tiempo["horas"]["script"],
        "desde_last_run_time": tiempo["horas"]["desde_last_run_time"],
        "arranque_del_contenedor_s": tiempo["arranque_del_contenedor_s"],
        "con_cola_en_el_last_run_time": con_cola,
        "cuota_kaggle": {"antes": obs["cuota_antes"], "despues": obs["cuota_despues"], "descontado_min": obs["cuota_descontada_min"]},
        "diferencia_con_la_cuota_min": {
            "valor": round(valor_s / 60 - obs["cuota_descontada_min"], 1),
            "script": round(tiempo["script_s"] / 60 - obs["cuota_descontada_min"], 1),
            "desde_last_run_time": round(tiempo["desde_last_run_time_s"] / 60 - obs["cuota_descontada_min"], 1),
            "nota": "La cuota la lee Samuel en minutos; la diferencia es medida − descontado.",
        },
    }


def fila_de_validacion(modelo: str, metricas: Mapping, fuente: Mapping, corrida: Mapping | None = None) -> dict:
    """Una fila de la tabla de validación de todos los modelos, con el JSON de results/ del que sale."""
    r5, c = metricas["regla5"], metricas["concentracion"]
    return {
        "modelo": modelo,
        "base": corrida["modelo"]["repo"] if corrida else None,
        "learning_rate": corrida["learning_rate"] if corrida else None,
        "rank": corrida["lora"]["r"] if corrida else None,
        "lora_alpha": corrida["lora"]["lora_alpha"] if corrida else None,
        "n": r5["total"],
        "aciertos": r5["aciertos"],
        "accuracy": r5["accuracy"],
        "ic95": metricas["ic95_accuracy"],
        "accuracy_sector": metricas["por_nivel"]["sector"]["accuracy"],
        "ic95_sector": metricas["por_nivel"]["sector"]["ic95"],
        "macro_f1_sector": metricas["macro_f1_sector"]["valor"],
        "pct_formato_valido": r5["pct_formato_valido"],
        "pct_codigos_inexistentes": r5["pct_codigos_inexistentes"],
        "clases_existentes_distintas_predichas": c["clases_existentes_distintas_predichas"],
        "codigo_valido_mas_predicho": c["codigo_valido_mas_predicho"],
        "pct_filas_con_ese_codigo": c["pct_filas_con_ese_codigo"],
        "fuente": dict(fuente),
    }


def tabla_de_validacion(ent: SimpleNamespace, corrida: Mapping, metricas: Mapping) -> list[dict]:
    """06b-inicial, las configuraciones del barrido, la corrida, TF-IDF + LR y el diccionario."""
    filas = [
        fila_de_validacion(n, q["resultado"]["metricas"], {"ruta": q["ruta"], "sha256": q["sha256"]},
                           q["resultado"]["configuracion"]["corrida"])
        for n, q in ent.resultados_qwen.items()
    ]
    filas.append(fila_de_validacion(corrida["nombre"], metricas, {"ruta": "este resultado"}, corrida))
    filas.append(fila_de_validacion(f"baseline_tfidf_lr (C = {ent.tfidf['detalle']['C']:g})", ent.tfidf["metricas"],
                                    {"ruta": RESULTADO_TFIDF, "sha256": ent.sha256_baselines[RESULTADO_TFIDF]}))
    filas.append(fila_de_validacion("baseline_diccionario", ent.diccionario["metricas"],
                                    {"ruta": RESULTADO_DICCIONARIO, "sha256": ent.sha256_baselines[RESULTADO_DICCIONARIO]}))
    return filas


def para_la_fase_4(m: Mapping, res_ref: Mapping, nombre: str, nombre_ref: str) -> dict:
    """Segundos por fila de la inferencia y descarga y carga del modelo, para estimar la GPU de la Fase 4."""
    inf, d = m.get("inferencia", {}), m.get("duraciones_s", {})
    inf_ref = res_ref["etapas"]["etapas"]["inferencia"]
    d_ref = res_ref["corrida_kaggle"]["tiempo"]["duraciones_etapas_s"]
    return {
        nombre: {
            "inferencia_s": inf.get("segundos"), "filas": inf.get("filas"), "s_por_fila": inf.get("s_por_fila"),
            "vram_pico_inferencia_mib": inf.get("vram_pico_mib"), "descarga_s": d.get("descarga"), "carga_s": d.get("carga"),
        },
        nombre_ref: {
            "inferencia_s": inf_ref.get("segundos"), "filas": res_ref["metricas"]["regla5"]["total"],
            "s_por_fila": inf_ref.get("s_por_fila"), "vram_pico_inferencia_mib": inf_ref.get("vram_pico_mib"),
            "descarga_s": d_ref.get("descarga"), "carga_s": d_ref.get("carga"),
        },
        "fuente": (
            f"mediciones.json de cada corrida: el de {nombre} en esta descarga y el de {nombre_ref} a través de su JSON "
            "de results/ (etapas.inferencia y corrida_kaggle.tiempo.duraciones_etapas_s)."
        ),
        "nota": (
            "Inferencia greedy de una fila a la vez con el adaptador cargado, en una T4, con la plantilla del "
            "clasificador (sin ejemplos en el prompt). La descarga depende de la red de Kaggle en esa sesión."
        ),
    }


def evaluar_corrida_unica(ent: SimpleNamespace, manifest: Mapping, contexto: Mapping) -> tuple[dict, dict]:
    """Verifica, revisa nombres, resume orquestador y etapas, mide, compara y arma el resultado.

    Devuelve el resultado y la revisión de privacidad completa (con líneas, solo para la terminal).
    PARADA ante cualquier problema que no sea una etapa con comprobaciones fallidas; esas las junta
    `problemas_de_la_corrida_unica` y detienen `main`.
    """
    b = ent.barrido_json
    corridas = [dict(c) for c in ent.espec.corridas]
    if len(corridas) != 1:
        raise ErrorCorrida(f"Una corrida única lleva una sola configuración; el kernel tiene {len(corridas)}.")
    corrida = corridas[0]
    n = corrida["nombre"]
    procesos = {c.get("nombre"): c for c in b.get("configuraciones", [])}
    completa = procesos.get(n, {}).get("estado") == "completa"

    descarga = verificar_descarga_barrido(
        ent.inventario, b, ent.mediciones, ck.hashes_del_manifest(manifest), ent.conf["log"], [n] if completa else []
    )
    filas_ref = ent.train + ent.val_paquete
    ref = referencias_de_nombres(filas_ref)
    idx = indice_de_nombres(ref)
    jsons = {bk.ARCHIVO_BARRIDO: b, **{f"{k}/{ARCHIVO_MEDICIONES}": v for k, v in ent.mediciones.items() if v is not None}}
    tapados = {k: len(v.get("tapados", [])) for k, v in ent.mediciones.items() if v is not None}
    privacidad = revisar_privacidad_barrido(ent.textos, jsons, idx, len(filas_ref), contexto["texto_kernel"], tapados)
    if privacidad["posible_fuga"]:
        raise ErrorCorrida(f"Posible fuga de nombres en {privacidad['archivos_con_posible_fuga']}; decide Samuel.")
    if not completa:
        raise ErrorCorrida(f"{n} quedó incompleta (estado {procesos.get(n, {}).get('estado')}); decide Samuel.")

    archivos = manifest[SECCION][ENTRADA_PAQUETE]["archivos"]
    esperados = {a: archivos[a]["sha256"] for a in ARCHIVOS_ENTRADA}
    orquestador = resumen_orquestador(
        b, corridas=corridas, esperados=esperados, fuentes_actuales=contexto["fuentes_actuales_sha256"],
        kernel_igual_al_generado=contexto["kernel_igual_al_generado"],
    )
    m = ent.mediciones[n]
    log = ent.textos.get(f"{bk.CARPETA_LOGS}/{n}.log")
    if log is None:
        raise ErrorCorrida(f"Falta el log de {n}.")
    prefijo = f"{n}/"
    etapas = resumen_etapas(
        m, corrida=corrida, esperados=esperados, fuentes_actuales=contexto["fuentes_actuales_sha256"],
        kernel_igual_al_generado=contexto["kernel_igual_al_generado"], filas_train=len(ent.train),
        filas_val=len(ent.val_paquete), adapter_config=ent.adapter_configs[n],
        archivos_bajados={a["ruta"][len(prefijo):] for a in ent.inventario if a["ruta"].startswith(prefijo)},
        llaves=set(ref["varias"] | ref["una"]), texto_del_log=log,
        dir_trabajo=(DIR_TRABAJO / n).as_posix(),  # el orquestador le da a cada proceso su propia carpeta
    )
    res_ini = ent.resultado_inicial
    etapas["etapas"]["barrido"] = revisar_configuracion_del_barrido(m, res_ini["para_el_barrido"]["versiones_entorno"])
    nombre_ref = ent.referencia
    q_ref = ent.resultados_qwen[nombre_ref]
    res_ref = q_ref["resultado"]
    etapas["etapas"]["referencia"] = revisar_referencia(corrida, q_ref, nombre_ref, ent.resumen_barrido, ent.adaptador_referencia)
    etapas["todas_ok"] = all(x["estado"] == "ok" for x in etapas["etapas"].values())

    predicciones = leer_predicciones(ent.textos[f"{n}/{ARCHIVO_PREDICCIONES}"].encode("utf-8"))
    filas = alinear(predicciones, ent.val_splits, ent.val_paquete)
    metricas = evaluar_validacion(predicciones, filas, ent.codigos)
    aciertos = aciertos_alineados(predicciones, filas, ent.codigos)
    if intervalo_bootstrap(aciertos["clase"]) != metricas["ic95_accuracy"] or sum(aciertos["clase"]) != metricas["regla5"]["aciertos"]:
        raise ErrorCorrida(f"{n}: los aciertos alineados no reproducen las métricas del evaluador.")
    comprobacion_tfidf = comprobar_registro_tfidf(ent.registro_tfidf, ent.tfidf, filas)
    comp = comparacion(aciertos, metricas, ent.tfidf, ent.diccionario, ent.registro_tfidf, comprobacion_tfidf,
                       ent.ruta_registro_tfidf, ent.sha_registro_tfidf)
    aciertos_ref, info_ref = aciertos_de_la_inicial(
        ent.predicciones_referencia, res_ref, ent.val_splits, ent.val_paquete, ent.codigos,
        ruta_pred=f"{nombre_ref}/{ARCHIVO_PREDICCIONES}", etiqueta=nombre_ref,
    )
    contra = contra_referencia(aciertos_ref, aciertos, res_ref, info_ref, q_ref["ruta"], nombre_ref, n)

    obs = ent.conf["observaciones"]
    tiempo = duracion_barrido(b, obs["last_run_time_cli"])
    horas = horas_gpu_corrida_unica(tiempo, obs)
    tiempos = tiempos_contra_estimacion(b, m, n)
    lora = corrida["lora"]
    advertencias = [
        f"La configuración (tasa {_tasa(corrida['learning_rate'])}, rank {lora['r']}, alpha {lora['lora_alpha']}) se eligió en el "
        f"barrido del 0.6B y se transfirió a {n} sin barrido propio: puede no ser la mejor para este modelo.",
        f"La accuracy en validación de {nombre_ref} es algo optimista: se eligió con la misma validación con la que se mide.",
        f"La de {n} no lo es por elección: es una sola corrida, no se eligió entre varias con su validación (la "
        "configuración sí salió de esta validación, pero en el 0.6B).",
        f"Por eso la pareada {n} − {nombre_ref} queda, si acaso, sesgada en contra de {n}.",
    ]
    advertencias += [
        f"{p} es {v['instalada']}; en 06b-inicial era {v['06b_inicial']} (va con la imagen de Docker, no se fija con pip)."
        for p, v in etapas["etapas"]["barrido"]["advertencias_no_fijables"].items()
    ]
    inv = {a["ruta"]: a for a in ent.inventario}
    propio = inv.get(f"{n}/{CARPETA_ADAPTADOR}/adapter_model.safetensors", {})
    adaptadores = {
        n: {"ruta": f"{ent.conf['carpeta']}/{n}/{CARPETA_ADAPTADOR}/adapter_model.safetensors", "bytes": propio.get("bytes"),
            "sha256": propio.get("sha256"), "igual_al_de_barrido_json_y_mediciones": True},
        nombre_ref: {**ent.adaptador_referencia,
                     "igual_al_de_su_resultado": ent.adaptador_referencia["sha256"] == res_ref["modelo"]["adaptador_sha256"]},
        "nota": (
            f"El de {n} coincide con barrido.json y con su mediciones.json (la descarga se verificó); el de {nombre_ref} se "
            "volvió a hashear y se comparó con su resultado. Los adaptadores solo se hashean: nunca se cargan."
        ),
    }
    curvas_resumen = list(ent.resumen_barrido.get("curvas", {}))
    resultado = armar_resultado_corrida_unica(
        manifest=manifest, contexto=contexto, ent=ent, corrida=corrida, m=m, proceso=procesos[n], descarga=descarga,
        privacidad=privacidad, orquestador=orquestador, etapas=etapas,
        metricas={k: v for k, v in metricas.items() if k != "definiciones"},
        concentracion_extra=_concentracion_extra(predicciones, filas, metricas), comparacion=comp, contra=contra,
        tiempo=tiempo, horas=horas, tiempos=tiempos, tabla=tabla_de_validacion(ent, corrida, metricas),
        adaptadores=adaptadores, fase4=para_la_fase_4(m, res_ref, n, nombre_ref),
        curvas={
            "validacion": etapas["etapas"]["entrenamiento"]["perdida_validacion"],
            "train_por_ventana": curva_de_train_agregada(m.get("entrenamiento", {}).get("curvas", {}).get("train", [])),
            "ventana_pasos": VENTANA_CURVA,
        },
        curvas_ref={
            "nombre": nombre_ref,
            "posicion_en_la_figura_del_barrido": curvas_resumen.index(nombre_ref) if nombre_ref in curvas_resumen else None,
            "validacion": res_ref["curvas"]["validacion"],
            "train_por_ventana": res_ref["curvas"]["train_por_ventana"],
        },
        advertencias=advertencias,
    )
    candado_resultado(resultado, ref)
    return resultado, privacidad


def armar_resultado_corrida_unica(*, manifest, contexto, ent, corrida, m, proceso, descarga, privacidad, orquestador, etapas,
                                  metricas, concentracion_extra, comparacion, contra, tiempo, horas, tiempos, tabla,
                                  adaptadores, fase4, curvas, curvas_ref, advertencias) -> dict:
    """JSON de results/ de la corrida única (regla 4); sin nombres, llaves, ids ni respuestas crudas."""
    nombre = corrida["nombre"]
    obs = ent.conf["observaciones"]
    b = ent.barrido_json
    archivos = manifest[SECCION][ENTRADA_PAQUETE]["archivos"]
    sp = manifest["splits"]
    q_ref = ent.resultados_qwen[ent.referencia]
    return {
        "fecha": contexto["fecha"],
        "experimento": f"fase2_paso13_{nombre}",
        "commit": contexto["commit"],
        "arbol_limpio": contexto["arbol_limpio"],
        "modelo": {
            "nombre": f"{_prefijo_modelo(corrida['modelo']['repo'])}_qlora_{nombre}",
            "repo": corrida["modelo"]["repo"],
            "revision": corrida["modelo"]["revision"],
            "metodo": "QLoRA (NF4 de bitsandbytes) con Unsloth; enable_thinking=False; responde solo el código de 6 dígitos",
            "adaptador_sha256": adaptadores[nombre]["sha256"],
        },
        "configuracion": {
            "corrida": dict(corrida),
            "del_kernel": m.get("configuracion", {}),
            "kernel": f"{ent.espec.carpeta_relativa}/{ent.espec.archivo_kernel}",
            "kernel_generador": "src/construir_kernel.py",
            "commit_kernel": contexto["commit_kernel"],
            "kernel_kaggle": {"id": ent.espec.id_kernel, "version": obs["version_kaggle"]},
            "imagen_docker": obs["imagen_docker"],
            "tiempo_kaggle_s": obs["tiempo_kaggle_s"],
            "orquestador": "src/barrido_kaggle.py con una sola configuración (src/corrida_4b_kaggle.py)",
            "referencia": {
                "nombre": ent.referencia,
                "resultado": q_ref["ruta"],
                "sha256": q_ref["sha256"],
                "resumen_del_barrido": {"ruta": ent.conf["resumen_barrido"], "sha256": ent.resumen_barrido_sha256},
            },
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
        "horas_gpu": horas,
        "corrida_kaggle": {
            "estado_cli": obs["estado_cli"],
            "lanzamiento": obs["lanzamiento"],
            "fin": obs["fin"],
            "last_run_time_cli": obs["last_run_time_cli"],
            "tiempo": tiempo,
            "proceso": {k: proceso.get(k) for k in ("estado", "codigo_de_salida", "matado_por_tiempo", "estimado_s", "tiempo_maximo_s",
                                                     "transcurrido_al_empezar_s", "segundos", "memoria_gpu0_al_salir_mib")},
            "estado_orquestador": b.get("estado"),
            "fallas_orquestador": len(b.get("fallas", [])),
            "estado_kernel": m.get("estado"),
            "fallas": len(m.get("fallas", [])),
            "duraciones_etapas_s": m.get("duraciones_s"),
            "archivos_bajados": [dict(a) for a in ent.inventario],
            "descarga": descarga,
        },
        "tiempos_contra_estimacion": tiempos,
        "privacidad": privacidad_para_resultado(privacidad),
        "orquestador": orquestador,
        "etapas": etapas,
        "metricas": metricas,
        "concentracion_extra": concentracion_extra,
        "definiciones": {"pareada": DEFINICION_PAREADA, **DEFINICIONES},
        "comparacion": comparacion,
        "contra_referencia": contra,
        "tabla_validacion": tabla,
        "adaptadores": adaptadores,
        "para_la_fase_4": fase4,
        "curvas": curvas,
        "curvas_referencia": curvas_ref,
        "advertencias": advertencias,
    }


def problemas_de_la_corrida_unica(resultado: Mapping) -> list[str]:
    """Comprobaciones que fallaron (en el orquestador o en una etapa); cada una es PARADA."""
    res = [f"orquestador: {k}" for k, v in resultado["orquestador"]["comprobaciones"].items() if not v]
    for etapa, e in resultado["etapas"]["etapas"].items():
        res += [f"{etapa}: {k}" for k, v in e.get("comprobaciones", {}).items() if not v]
    return res


def figura_corrida_unica(resultado: Mapping, ruta: Path) -> None:
    """Curvas de pérdida de la corrida junto a las de la referencia, desde el JSON de results/."""
    corrida = resultado["configuracion"]["corrida"]
    cr = resultado["curvas_referencia"]
    i = cr["posicion_en_la_figura_del_barrido"]
    if i is None:
        raise ErrorCorrida("La referencia no tiene posición en la figura del barrido.")
    lora = corrida["lora"]
    series = [
        (corrida["nombre"], resultado["curvas"], COLOR_4B, MARCADOR_4B),
        (cr["nombre"], cr, SERIES_BARRIDO[i], MARCADORES_BARRIDO[i]),
    ]
    _figura_dos_paneles(
        series,
        corrida=corrida,
        titulo="Curvas de pérdida: Qwen3-4B frente a Qwen3-0.6B",
        subtitulo=(
            f"Misma receta: tasa {_tasa(corrida['learning_rate'])}, rank {lora['r']}, alpha {lora['lora_alpha']}, lote "
            f"{corrida['lote']}, {corrida['epocas']} época; validación cada {corrida['eval_cada_pasos']} pasos y al final."
        ),
        ruta=ruta,
        columnas_leyenda=2,
    )


def escribir_corrida_unica(resultado: Mapping, conf: Mapping, raiz: Path = REPO_RAIZ) -> list[str]:
    """Escribe el JSON de results/ y la figura (hecha desde ese JSON); devuelve las rutas escritas."""
    ruta = _ruta(conf["resultado"], resultado["fecha"])
    (Path(raiz) / ruta).write_bytes(serializar(resultado))
    ruta_fig = _ruta(conf["figura"], resultado["fecha"])
    figura_corrida_unica(json.loads((Path(raiz) / ruta).read_bytes()), Path(raiz) / ruta_fig)
    return [ruta, ruta_fig]


def imprimir_corrida_unica(res: Mapping, privacidad: Mapping) -> None:
    d = res["corrida_kaggle"]
    x = d["descarga"]
    print(f"Archivos bajados: {len(d['archivos_bajados'])}; coinciden con barrido.json {x['coinciden_con_kaggle']} de "
          f"{x['con_sha256_de_kaggle']}; sin hash de Kaggle: {x['sin_sha256_de_kaggle']}; con hash del manifest: "
          f"{x['con_sha256_del_manifest']}; por configuración: {x['por_configuracion']}")
    p = privacidad
    print(f"Privacidad: posible fuga = {p['posible_fuga']}; tapados por el kernel = {p['tapados_por_el_kernel']}; "
          f"{p['nombres_buscados']:,} nombres, {p['llaves_varias_palabras_buscadas']:,} llaves de varias palabras y "
          f"{p['llaves_una_palabra_buscadas']:,} de una")
    for nombre, r in p["archivos"].items():
        c = r["clasificacion"]
        print(f"  {nombre}: crudos {r['nombres_crudos']} (solo dígitos {c['nombres_crudos_solo_digitos']}, cortos dentro de un "
              f"token {c['nombres_crudos_cortos_solo_dentro_de_un_token']}, posible fuga {c['nombres_crudos_posible_fuga']}), "
              f"varias palabras {r['llaves_varias_palabras']} (solo dígitos {c['llaves_varias_palabras_solo_digitos']}), "
              f"una palabra {r['llaves_una_palabra']}")
    o = res["orquestador"]
    malas = [k for k, v in o["comprobaciones"].items() if not v]
    print(f"Orquestador: {o['estado']}" + (f" (fallan: {malas})" if malas else "") + f"; advertencias: {o['advertencias']}; "
          f"no fijables: {o['no_fijables']}; controlador: {o['controlador_nvidia']}")
    for c in o["configuraciones"]:
        print(f"  {c['nombre']}: {c['estado']}, {c['segundos']} s (estimado {c['estimado_s']} s, máximo {c['tiempo_maximo_s']} s), "
              f"salida {c['codigo_de_salida']}, matado {c['matado_por_tiempo']}, GPU 0 al salir {c['memoria_gpu0_al_salir_mib']} MiB")
    print("Etapas:")
    for nombre, e in res["etapas"]["etapas"].items():
        malas = [k for k, v in e.get("comprobaciones", {}).items() if not v]
        print(f"  {nombre}: {e['estado']}" + (f" (fallan: {malas})" if malas else ""))
    et = res["etapas"]["etapas"]
    en = et["entrenamiento"]
    print(f"  GPU: {et['gpu']['gpus_en_la_sesion']}; modelo: revisión {res['modelo']['revision'][:8]}…, capas de 4 bits "
          f"{et['modelo']['capas_linear4bit']}, LoRA {et['modelo']['lora']}")
    print(f"  Entrenamiento: {en['pasos']} pasos, warmup {en['pasos_warmup']}, orden {en['orden_train']['sha256_orden'][:8]}…, "
          f"VRAM pico {en['vram_pico_mib']}, estadísticas {en['estadisticas']}")
    print(f"  Validación: {[(v['paso'], v['eval_loss'], v['fuente']) for v in en['perdida_validacion']]}")
    print(f"  Inferencia: {et['inferencia']['segundos']} s, {et['inferencia']['s_por_fila']} s/fila, terminación "
          f"{et['inferencia']['terminacion']}, fichas máx {et['inferencia']['fichas_generadas_max']}")
    mt = res["metricas"]
    r5 = mt["regla5"]
    print(f"Regla 5 (n = {r5['total']}): {r5['conteos']}; aciertos {r5['aciertos']}, accuracy {r5['accuracy']} "
          f"(IC 95 % {mt['ic95_accuracy']['inferior']}-{mt['ic95_accuracy']['superior']})")
    print("Por nivel: " + ", ".join(f"{k} {v['accuracy']}" for k, v in mt["por_nivel"].items())
          + f"; macro-F1 de sector {mt['macro_f1_sector']['valor']}")
    for marca, g in mt["subgrupos"].items():
        print(f"  {marca}: con {g['con']['accuracy']} ({g['con']['filas']} filas), sin {g['sin']['accuracy']} ({g['sin']['filas']} filas)")
    c, ce = mt["concentracion"], res["concentracion_extra"]
    print(f"Concentración: {c['clases_existentes_distintas_predichas']} clases existentes distintas; {c['codigo_valido_mas_predicho']} en "
          f"{c['filas_con_ese_codigo']} filas ({c['pct_filas_con_ese_codigo']} %; en validación {ce['filas_de_validacion_con_ese_codigo']}); "
          f"respuestas distintas {ce['respuestas_distintas_tras_limpiar']}")
    print("Tabla de validación:")
    for f in res["tabla_validacion"]:
        print(f"  {f['modelo']}: {f['aciertos']}/{f['n']} = {f['accuracy']} ({f['ic95']['inferior']}-{f['ic95']['superior']}), sector "
              f"{f['accuracy_sector']} ({f['ic95_sector']['inferior']}-{f['ic95_sector']['superior']}), macro-F1 sector "
              f"{f['macro_f1_sector']}, clases {f['clases_existentes_distintas_predichas']}, más predicho {f['codigo_valido_mas_predicho']} "
              f"({f['pct_filas_con_ese_codigo']} %)")
    for titulo, dp in (("contra la referencia", res["contra_referencia"]),
                       ("contra TF-IDF", res["comparacion"]["diferencia_pareada_contra_tfidf"])):
        for nivel in ("clase", "sector"):
            v = dp[nivel]
            print(f"  pareada {titulo}, {nivel} ({dp['signo']}): {v['diferencia']:+.3f} (IC 95 % {v['ic95']['inferior']:+.3f} a "
                  f"{v['ic95']['superior']:+.3f}); solo A {v['filas_solo_a_acierta']}, solo B {v['filas_solo_b_acierta']}")
    t = res["tiempos_contra_estimacion"]
    print("Tiempos (estimado → real, razón):")
    for k, v in [*t["partes"].items(), ("proceso", t["proceso"]), ("orquestador", t["orquestador"]), ("total", t["total"])]:
        print(f"  {k}: {v['estimado_s']} → {v['real_s']} s ({v['real_entre_estimado']})")
    print(f"  s/paso: {t['s_por_paso']}")
    h = res["horas_gpu"]
    print(f"Horas de GPU: {h['valor']} (script {h['script']}, desde lastRunTime {h['desde_last_run_time']}, arranque "
          f"{h['arranque_del_contenedor_s']} s); diferencia con la cuota (min): {h['diferencia_con_la_cuota_min']}")
    print(f"Adaptadores: { {k: v for k, v in res['adaptadores'].items() if k != 'nota'} }")
    print(f"Para la Fase 4: { {k: v for k, v in res['para_la_fase_4'].items() if k not in ('fuente', 'nota')} }")
    for a in res["advertencias"]:
        print(f"  ADVERTENCIA: {a}")


# --- Programa principal ---


def contexto_git(manifest: Mapping, corrida: str) -> dict:
    return contexto_de_kernel(manifest, CORRIDAS[corrida]["kernel"])


def contexto_de_kernel(manifest: Mapping, clave_kernel: str) -> dict:
    """Fecha, commit, árbol limpio y lo que se compara con el kernel generado hoy."""
    espec = ck.KERNELS[clave_kernel]
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
    cual = parser.add_mutually_exclusive_group(required=True)
    cual.add_argument("--corrida", choices=list(CORRIDAS))
    cual.add_argument("--barrido", choices=list(BARRIDOS))
    parser.add_argument("--escribir", action="store_true", help="escribe los JSON de results/ y la figura")
    parser.add_argument("--aceptar-incompletas", nargs="+", default=[], metavar="NOMBRE",
                        help="solo con --barrido y con la decisión de Samuel: configuraciones incompletas que se reportan y no entran a la regla")
    args = parser.parse_args(argv)
    if args.barrido:
        main_barrido(args)
        return
    if args.aceptar_incompletas:
        parser.error("--aceptar-incompletas solo va con --barrido.")

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


def main_barrido(args) -> None:
    from src.muestra_humo import ErrorMuestra
    from src.scian import leer_manifest
    from src.splits import esta_ignorado

    conf = BARRIDOS[args.barrido]
    if conf.get("tipo") == "corrida_unica":
        main_corrida_unica(args, conf)
        return
    carpeta = REPO_RAIZ / conf["carpeta"]
    manifest = leer_manifest()
    try:
        if not carpeta.is_dir():
            raise ErrorCorrida(f"No existe {conf['carpeta']}: falta bajar la salida del barrido.")
        if not esta_ignorado(carpeta) or not all(esta_ignorado(carpeta / a["ruta"]) for a in inventario(carpeta)):
            raise ErrorCorrida("La carpeta de la salida o alguno de sus archivos no está ignorado por git.")
        contexto = contexto_de_kernel(manifest, conf["kernel"])
        ent = cargar_barrido(args.barrido, manifest)
        resumen, resultados, privacidad = evaluar_barrido(ent, manifest, contexto, args.aceptar_incompletas)
    except (ErrorCorrida, ErrorMetricas, ErrorEvaluador, ErrorMuestra, AbortoHumo) as e:
        sys.exit(f"PARADA: {e}")
    imprimir_barrido(resumen, resultados, privacidad)
    print(f"commit {contexto['commit'][:7]} | árbol limpio: {contexto['arbol_limpio']} | kernel de {contexto['commit_kernel'][:7]}")
    problemas = problemas_del_barrido(resumen, resultados)
    if problemas:
        sys.exit(f"PARADA: comprobaciones que fallan: {problemas}")
    if args.escribir:
        if not contexto["arbol_limpio"]:
            sys.exit("PARADA: el árbol tiene cambios sin commit; el resultado debe registrar un commit limpio.")
        faltan = sorted(k for k, v in conf["observaciones"].items() if v is None)
        if faltan:
            sys.exit(f"PARADA: faltan observaciones del barrido (BARRIDOS): {faltan}.")
        for ruta in escribir_barrido(resumen, resultados, conf):
            p = REPO_RAIZ / ruta
            print(f"Escrito {ruta} ({p.stat().st_size:,} bytes, SHA-256 {sha256_archivo(p)}).")


def main_corrida_unica(args, conf: Mapping) -> None:
    from src.muestra_humo import ErrorMuestra
    from src.scian import leer_manifest
    from src.splits import esta_ignorado

    if args.aceptar_incompletas:
        sys.exit("PARADA: --aceptar-incompletas no va con una corrida única; si quedó incompleta, decide Samuel.")
    carpeta = REPO_RAIZ / conf["carpeta"]
    manifest = leer_manifest()
    try:
        if not carpeta.is_dir():
            raise ErrorCorrida(f"No existe {conf['carpeta']}: falta bajar la salida de la corrida.")
        if not esta_ignorado(carpeta) or not all(esta_ignorado(carpeta / a["ruta"]) for a in inventario(carpeta)):
            raise ErrorCorrida("La carpeta de la salida o alguno de sus archivos no está ignorado por git.")
        contexto = contexto_de_kernel(manifest, conf["kernel"])
        ent = cargar_corrida_unica(args.barrido, manifest)
        resultado, privacidad = evaluar_corrida_unica(ent, manifest, contexto)
    except (ErrorCorrida, ErrorMetricas, ErrorEvaluador, ErrorMuestra, AbortoHumo) as e:
        sys.exit(f"PARADA: {e}")
    imprimir_corrida_unica(resultado, privacidad)
    print(f"commit {contexto['commit'][:7]} | árbol limpio: {contexto['arbol_limpio']} | kernel de {contexto['commit_kernel'][:7]}")
    problemas = problemas_de_la_corrida_unica(resultado)
    if problemas:
        sys.exit(f"PARADA: comprobaciones que fallan: {problemas}")
    if args.escribir:
        if not contexto["arbol_limpio"]:
            sys.exit("PARADA: el árbol tiene cambios sin commit; el resultado debe registrar un commit limpio.")
        faltan = sorted(k for k, v in conf["observaciones"].items() if v is None)
        if faltan:
            sys.exit(f"PARADA: faltan observaciones de la corrida (BARRIDOS): {faltan}.")
        for ruta in escribir_corrida_unica(resultado, conf):
            p = REPO_RAIZ / ruta
            print(f"Escrito {ruta} ({p.stat().st_size:,} bytes, SHA-256 {sha256_archivo(p)}).")


if __name__ == "__main__":
    main()
