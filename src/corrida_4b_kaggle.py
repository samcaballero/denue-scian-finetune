"""Kernel de Kaggle de la corrida del clasificador con Qwen3-4B (Fase 2, paso 12).

Usa el orquestador del barrido (`src/barrido_kaggle.py`) con una sola configuración: la ganadora
del barrido del 0.6B (tasa y rank, con alpha = rank) sobre `Qwen/Qwen3-4B` en su revisión fijada.
Lo demás es la receta del barrido, sin cambios:
- datos: solo los dos archivos del dataset, con su SHA-256;
- los 20 paquetes fijados con pip y la revisión de versiones contra 06b-inicial (si una fijada no
  quedó instalada, aborta antes de entrenar; si cambia una no fijable, solo advierte);
- la imagen de Docker de 06b-inicial (en kernel-metadata.json);
- la configuración corre en su propio proceso, que llama a `src.clasificador_kaggle.main()` sin
  cambios: entrena, guarda el adaptador LoRA (etapa `adaptador`) y después infiere en validación;
- su log completo en /kaggle/working/logs/<nombre>.log, con la pérdida de validación del paso 0.

Lo propio del 4B: la estimación de tiempo (`estimacion_s`, con la razón de segundos por paso
4B/0.6B medida en la prueba de humo) y el tiempo de Kaggle (`TIEMPO_KAGGLE_S`, el -t de
`kaggle kernels push`). El registro del orquestador sigue llamándose `barrido.json`.

El orquestador mata el proceso de la configuración al pasar el presupuesto (el -t menos un margen
para cerrar) y sale con 0, así que Kaggle guarda la salida (con el adaptador, si el entrenamiento
terminó) aunque la inferencia no alcance a terminar. Se espera el mismo orden del train que en
06b-inicial (mismos datos, lote y `data_seed`); si no lo es, queda como falla y la corrida sigue.

Nunca importa torch ni lee nom_estab.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import sys
import time
import traceback
from collections.abc import Mapping, Sequence

import src.barrido_kaggle as bk
import src.clasificador_kaggle as kc
import src.humo_kaggle as hk

# --- Mediciones (segundos) ---

# Prueba de humo (results/2026-11-21_fase2_prueba-humo.json, `rendimiento`): media de segundos por paso
# con lote 32 en la misma T4, sin los 5 pasos de calentamiento.
S_PASO_06B_HUMO = 0.62874
S_PASO_4B_HUMO = 3.75792
RAZON_4B_06B = S_PASO_4B_HUMO / S_PASO_06B_HUMO  # ~5.98
# Descarga y carga: del 0.6B en 06b-inicial (duraciones por etapa de su resultado) y del 4B en la prueba.
S_DESCARGA_CARGA_06B = 14.926 + 6.237
S_DESCARGA_CARGA_4B = 63.6 + 11.8
# Las 13 validaciones de 06b-inicial: la del paso 0 y las 12 de dentro del entrenamiento (van en
# S_ENTRENAMIENTO_06B); solo para el desglose.
S_VALIDACIONES_06B = 10.93 + 74.803

# --- Tiempo de Kaggle ---

TIEMPO_KAGGLE_S = 39_600  # el -t de `kaggle kernels push`: 11 h, unas 1.5 veces el total estimado con rank 32
MARGEN_CIERRE_S = 900
PRESUPUESTO_S = TIEMPO_KAGGLE_S - MARGEN_CIERRE_S
LIMITE_SESION_KAGGLE_S = 12 * 3600  # https://www.kaggle.com/docs/notebooks

TASAS_PERMITIDAS = (1e-4, 2e-4)  # las del barrido del 0.6B
LOTE = 32


# --- Funciones puras (se prueban en la laptop) ---


def desglose_s(corrida: Mapping) -> dict:
    """Segundos estimados de la configuración por parte: tiempos reales de 06b-inicial × razón 4B/0.6B."""
    pasos = kc.pasos_por_epoca(bk.FILAS_TRAIN_06B, corrida["lote"], corrida["gradient_accumulation_steps"]) * corrida["epocas"]
    factor = bk.FACTOR_ENTRENAMIENTO_POR_RANK[corrida["lora"]["r"]]
    escala = RAZON_4B_06B * factor * pasos / bk.PASOS_06B
    return {
        "antes_de_entrenar": bk.S_ANTES_DE_ENTRENAR_06B - S_DESCARGA_CARGA_06B + S_DESCARGA_CARGA_4B,
        "pasos_de_entrenamiento": (bk.S_ENTRENAMIENTO_06B - S_VALIDACIONES_06B) * escala,
        "validaciones": S_VALIDACIONES_06B * escala,
        "adaptador_inferencia_y_cierre": bk.S_DESPUES_DE_ENTRENAR_06B * RAZON_4B_06B,
    }


def estimacion_s(corrida: Mapping) -> float:
    """Segundos estimados del proceso de la configuración (sin el orquestador)."""
    return round(sum(desglose_s(corrida).values()), 1)


def estimacion_total_s(corridas: Sequence[Mapping]) -> float:
    return round(bk.S_ORQUESTADOR + sum(estimacion_s(c) for c in corridas), 1)


def validar_corridas_4b(corridas: Sequence[Mapping]) -> list[dict]:
    """Una sola corrida del 4B, con alpha = rank, lote 32 y una tasa del barrido, que quepa en el presupuesto."""
    res = bk.validar_corridas(corridas)
    if len(res) != 1:
        raise bk.AbortoBarrido(f"La corrida del 4B lleva exactamente una configuración; llegaron {len(res)}.")
    c = res[0]
    problemas = []
    if dict(c["modelo"]) != hk.MODELO_4B:
        problemas.append("el modelo debe ser Qwen/Qwen3-4B en su revisión fijada")
    if c["lora"]["lora_alpha"] != c["lora"]["r"]:
        problemas.append("alpha debe ser igual al rank")
    if c["lote"] != LOTE or c["gradient_accumulation_steps"] != 1 or c["epocas"] != 1:
        problemas.append(f"lote {LOTE}, sin acumulación y una época, como el barrido")
    if c["learning_rate"] not in TASAS_PERMITIDAS:
        problemas.append(f"la tasa debe ser una del barrido {list(TASAS_PERMITIDAS)}")
    if TIEMPO_KAGGLE_S > LIMITE_SESION_KAGGLE_S:
        problemas.append("el tiempo de Kaggle pasa del límite de 12 h por sesión")
    if bk.S_ORQUESTADOR + bk.FACTOR_CANDADO * estimacion_s(c) > PRESUPUESTO_S:
        problemas.append("el candado de tiempo del orquestador la omitiría con este presupuesto")
    if problemas:
        raise bk.AbortoBarrido("Corrida del 4B inválida: " + "; ".join(problemas) + ".")
    return res


def dependencias_4b() -> bk.Dependencias:
    """Las del barrido en Kaggle, con la estimación y el presupuesto del 4B."""
    return dataclasses.replace(bk.dependencias_kaggle(), estimar=estimacion_s, presupuesto_s=PRESUPUESTO_S)


# --- Punto de entrada ---


def main(
    esperados: Mapping[str, str],
    t_arranque: float | None = None,
    fuentes: Mapping[str, str] | None = None,
    corridas: Sequence[Mapping] | None = None,
    dep: bk.Dependencias | None = None,
) -> None:
    """Punto de entrada en Kaggle. `esperados` y `corridas` los incrusta el generador."""
    t_arranque = time.time() if t_arranque is None else t_arranque
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(line_buffering=True)
    hk.informar = bk.informar
    corridas = validar_corridas_4b(corridas or [])
    if set(esperados) != set(kc.ARCHIVOS_ENTRADA):
        raise bk.AbortoBarrido(f"Los SHA-256 esperados deben ser exactamente los de {list(kc.ARCHIVOS_ENTRADA)}.")
    dep = dep or dependencias_4b()
    dep.dir_salida.mkdir(parents=True, exist_ok=True)
    dep.dir_trabajo.mkdir(parents=True, exist_ok=True)
    reg = hk.Registro(dep.dir_salida / bk.ARCHIVO_BARRIDO, t_arranque)
    reg.datos["descripcion"] = (
        "Corrida del clasificador con Qwen3-4B (Fase 2, paso 12): el orquestador del barrido con una sola configuración."
    )
    reg.datos["configuracion"] = {
        "corridas": [dict(c) for c in corridas],
        "paquetes_fijados": list(bk.PAQUETES_BARRIDO),
        "no_fijables": list(bk.NO_FIJABLES),
        "versiones_06b_inicial": bk.VERSIONES_06B_INICIAL,
        "controlador_06b_inicial": bk.CONTROLADOR_06B_INICIAL,
        "sha256_orden_06b_inicial": bk.SHA256_ORDEN_06B_INICIAL,
        "estimado_s": {c["nombre"]: dep.estimar(c) for c in corridas},
        "desglose_estimado_s": {c["nombre"]: {k: round(v, 1) for k, v in desglose_s(c).items()} for c in corridas},
        "razon_4b_06b_s_por_paso": round(RAZON_4B_06B, 4),
        "presupuesto_s": dep.presupuesto_s,
        "tiempo_kaggle_s": TIEMPO_KAGGLE_S,
        "factor_candado": bk.FACTOR_CANDADO,
        "factor_tiempo_maximo": bk.FACTOR_TIEMPO_MAXIMO,
        "memoria_libre_maxima_mib": bk.MEMORIA_LIBRE_MAXIMA_MIB,
    }
    reg.datos["fuentes_incrustadas_sha256"] = dict(fuentes or {})
    reg.datos["datos_sha256_esperados"] = dict(esperados)
    reg.escribir()
    try:
        bk.orquestar(reg, esperados, fuentes or {}, corridas, dep, t_arranque)
    except hk.AbortoHumo as e:
        reg.datos["estado"] = "abortado"
        reg.datos["aborto"] = str(e)
        reg.escribir()
        bk.informar(f"ABORTO: {e}")
        sys.exit(1)
    except Exception:
        reg.datos["estado"] = "error"
        reg.datos["error"] = traceback.format_exc()[-4000:]
        reg.escribir()
        print(reg.datos["error"], flush=True)
        sys.exit(1)
    reg.escribir()
    bk.informar(f"estado final: {reg.datos['estado']}; fallas: {len(reg.datos['fallas'])}")
