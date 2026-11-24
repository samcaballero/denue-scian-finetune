"""Regla de elección del barrido del clasificador (Fase 2, paso 12).

La regla se escribió antes de lanzar el barrido (docs/bitacora.md, paso 11, Parte B). La redactó el
chat de Claude del proyecto y Samuel la aprobó. El complemento lo propuso el chat a partir de los
dos casos que señaló Claude Code, y Samuel lo aprobó (commit aac0c7f):

1. Candidatas: 06b-inicial y las configuraciones del barrido. Solo cuentan las completas: todos los
   pasos de entrenamiento (5,855) e inferencia de las 1,000 filas de validación. Una configuración
   incompleta se reporta y no se repite antes del 4B.
2. Gana la mayor accuracy de clase en validación. Con el mismo número de filas en todas, es el
   mayor número de aciertos; se comparan enteros, no flotantes.
3. Empate exacto (los mismos aciertos): si la inicial está entre las empatadas, gana la inicial; si
   no, gana la de menor rank; si siguen empatadas, gana la de lr 2e-4.

Como se elige en la misma validación, la accuracy de la elegida es algo optimista.

Funciones puras: no leen archivos. `candidata_desde_resultado` arma la fila de una configuración
desde su JSON de `results/` ya cargado.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

PASOS_COMPLETOS = 5855  # 187,350 filas de train_final.jsonl con lote 32, una época
FILAS_VALIDACION = 1000
LR_DESEMPATE = 2e-4
CLAVES = ("nombre", "es_inicial", "rank", "learning_rate", "pasos_hechos", "filas_inferidas", "aciertos_clase", "n")

REGLA = (
    "Gana la mayor accuracy de clase en validación entre 06b-inicial y las configuraciones del barrido. Solo cuentan "
    "las completas (5,855 pasos de entrenamiento e inferencia de las 1,000 filas de validación); una incompleta se "
    "reporta y no se repite antes del 4B. En empate exacto (los mismos aciertos): si la inicial está entre las "
    "empatadas, gana la inicial; si no, la de menor rank; si siguen empatadas, la de lr 2e-4. Como se elige en la "
    "misma validación, la accuracy de la elegida es algo optimista. Regla escrita antes de lanzar el barrido "
    "(redactada por el chat de Claude del proyecto y aprobada por Samuel; complemento propuesto por el chat a partir "
    "de los casos que señaló Claude Code y aprobado por Samuel, commit aac0c7f)."
)

CLAUSULAS = {
    "mayor_accuracy": "una sola configuración con la mayor accuracy de clase",
    "empate_con_la_inicial": "empate exacto con la inicial entre las empatadas: gana la inicial",
    "empate_por_rank": "empate exacto sin la inicial: gana la de menor rank",
    "empate_por_lr": "empate exacto sin la inicial y con el mismo rank: gana la de lr 2e-4",
}


class ErrorEleccion(ValueError):
    """Las candidatas no permiten aplicar la regla."""


def _entero(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def completa(c: Mapping, pasos: int = PASOS_COMPLETOS, filas: int = FILAS_VALIDACION) -> bool:
    """¿Terminó todos los pasos y la inferencia de todas las filas de validación?"""
    return c["pasos_hechos"] == pasos and c["filas_inferidas"] == filas and c["n"] == filas and c["aciertos_clase"] is not None


def _validar(candidatas: Sequence[Mapping], filas: int) -> list[dict]:
    res = [dict(c) for c in candidatas]
    if not res:
        raise ErrorEleccion("No hay candidatas.")
    for c in res:
        if set(c) != set(CLAVES):
            raise ErrorEleccion(f"Cada candidata debe tener exactamente las claves {list(CLAVES)}.")
        if not isinstance(c["nombre"], str) or not c["nombre"]:
            raise ErrorEleccion("Cada candidata debe tener un nombre.")
        if not isinstance(c["es_inicial"], bool):
            raise ErrorEleccion(f"{c['nombre']}: es_inicial debe ser True o False.")
        if not _entero(c["rank"]) or c["rank"] < 1:
            raise ErrorEleccion(f"{c['nombre']}: el rank debe ser un entero positivo.")
        lr = c["learning_rate"]
        if not isinstance(lr, (int, float)) or isinstance(lr, bool) or not lr > 0:
            raise ErrorEleccion(f"{c['nombre']}: learning_rate debe ser un número positivo.")
        a = c["aciertos_clase"]
        if a is not None and (not _entero(a) or not 0 <= a <= filas):
            raise ErrorEleccion(f"{c['nombre']}: aciertos_clase debe ser un entero entre 0 y {filas}, o None.")
    nombres = [c["nombre"] for c in res]
    if len(set(nombres)) != len(nombres):
        raise ErrorEleccion("Las candidatas tienen nombres repetidos.")
    iniciales = [c["nombre"] for c in res if c["es_inicial"]]
    if len(iniciales) != 1:
        raise ErrorEleccion(f"Debe haber exactamente una configuración inicial; hay {len(iniciales)}.")
    return res


def _lr_de_desempate(c: Mapping) -> bool:
    return math.isclose(c["learning_rate"], LR_DESEMPATE, rel_tol=1e-12, abs_tol=0.0)


def elegir(candidatas: Sequence[Mapping], pasos: int = PASOS_COMPLETOS, filas: int = FILAS_VALIDACION) -> dict:
    """Aplica la regla de elección y devuelve la ganadora, la cláusula que decidió y el orden."""
    cands = _validar(candidatas, filas)
    completas = [c for c in cands if completa(c, pasos, filas)]
    excluidas = [c["nombre"] for c in cands if not completa(c, pasos, filas)]
    if not completas:
        raise ErrorEleccion("Ninguna candidata está completa.")
    maximo = max(c["aciertos_clase"] for c in completas)
    empatadas = [c for c in completas if c["aciertos_clase"] == maximo]
    if len(empatadas) == 1:
        ganadora, clausula = empatadas[0], "mayor_accuracy"
    elif any(c["es_inicial"] for c in empatadas):
        ganadora, clausula = next(c for c in empatadas if c["es_inicial"]), "empate_con_la_inicial"
    else:
        rank_minimo = min(c["rank"] for c in empatadas)
        quedan = [c for c in empatadas if c["rank"] == rank_minimo]
        if len(quedan) == 1:
            ganadora, clausula = quedan[0], "empate_por_rank"
        else:
            quedan = [c for c in quedan if _lr_de_desempate(c)]
            if len(quedan) != 1:
                raise ErrorEleccion(
                    f"Empate sin resolver entre {sorted(c['nombre'] for c in empatadas)} tras el rank y la tasa."
                )
            ganadora, clausula = quedan[0], "empate_por_lr"

    def clave_orden(c: Mapping) -> tuple:
        # El mismo criterio de la regla: aciertos, inicial, rank menor y lr 2e-4; el nombre solo fija el orden.
        return (-c["aciertos_clase"], not c["es_inicial"], c["rank"], not _lr_de_desempate(c), c["nombre"])

    orden = [
        {
            "nombre": c["nombre"],
            "es_inicial": c["es_inicial"],
            "rank": c["rank"],
            "learning_rate": c["learning_rate"],
            "completa": True,
            "aciertos_clase": c["aciertos_clase"],
            "n": c["n"],
            "accuracy": round(c["aciertos_clase"] / c["n"], 6),
        }
        for c in sorted(completas, key=clave_orden)
    ] + [
        {
            "nombre": c["nombre"],
            "es_inicial": c["es_inicial"],
            "rank": c["rank"],
            "learning_rate": c["learning_rate"],
            "completa": False,
            "pasos_hechos": c["pasos_hechos"],
            "filas_inferidas": c["filas_inferidas"],
        }
        for c in cands
        if not completa(c, pasos, filas)
    ]
    return {
        "ganadora": ganadora["nombre"],
        "ganadora_es_la_inicial": ganadora["es_inicial"],
        "clausula": clausula,
        "clausula_texto": CLAUSULAS[clausula],
        "aciertos_maximos": maximo,
        "accuracy_maxima": round(maximo / ganadora["n"], 6),
        "empatadas": sorted(c["nombre"] for c in empatadas),
        "excluidas_incompletas": excluidas,
        "orden": orden,
        "regla": REGLA,
        "advertencia": "Se elige en la misma validación con la que se mide: la accuracy de la elegida es algo optimista.",
    }


def candidata_desde_resultado(res: Mapping, es_inicial: bool) -> dict:
    """Fila de la regla desde el JSON de results/ de una configuración evaluada (con predicciones).

    `filas_inferidas` es el total de respuestas evaluadas: la evaluación ya comprobó que las
    predicciones son exactamente las filas de validación (mismos ids, sin repetidas).
    """
    corrida = res["configuracion"]["corrida"]
    r5 = res["metricas"]["regla5"]
    return {
        "nombre": corrida["nombre"],
        "es_inicial": es_inicial,
        "rank": corrida["lora"]["r"],
        "learning_rate": corrida["learning_rate"],
        "pasos_hechos": res["etapas"]["etapas"]["entrenamiento"]["pasos"],
        "filas_inferidas": r5["total"],
        "aciertos_clase": r5["aciertos"],
        "n": r5["total"],
    }
