"""Métrica tolerante del clasificador (Fase 4, paso 17): solo secundaria y reportada aparte.

Definición del pre-registro (`evals/fase4_preregistro.json`, metricas.tolerante; CAMBIO DE DISEÑO aprobado
por Samuel el 2026-11-24): sobre la respuesta cruda guardada (la salida completa antes del primer token de
fin) se busca la primera secuencia de exactamente 6 dígitos rodeada de no-dígitos o de las orillas:

    re.search(r"(?<![0-9])[0-9]{6}(?![0-9])", respuesta)

Si no hay ninguna, es formato inválido. Si la hay, es código existente o inexistente según el catálogo de
la versión fijada, y es acierto si es igual a codigo_act. Solo dígitos ASCII [0-9], como en la regla 5:
un dígito Unicode al lado cuenta como no-dígito.

Devuelve solo conteos, porcentajes, accuracy e IC: nunca respuestas, nombres ni ids. La regla 5 (la
métrica principal) sigue en `src/metricas_clasificador.py`, sin cambios.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence

from src.evaluador_clasificador import intervalo_bootstrap
from src.metricas_clasificador import (
    CATEGORIAS,
    CODIGO_EXISTENTE,
    CODIGO_INEXISTENTE,
    FORMATO_INVALIDO,
    ErrorMetricas,
    categoria,
    limpiar,
    verificar_ids,
)

RE_TOLERANTE = re.compile(r"(?<![0-9])[0-9]{6}(?![0-9])")
DECIMALES_PCT = 2
DECIMALES_ACCURACY = 6
DEFINICION = (
    "Métrica tolerante (solo secundaria): la primera secuencia de exactamente 6 dígitos ASCII de la respuesta cruda "
    "completa, rodeada de no-dígitos o de las orillas (re.search(r\"(?<![0-9])[0-9]{6}(?![0-9])\", respuesta)). Sin "
    "ninguna: formato inválido; con una: código existente o inexistente según el catálogo, y acierto si es igual a "
    "codigo_act. Porcentajes y accuracy sobre el total."
)


def extraer(respuesta: str) -> str | None:
    """El código que lee la métrica tolerante, o None si la respuesta no trae 6 dígitos aislados."""
    if not isinstance(respuesta, str):
        raise ErrorMetricas(f"La respuesta debe ser texto (str) y llegó {type(respuesta).__name__}.")
    m = RE_TOLERANTE.search(respuesta)
    return m.group(0) if m else None


def categoria_tolerante(respuesta: str, codigos: Collection[str]) -> str:
    codigo = extraer(respuesta)
    if codigo is None:
        return FORMATO_INVALIDO
    return CODIGO_EXISTENTE if codigo in codigos else CODIGO_INEXISTENTE


def aciertos_tolerantes(predicciones: Mapping[str, str], filas: Sequence[Mapping], codigos: Collection[str]) -> list[bool]:
    """Acierto tolerante de cada fila, en el orden de `filas` (cada una con id y codigo_act)."""
    res = []
    for f in filas:
        codigo = extraer(predicciones[f["id"]])
        res.append(codigo is not None and codigo in codigos and codigo == f["codigo_act"])
    return res


def evaluar_tolerante(predicciones: Mapping[str, str], filas: Sequence[Mapping], codigos: Collection[str]) -> dict:
    """Conteos de las 3 categorías, accuracy con su IC y cuánto cambia contra la regla 5; filas en orden de int(id)."""
    if not filas:
        raise ErrorMetricas("No hay filas.")
    verificar_ids(predicciones, [f["id"] for f in filas])
    filas = sorted(filas, key=lambda f: int(f["id"]))
    conteos = dict.fromkeys(CATEGORIAS, 0)
    for f in filas:
        conteos[categoria_tolerante(predicciones[f["id"]], codigos)] += 1
    aciertos = aciertos_tolerantes(predicciones, filas, codigos)
    estrictos = [categoria(predicciones[f["id"]], codigos) == CODIGO_EXISTENTE and limpiar(predicciones[f["id"]]) == f["codigo_act"]
                 for f in filas]
    if any(e and not t for e, t in zip(estrictos, aciertos)):  # no debería pasar: un acierto estricto es tolerante
        raise ErrorMetricas("Un acierto de la regla 5 no es acierto tolerante.")
    total = len(filas)
    k = sum(aciertos)
    return {
        "total": total,
        "conteos": conteos,
        "porcentajes": {c: round(100 * n / total, DECIMALES_PCT) for c, n in conteos.items()},
        "aciertos": k,
        "accuracy": round(k / total, DECIMALES_ACCURACY),
        "ic95": intervalo_bootstrap(aciertos),
        "aciertos_regla5": sum(estrictos),
        "filas_aciertan_solo_con_la_tolerante": sum(t and not e for e, t in zip(estrictos, aciertos)),
    }
