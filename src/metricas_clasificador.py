"""Métricas del clasificador según la regla 5 de CLAUDE.md: la única implementación del proyecto.

Cada respuesta cae en una de tres categorías:

- formato inválido: tras quitar en las orillas solo el espacio ASCII, "\\r" y "\\n", la
  respuesta no es exactamente seis dígitos ASCII (`RE_CODIGO.fullmatch`, sin `\\d`). Un
  tabulador, un espacio no separable u otro espacio Unicode en las orillas deja la respuesta
  en formato inválido (precisión aprobada por Samuel el 2026-11-21);
- código inexistente: formato válido, pero el código no está en el catálogo oficial del SCIAN
  de la versión fijada (`data/scian_2023_clases.csv`);
- código existente: formato válido y código del catálogo.

Los porcentajes y la accuracy se calculan sobre el total de respuestas: el formato inválido y
el código inexistente cuentan como error. Los resultados guardan los conteos, no solo los
porcentajes. Los mensajes de error nunca incluyen respuestas, nombres ni ids.

Solo usa la biblioteca estándar y `src/prompt_clasificador.py`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Iterable, Mapping

from src.prompt_clasificador import RE_CODIGO

FORMATO_INVALIDO = "formato_invalido"
CODIGO_INEXISTENTE = "codigo_inexistente"
CODIGO_EXISTENTE = "codigo_existente"
CATEGORIAS = (FORMATO_INVALIDO, CODIGO_INEXISTENTE, CODIGO_EXISTENTE)

ORILLAS = " \r\n"  # lo único que se quita al inicio y al final (regla 5)
CAMPOS_PREDICCION = ("id", "respuesta")
RE_ID = re.compile(r"[0-9]+")
DECIMALES_PCT = 2
DECIMALES_ACCURACY = 6

DEFINICION = (
    "Regla 5: tras quitar al inicio y al final solo el espacio ASCII, \\r y \\n, la respuesta tiene "
    "formato válido si es exactamente seis dígitos ASCII ([0-9]{6}); es código existente si además "
    "está en el catálogo del SCIAN de la versión fijada. Porcentajes y accuracy sobre el total; "
    "formato inválido y código inexistente cuentan como error."
)


class ErrorMetricas(ValueError):
    """Entrada inválida para las métricas; el mensaje nunca incluye respuestas, nombres ni ids."""


def limpiar(respuesta: str) -> str:
    """La respuesta sin espacios ASCII, "\\r" ni "\\n" en las orillas (nada más se quita)."""
    if not isinstance(respuesta, str):
        raise ErrorMetricas(f"La respuesta debe ser texto (str) y llegó {type(respuesta).__name__}.")
    return respuesta.strip(ORILLAS)


def categoria(respuesta: str, codigos: Collection[str]) -> str:
    """Categoría de la regla 5 de una respuesta cruda del modelo."""
    limpia = limpiar(respuesta)
    if not RE_CODIGO.fullmatch(limpia):
        return FORMATO_INVALIDO
    return CODIGO_EXISTENTE if limpia in codigos else CODIGO_INEXISTENTE


def leer_predicciones(datos: bytes) -> dict[str, str]:
    """{id: respuesta} de un JSONL de predicciones (UTF-8 sin BOM, una línea por id).

    Cada línea es un objeto con las claves exactamente `id` y `respuesta`, en ese orden; el id
    es de dígitos ASCII y no se repite. Corta solo en "\\n": `splitlines()` también cortaría en
    U+2028, que JSON deja sin escapar.
    """
    if datos.startswith(b"\xef\xbb\xbf"):
        raise ErrorMetricas("Las predicciones tienen BOM.")
    texto = datos.decode("utf-8")
    if not texto.endswith("\n"):
        raise ErrorMetricas("Las predicciones están vacías o no terminan en salto de línea.")
    res: dict[str, str] = {}
    for i, linea in enumerate(texto[:-1].split("\n"), start=1):
        try:
            r = json.loads(linea)
        except json.JSONDecodeError as e:
            raise ErrorMetricas(f"Línea {i}: no es JSON válido.") from e
        if not isinstance(r, dict) or tuple(r) != CAMPOS_PREDICCION:
            raise ErrorMetricas(f"Línea {i}: las claves no son id y respuesta, en ese orden.")
        if not isinstance(r["id"], str) or not RE_ID.fullmatch(r["id"]):
            raise ErrorMetricas(f"Línea {i}: el id no es solo dígitos ASCII.")
        if not isinstance(r["respuesta"], str):
            raise ErrorMetricas(f"Línea {i}: la respuesta no es texto.")
        if r["id"] in res:
            raise ErrorMetricas(f"Línea {i}: id repetido.")
        res[r["id"]] = r["respuesta"]
    return res


def verificar_ids(predichos: Iterable[str], esperados: Iterable[str]) -> None:
    """Falla si los ids predichos no son exactamente los esperados (solo da conteos)."""
    p, e = set(predichos), set(esperados)
    if p != e:
        raise ErrorMetricas(f"Los ids no coinciden: faltan {len(e - p)} y sobran {len(p - e)}.")


def _pct(n: int, total: int) -> float:
    return round(100 * n / total, DECIMALES_PCT)


def evaluar(predicciones: Mapping[str, str], referencias: Mapping[str, str], codigos: Collection[str]) -> dict:
    """Conteos, porcentajes y accuracy de la regla 5, todo sobre el total de respuestas.

    `predicciones`: {id: respuesta cruda}; `referencias`: {id: código correcto}; `codigos`: los
    códigos del catálogo de la versión fijada. Falla si los ids no son idénticos o si alguna
    referencia no es un código del catálogo.
    """
    if not predicciones:
        raise ErrorMetricas("No hay predicciones.")
    verificar_ids(predicciones, referencias)
    malas = sum(1 for c in referencias.values() if not isinstance(c, str) or not RE_CODIGO.fullmatch(c) or c not in codigos)
    if malas:
        raise ErrorMetricas(f"{malas} referencias no son códigos del catálogo.")
    conteos = dict.fromkeys(CATEGORIAS, 0)
    aciertos = 0
    for id_, respuesta in predicciones.items():
        cat = categoria(respuesta, codigos)
        conteos[cat] += 1
        if cat == CODIGO_EXISTENTE and limpiar(respuesta) == referencias[id_]:
            aciertos += 1
    total = len(predicciones)
    if sum(conteos.values()) != total:  # no debería pasar: cada respuesta cae en una sola categoría
        raise ErrorMetricas("Las categorías no suman el total.")
    validos = conteos[CODIGO_INEXISTENTE] + conteos[CODIGO_EXISTENTE]
    return {
        "total": total,
        "conteos": conteos,
        "porcentajes": {c: _pct(n, total) for c, n in conteos.items()},
        "pct_formato_valido": _pct(validos, total),
        "pct_codigos_inexistentes": _pct(conteos[CODIGO_INEXISTENTE], total),
        "aciertos": aciertos,
        "errores": total - aciertos,
        "accuracy": round(aciertos / total, DECIMALES_ACCURACY),
        "pct_accuracy": _pct(aciertos, total),
    }
