"""Normalización de `nom_estab` para agrupar (CLAUDE.md, sección c, regla 2).

La llave normalizada solo sirve para agrupar filas (splits, baseline de
diccionario y hard set); los modelos Qwen reciben el `nom_estab` original.
Todo el proyecto usa `normalizar_nombre()`; no hay otra implementación.

Pasos, en este orden:
1. Forma NFKD: separa cada letra de su acento y cambia las variantes de
   compatibilidad por su carácter base (espacio no separable → espacio,
   º → o, ª → a, ² → 2).
2. Quitar las marcas combinantes (categoría Unicode M): Á → A, Ñ → N, Ü → U.
3. Mayúsculas con `str.upper()`, que no depende del locale.
4. Puntuación (variante C): el punto y el apóstrofo (' y ’) se borran; cualquier
   otro carácter que no sea letra (L), número (N) ni espacio se cambia por
   espacio. Eso incluye símbolos como + o ° y caracteres de control.
5. Espacios colapsados y recortados.

Una entrada vacía o de pura puntuación da "". La función es pura, determinista,
idempotente y solo usa `unicodedata` de la biblioteca estándar.
"""

from __future__ import annotations

import unicodedata

# Signos que se borran en cada variante de puntuación; los demás se cambian por
# espacio. None quiere decir "borrar todos". Las variantes A y B solo existen
# para reproducir la comparación del paso 3 (src/grupos_nombres.py).
VARIANTES: dict[str, frozenset[str] | None] = {
    "A": frozenset(),  # todo signo se cambia por espacio
    "B": None,  # todo signo se borra
    "C": frozenset(".'’"),  # punto y apóstrofo (' y ’) se borran; el resto, espacio
}
VARIANTE = "C"  # elegida por Samuel el 2026-11-20 (Fase 1, paso 3)


def _normalizar(nombre: str, se_borran: frozenset[str] | None) -> str:
    """Normalización de la regla 2 con una política de puntuación dada."""
    if not isinstance(nombre, str):
        raise TypeError(f"Se esperaba texto (str) y llegó {type(nombre).__name__}.")
    texto = unicodedata.normalize("NFKD", nombre)
    texto = "".join(c for c in texto if not unicodedata.category(c).startswith("M"))
    texto = texto.upper()
    partes = []
    for c in texto:
        if c.isspace() or unicodedata.category(c)[0] in "LN":
            partes.append(c)
        elif se_borran is None or c in se_borran:
            continue
        else:
            partes.append(" ")
    return " ".join("".join(partes).split())


def normalizar_nombre(nombre: str) -> str:
    """Llave normalizada de un `nom_estab` (CLAUDE.md, sección c, regla 2)."""
    return _normalizar(nombre, VARIANTES[VARIANTE])
