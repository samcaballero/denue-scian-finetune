"""Prompt del clasificador (tarea 1): una sola definición para Kaggle y la laptop.

La entrada es el `nom_estab` original (CLAUDE.md, regla 2) y la respuesta es solo el
código SCIAN de 6 dígitos (sección a). El modelo es Qwen3 en su versión instruct, siempre
con `enable_thinking=False`.

`prompt_generacion()` escribe a mano el texto que produce la plantilla de chat de Qwen3
con `add_generation_prompt=True` y `enable_thinking=False` para un solo mensaje de usuario:

    <|im_start|>user\\n{contenido}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n

El kernel de Kaggle comprueba, fila por fila, que la plantilla del tokenizador da
exactamente este texto (comprobación (a) del paso 6b). Así la laptop puede reproducir el
prompt sin transformers.

`texto_entrenamiento()` se construye a propósito como prompt de generación + código +
`<|im_end|>`. La plantilla oficial agrega un "\\n" después de `<|im_end|>`; ese "\\n" no
entra al texto de entrenamiento (comprobación (b), que solo lo documenta).

Este módulo solo usa la biblioteca estándar: el generador `src/construir_kernel.py` lo
copia tal cual dentro del kernel.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

INSTRUCCION = (
    "Clasifica el establecimiento en una clase del SCIAN 2023 según su nombre. "
    "Responde solo con el código de 6 dígitos, sin texto adicional."
)
PLANTILLA_USUARIO = INSTRUCCION + "\nNombre: {nom_estab}"

INICIO_DE_TURNO = "<|im_start|>"
FIN_DE_TURNO = "<|im_end|>"
BLOQUE_SIN_RAZONAMIENTO = "<think>\n\n</think>\n\n"
MARCA_TOKEN_ESPECIAL = "<|"  # inicio de los tokens especiales de Qwen (<|im_start|>, <|im_end|>, ...)

IGNORAR = -100  # etiqueta que la pérdida ignora (convención de transformers)

# Formato válido de la regla 5: seis dígitos ASCII (no `\d`, que acepta dígitos Unicode).
RE_CODIGO = re.compile(r"[0-9]{6}")


class ErrorPrompt(ValueError):
    """Entrada inválida para el prompt o para las etiquetas; el mensaje nunca incluye el nombre."""


def _validar_nombre(nom_estab: str) -> str:
    if not isinstance(nom_estab, str):
        raise ErrorPrompt(f"nom_estab debe ser texto (str) y llegó {type(nom_estab).__name__}.")
    return nom_estab


def contenido_usuario(nom_estab: str) -> str:
    """Contenido del único mensaje de usuario; el nombre va sin cambios."""
    return PLANTILLA_USUARIO.format(nom_estab=_validar_nombre(nom_estab))


def mensajes(nom_estab: str) -> list[dict[str, str]]:
    """Mensajes para `apply_chat_template`: un solo mensaje de usuario, sin mensaje de sistema."""
    return [{"role": "user", "content": contenido_usuario(nom_estab)}]


def prompt_generacion(nom_estab: str) -> str:
    """Prompt de generación de Qwen3 sin razonamiento, igual al de la plantilla oficial."""
    return (
        f"{INICIO_DE_TURNO}user\n{contenido_usuario(nom_estab)}{FIN_DE_TURNO}\n"
        f"{INICIO_DE_TURNO}assistant\n{BLOQUE_SIN_RAZONAMIENTO}"
    )


def validar_codigo(codigo: str) -> str:
    """El código de la respuesta: seis dígitos ASCII exactos (regla 5)."""
    if not isinstance(codigo, str) or not RE_CODIGO.fullmatch(codigo):
        raise ErrorPrompt("El código debe ser exactamente seis dígitos ASCII.")
    return codigo


def texto_entrenamiento(nom_estab: str, codigo: str) -> str:
    """Texto de entrenamiento: prompt de generación + código + fin de turno, sin "\\n" final."""
    return prompt_generacion(nom_estab) + validar_codigo(codigo) + FIN_DE_TURNO


def tiene_marca_especial(nom_estab: str) -> bool:
    """True si el nombre contiene "<|", que el tokenizador podría leer como token especial."""
    return MARCA_TOKEN_ESPECIAL in _validar_nombre(nom_estab)


def etiquetas(ids_prompt: Sequence[int], ids_completo: Sequence[int]) -> list[int]:
    """Etiquetas para entrenar solo sobre la respuesta.

    Los tokens del prompt llevan IGNORAR; los que siguen (código y fin de turno) copian su id.
    Falla si los tokens del prompt no son el inicio exacto de los del texto completo, o si no
    queda ningún token de respuesta.
    """
    p, c = list(ids_prompt), list(ids_completo)
    if len(c) <= len(p) or c[: len(p)] != p:
        raise ErrorPrompt("Los tokens del prompt no son el inicio de los del texto completo.")
    return [IGNORAR] * len(p) + c[len(p) :]
