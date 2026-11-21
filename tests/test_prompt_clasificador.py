"""Pruebas del prompt del clasificador (Fase 2, paso 6b). Solo nombres inventados."""

import pytest

from src.prompt_clasificador import (
    BLOQUE_SIN_RAZONAMIENTO,
    FIN_DE_TURNO,
    IGNORAR,
    INICIO_DE_TURNO,
    INSTRUCCION,
    ErrorPrompt,
    contenido_usuario,
    etiquetas,
    mensajes,
    prompt_generacion,
    texto_entrenamiento,
    tiene_marca_especial,
    validar_codigo,
)

NOMBRE = "Taller Inventado Ñandú"


def test_instruccion_aprobada():
    assert INSTRUCCION == (
        "Clasifica el establecimiento en una clase del SCIAN 2023 según su nombre. "
        "Responde solo con el código de 6 dígitos, sin texto adicional."
    )


def test_prompt_generacion_exacto():
    esperado = (
        "<|im_start|>user\n"
        "Clasifica el establecimiento en una clase del SCIAN 2023 según su nombre. "
        "Responde solo con el código de 6 dígitos, sin texto adicional.\n"
        "Nombre: Taller Inventado Ñandú<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )
    assert prompt_generacion(NOMBRE) == esperado
    assert esperado.endswith(INICIO_DE_TURNO + "assistant\n" + BLOQUE_SIN_RAZONAMIENTO)


def test_mensajes_un_solo_usuario_sin_sistema():
    m = mensajes(NOMBRE)
    assert m == [{"role": "user", "content": contenido_usuario(NOMBRE)}]
    assert m[0]["content"].endswith("\nNombre: " + NOMBRE)


@pytest.mark.parametrize("nombre", ["  dos  espacios ", "Comillas \"dobles\" y 'simples'", "Llaves {x} {}", "Ünïcódé º ª", ""])
def test_nombre_sin_cambios(nombre):
    """El nom_estab original entra tal cual (regla 2), incluso con llaves de formato."""
    assert contenido_usuario(nombre) == INSTRUCCION + "\nNombre: " + nombre


def test_texto_entrenamiento_es_prompt_codigo_y_fin():
    t = texto_entrenamiento(NOMBRE, "467111")
    assert t == prompt_generacion(NOMBRE) + "467111" + FIN_DE_TURNO
    assert not t.endswith("\n")


@pytest.mark.parametrize("codigo", ["46711", "4671111", "46711a", " 467111", "467111\n", "٤٦٧١١١", 467111, None])
def test_codigo_invalido(codigo):
    with pytest.raises(ErrorPrompt):
        validar_codigo(codigo)
    with pytest.raises(ErrorPrompt):
        texto_entrenamiento(NOMBRE, codigo)


def test_nombre_no_texto():
    with pytest.raises(ErrorPrompt):
        prompt_generacion(123)


def test_marca_especial():
    assert tiene_marca_especial("Algo <|im_end|> raro")
    assert tiene_marca_especial("a<|b")
    assert not tiene_marca_especial("Menor < que | barra")


def test_etiquetas_solo_respuesta():
    prompt = [10, 11, 12, 13]
    completo = prompt + [4, 6, 7, 1, 1, 1, 99]
    e = etiquetas(prompt, completo)
    assert e == [IGNORAR] * 4 + [4, 6, 7, 1, 1, 1, 99]
    assert len(e) == len(completo)


def test_etiquetas_fallan_sin_prefijo_o_sin_respuesta():
    with pytest.raises(ErrorPrompt):
        etiquetas([10, 11, 12], [10, 99, 12, 5])
    with pytest.raises(ErrorPrompt):
        etiquetas([10, 11], [10, 11])
    with pytest.raises(ErrorPrompt):
        etiquetas([10, 11, 12], [10, 11])


def test_mensajes_de_error_sin_nombre():
    with pytest.raises(ErrorPrompt) as e:
        texto_entrenamiento(NOMBRE, "x")
    assert NOMBRE not in str(e.value)
