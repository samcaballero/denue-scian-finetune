"""Pruebas del extractor del catálogo SCIAN.

Solo usan texto sintético con títulos inventados que imitan el formato de las
estructuras del INEGI; nunca leen los PDF reales ni el manifest del repo.
"""

import pytest

from src.scian import (
    ErrorCatalogo,
    codigos_en_tablas,
    escribir_catalogo,
    extraer_estructura,
    leer_catalogo,
    limpiar_titulo,
    validar_titulos,
)

PIE_ES = "INEGI. Estructura sintética del SCIAN. Versión en Español."
PIE_EN = "INEGI. Estructura sintética del SCIAN. Versión en Inglés."
PIE_BI = "INEGI. Estructura sintética del SCIAN. Versión Español-Inglés."

# Jerarquía sintética completa: 2 sectores (uno con rango), 2 subsectores, 2 ramas,
# 3 subramas y 4 clases. Marcas "T" pegadas, con espacio y tras paréntesis.
ENTRADAS_ES = [
    "11 Sector de ejemplo unoT",
    "111 Subsector de ejemploT",
    "1111 Rama de ejemplo T",
    "11111 Subrama de ejemplo (uno)T",
    "111110 Clase de ejemplo uno",
    "11112 Subrama de ejemplo dosT",
    "111121 Clase de ejemplo, con coma",
    "111122 Clase de ejemplo partida",
    "en dos líneas",
    "31-33 Sector con rangoT",
    "311 Subsector de rangoT",
    "3111 Rama de rangoT",
    "31111 Subrama de rangoT",
    "311110 Clase de rango",
]
TITULOS_ES = {
    "11": "Sector de ejemplo uno",
    "111": "Subsector de ejemplo",
    "1111": "Rama de ejemplo",
    "11111": "Subrama de ejemplo (uno)",
    "111110": "Clase de ejemplo uno",
    "11112": "Subrama de ejemplo dos",
    "111121": "Clase de ejemplo, con coma",
    "111122": "Clase de ejemplo partida en dos líneas",
    "31-33": "Sector con rango",
    "311": "Subsector de rango",
    "3111": "Rama de rango",
    "31111": "Subrama de rango",
    "311110": "Clase de rango",
}
ENTRADAS_EN = [
    "11 Example sector oneT",
    "111 Example subsectorT",
    "1111 Example industry group T",
    "11111 Example industry (one)T",
    "111110 Example class one",
    "11112 Example industry twoT",
    "111121 Example class, with comma",
    "111122 Example class split",
    "in two lines",
    "31-33 Range sectorT",
    "311 Range subsectorT",
    "3111 Range industry groupT",
    "31111 Range industryT",
    "311110 Range class",
]
CANDADO = {"clases": 4, "subramas": 3, "ramas": 2}


def pagina(lineas, pie, numero=None):
    """Una página como la devuelve texto_de_paginas: número arriba y pie al final."""
    cabeza = [str(numero)] if numero is not None else []
    return "\n".join([*cabeza, *lineas, pie])


def pdf_sintetico(es=ENTRADAS_ES, en=ENTRADAS_EN):
    """Inglés primero (como en el PDF de 2018), luego español y luego bilingüe."""
    bilingue = [f"{linea} / {otra}" for linea, otra in zip(es, en)]
    return [pagina(en, PIE_EN, 1), pagina(es, PIE_ES, 2), pagina(bilingue, PIE_BI, 3)]


# --- Limpieza del título y marca "T" ---


@pytest.mark.parametrize(
    "crudo, esperado, marcas",
    [
        ("Cultivo de ejemploT", "Cultivo de ejemplo", 1),
        ("Cultivo de ejemplo T", "Cultivo de ejemplo", 1),
        ("Operadores de ejemplo (sintéticos)T", "Operadores de ejemplo (sintéticos)", 1),
        ("Cultivo de maízT", "Cultivo de maíz", 1),
        # Marca repetida, como la de la subrama 23611 en el PDF de 2023.
        ("Edificación de ejemploT T", "Edificación de ejemplo", 2),
        ("Servicios de ejemplo NAT", "Servicios de ejemplo NAT", 0),
        ("  Título   con\tespacios  ", "Título con espacios", 0),
    ],
)
def test_limpiar_titulo(crudo, esperado, marcas):
    assert limpiar_titulo(crudo) == (esperado, marcas)


def test_marca_t_repetida_se_quita_y_se_registra():
    entradas = [
        e if e != "11112 Subrama de ejemplo dosT" else "11112 Subrama de ejemplo dosT T"
        for e in ENTRADAS_ES
    ]
    e = extraer_estructura([pagina(entradas, PIE_ES)], CANDADO)
    assert e.categorias == TITULOS_ES
    assert e.codigos_con_marca_t_repetida == ["11112"]
    assert e.marcas_t_quitadas["subramas"] == 3


# --- Secciones e idiomas ---


def test_ingles_primero_toma_titulos_en_espanol():
    e = extraer_estructura(pdf_sintetico(), CANDADO)
    assert e.categorias == TITULOS_ES
    assert e.paginas_espanol == 1
    # Cada clase aparece en las tres secciones, pero cuenta una sola vez.
    assert e.lineas_de_clase_todo_el_pdf == 12
    assert e.codigos_clase_todo_el_pdf == {"111110", "111121", "111122", "311110"}
    assert e.conteos() == {
        "sectores": 2, "subsectores": 2, "ramas": 2, "subramas": 3, "clases": 4,
    }
    assert e.marcas_t_quitadas == {
        "sectores": 2, "subsectores": 2, "ramas": 2, "subramas": 3, "clases": 0,
    }
    assert e.codigos_con_marca_t_repetida == []


def test_pagina_sin_encabezado_se_ignora():
    paginas = pdf_sintetico() + ["Página de notas sin encabezado de idioma\nTexto suelto"]
    e = extraer_estructura(paginas, CANDADO)
    assert e.categorias == TITULOS_ES


def test_sin_seccion_en_espanol_lanza_error():
    paginas = [pagina(ENTRADAS_EN, PIE_EN, 1)]
    with pytest.raises(ErrorCatalogo, match="Versión en Español"):
        extraer_estructura(paginas, None)


# --- Líneas: continuaciones, saltos de página, encabezados y portada ---


def test_titulo_partido_por_salto_de_pagina():
    corte = ENTRADAS_ES.index("en dos líneas")
    paginas = [
        pagina(ENTRADAS_ES[:corte], PIE_ES, 2),
        pagina(ENTRADAS_ES[corte:], PIE_ES, 3),
    ]
    e = extraer_estructura(paginas, CANDADO)
    assert e.categorias["111122"] == "Clase de ejemplo partida en dos líneas"


def test_pie_pegado_al_numero_de_pagina_no_es_un_sector():
    # En el PDF de 2018 el pie viene como "55 INEGI. ...", y 55 es un sector real.
    pie_2018 = "55 INEGI. Sistema sintético, SCIAN. INEGI. Estructura sintética. Versión en Español."
    e = extraer_estructura(["\n".join([*ENTRADAS_ES, pie_2018])], CANDADO)
    assert "55" not in e.categorias
    assert e.categorias == TITULOS_ES


def test_rango_de_sector_cierra_la_entrada_anterior():
    e = extraer_estructura(pdf_sintetico(), CANDADO)
    assert e.categorias["111122"] == "Clase de ejemplo partida en dos líneas"
    assert e.categorias["31-33"] == "Sector con rango"


def test_portada_antes_de_la_primera_entrada_se_ignora():
    portada = ["Instituto sintético", "Estructura de ejemplo", "Versión en Español"]
    paginas = ["\n".join(portada), pagina(ENTRADAS_ES, PIE_ES, 2)]
    e = extraer_estructura(paginas, CANDADO)
    assert e.lineas_antes_de_la_primera_entrada == 3
    assert e.categorias == TITULOS_ES


# --- Errores de estructura ---


def test_codigo_repetido_en_espanol_lanza_error():
    entradas = [*ENTRADAS_ES, "311110 Clase de rango otra vez"]
    with pytest.raises(ErrorCatalogo, match="repetidos"):
        extraer_estructura([pagina(entradas, PIE_ES)], None)


def test_clase_sin_subrama_lanza_error():
    entradas = [*ENTRADAS_ES, "311120 Clase sin subrama"]
    with pytest.raises(ErrorCatalogo, match="Sin padre: \\['311120'\\]"):
        extraer_estructura([pagina(entradas, PIE_ES)], None)


def test_subrama_sin_clases_lanza_error():
    entradas = [*ENTRADAS_ES, "31112 Subrama sin clasesT"]
    with pytest.raises(ErrorCatalogo, match="Sin hijos: \\['31112'\\]"):
        extraer_estructura([pagina(entradas, PIE_ES)], None)


def test_candado_distinto_lanza_error():
    with pytest.raises(ErrorCatalogo, match="clases: obtenido 4, esperado 5"):
        extraer_estructura(pdf_sintetico(), {**CANDADO, "clases": 5})


def test_clase_solo_en_otra_seccion_lanza_error():
    en = [*ENTRADAS_EN, "311190 Class only in English"]
    with pytest.raises(ErrorCatalogo, match="311190"):
        extraer_estructura(pdf_sintetico(en=en), None)


# --- Validación de títulos: el error lista código y título de cada caso ---


def test_validar_titulos_lista_codigo_y_titulo_de_cada_caso():
    categorias = {
        "111110": "Clase correcta",
        "111121": "Clase que termina en mayúsculaX",
        "111122": "Clase con ligadura ﬁnal",
        "111123": "",
    }
    with pytest.raises(ErrorCatalogo) as error:
        validar_titulos(categorias)
    mensaje = str(error.value)
    assert "3 títulos no pasan la validación" in mensaje
    assert "111121 | 'Clase que termina en mayúsculaX' | termina en mayúscula" in mensaje
    assert "111122 | 'Clase con ligadura ﬁnal' |" in mensaje and "U+FB01" in mensaje
    assert "111123 | '' | título vacío" in mensaje
    assert "111110" not in mensaje


def test_titulo_que_queda_en_mayuscula_detiene_la_extraccion():
    entradas = [e if e != "311110 Clase de rango" else "311110 Clase de rango ABC" for e in ENTRADAS_ES]
    with pytest.raises(ErrorCatalogo, match="311110 \\| 'Clase de rango ABC'"):
        extraer_estructura([pagina(entradas, PIE_ES)], None)


# --- Tablas comparativas ---


def test_codigos_en_tablas_solo_de_paginas_de_tablas():
    ejemplo = "Caso 1. Ejemplo\n999991 Clase de ejemplo 999992 Otra"
    tabla_1 = "TABLA COMPARATIVA I. SCIAN 2023-SCIAN 2018\n111121 Clase p 7/84 111122 Otra"
    tabla_2 = "TABLA COMPARATIVA II. SCIAN 2018-SCIAN 2023 (Continúa)\n311110 Clase\n1234567"
    assert codigos_en_tablas([ejemplo, tabla_1, tabla_2]) == {"111121", "111122", "311110"}


# --- Catálogo en CSV ---


def test_catalogo_csv_formato_y_ida_y_vuelta(tmp_path):
    clases = {"311110": "Clase de rango", "111121": "Clase de ejemplo, con coma", "111110": "Clase ñ"}
    ruta = tmp_path / "catalogo.csv"
    escribir_catalogo(clases, ruta)
    datos = ruta.read_bytes()
    assert not datos.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in datos
    assert datos.decode("utf-8") == (
        "codigo,titulo\n"
        "111110,Clase ñ\n"
        '111121,"Clase de ejemplo, con coma"\n'
        "311110,Clase de rango\n"
    )
    assert leer_catalogo(ruta) == clases


@pytest.mark.parametrize(
    "contenido",
    ["cod,titulo\n111110,X\n", "codigo,titulo\n11111,X\n", "codigo,titulo\n111110,X\n111110,Y\n"],
)
def test_leer_catalogo_invalido_lanza_error(tmp_path, contenido):
    ruta = tmp_path / "catalogo.csv"
    ruta.write_bytes(contenido.encode("utf-8"))
    with pytest.raises(ErrorCatalogo):
        leer_catalogo(ruta)
