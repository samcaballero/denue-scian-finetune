"""Pruebas del análisis exploratorio del paso 4.

Solo usan nombres, códigos y títulos sintéticos inventados; nunca leen el DENUE.
"""

import itertools
from collections import Counter

import numpy as np
import pytest

from src.exploracion_clasificador import (
    COLUMNAS_REVISION,
    ErrorPrivacidad,
    a_distancia_1,
    clases_por_llave,
    clasificar_titulo,
    codigo_en_nivel,
    conteos_por_llave,
    efecto_de_excluir,
    en_vocabulario,
    es_generico,
    escribir_revision,
    exclusion_esperada,
    filas_de_revision,
    mapa_sectores,
    por_nivel,
    posicion_marcador,
    prob_sin_tocar,
    pureza_grupos,
    quitar_marcador,
    resumen_clases,
    simular_exclusion,
    techo,
    tiene_errata_marcador,
    tiene_marcador,
    topes_por_clase,
    verificar_privacidad,
)

SECTORES = ["11", "31-33", "46", "48-49"]


# --- Niveles ---


def test_mapa_sectores_con_rangos():
    mapa = mapa_sectores(SECTORES)
    assert mapa == {"11": "11", "31": "31-33", "32": "31-33", "33": "31-33", "46": "46", "48": "48-49", "49": "48-49"}


def test_codigo_en_nivel():
    mapa = mapa_sectores(SECTORES)
    assert codigo_en_nivel("322110", 2, mapa) == "31-33"
    assert codigo_en_nivel("322110", 3, mapa) == "322"
    assert codigo_en_nivel("322110", 5, mapa) == "32211"
    assert codigo_en_nivel("322110", 6, mapa) == "322110"


def test_por_nivel_sectores():
    mapa = mapa_sectores(SECTORES)
    catalogo = ["111110", "311110", "322110", "461110", "481111"]
    filas = {"311110": 3, "322110": 1, "461110": 6}
    res = por_nivel(filas, {s: f"Sector {s}" for s in SECTORES}, catalogo, 2, mapa, con_titulo=True)
    assert res["categorias_catalogo"] == 4 and res["categorias_con_filas"] == 2 and res["filas"] == 10
    assert res["detalle"]["31-33"] == {
        "titulo": "Sector 31-33", "clases_catalogo": 2, "clases_presentes": 2, "filas": 4, "pct_filas": 40.0,
    }
    assert res["detalle"]["11"]["filas"] == 0 and res["detalle"]["11"]["clases_catalogo"] == 1


# --- Clases ---


def test_resumen_clases():
    filas = {"A": 1, "B": 5, "C": 20, "D": 200}
    res = resumen_clases(filas, ["A", "B", "C", "D", "E", "F"])
    assert res["clases_presentes"] == 4 and res["clases_del_catalogo_sin_filas"] == 2
    assert res["clases_presentes_fuera_del_catalogo"] == []
    q = res["filas_por_clase_presente"]
    assert (q["minimo"], q["q1"], q["mediana"], q["q3"], q["maximo"]) == (1, 4.0, 12.5, 65.0, 200)
    assert res["clases_con_menos_de"] == {"10": 2, "50": 3, "100": 3}
    assert res["cobertura_de_las_clases_mas_grandes"]["10"] == {"filas": 226, "pct_filas": 100.0}


def test_resumen_clases_fuera_del_catalogo():
    res = resumen_clases({"A": 2, "Z": 1}, ["A"])
    assert res["clases_presentes_fuera_del_catalogo"] == ["Z"]


# --- Marcador, erratas, genéricos y títulos ---


def test_marcador_como_palabras_completas():
    assert tiene_marcador("TALLER PATITO SIN NOMBRE")
    assert tiene_marcador("SIN NOMBRE TALLER")
    assert not tiene_marcador("TALLER SINNOMBRE")
    assert not tiene_marcador("TALLER SIN NOMBRES")
    assert not tiene_marcador("PASIN NOMBRE")


def test_quitar_marcador_y_posicion():
    assert quitar_marcador("TALLER PATITO SIN NOMBRE") == "TALLER PATITO"
    assert quitar_marcador("A SIN NOMBRE B") == "A B"
    assert quitar_marcador("SIN NOMBRE") == ""
    assert posicion_marcador("SIN NOMBRE") == "sola"
    assert posicion_marcador("TALLER SIN NOMBRE") == "final"
    assert posicion_marcador("SIN NOMBRE TALLER") == "inicio"
    assert posicion_marcador("A SIN NOMBRE B") == "en_medio"
    assert posicion_marcador("TALLER PATITO") is None


def test_distancia_1():
    assert a_distancia_1("NOMBR", "NOMBRE")  # falta una letra
    assert a_distancia_1("NONBRE", "NOMBRE")  # una letra cambiada
    assert a_distancia_1("NOMBRES", "NOMBRE")  # sobra una letra
    assert not a_distancia_1("NOMBRE", "NOMBRE")
    assert not a_distancia_1("NOMRBE", "NOMBRE")  # dos cambios
    assert not a_distancia_1("NOM", "NOMBRE")


def test_errata_marcador():
    assert tiene_errata_marcador("TALLER SIN NOMBR")
    assert tiene_errata_marcador("TALLER SIN NOMBRES")
    assert tiene_errata_marcador("SIN SIN NONBRE")  # la palabra tras el segundo SIN
    assert not tiene_errata_marcador("TALLER SIN NOMBRE")  # es marcador, no errata
    assert not tiene_errata_marcador("TALLER SIN AZUCAR")


def test_generico_con_vocabulario_chico():
    vocab = frozenset({"TIENDAS", "DE", "ABARROTES", "FLOR", "Y"})
    assert en_vocabulario("TIENDA", vocab)  # vocabulario en plural con S
    assert en_vocabulario("FLORES", vocab)  # plural con ES
    assert en_vocabulario("ABARROTE", vocab)
    assert not en_vocabulario("PATITO", vocab)
    assert es_generico("TIENDA DE ABARROTES", vocab)
    assert es_generico("ABARROTES SIN NOMBRE", vocab)  # la frase no cuenta
    assert not es_generico("ABARROTES PATITO", vocab)
    assert not es_generico("DE Y", vocab)  # sin palabras de 4 letras
    assert not es_generico("SIN NOMBRE", vocab)
    assert not es_generico("", vocab)


def test_clasificar_titulo():
    codigo_por_titulo = {"TALLER DE PRUEBA": "811111", "OTRA ACTIVIDAD": "999999"}
    assert clasificar_titulo("TALLER DE PRUEBA", "811111", codigo_por_titulo) == "su_clase"
    assert clasificar_titulo("TALLER DE PRUEBA", "811112", codigo_por_titulo) == "otra_clase"
    assert clasificar_titulo("TALLER PATITO", "811111", codigo_por_titulo) is None
    assert clasificar_titulo("", "811111", codigo_por_titulo) is None


# --- Grupos y techo ---

LLAVES = ["A", "A", "A", "A", "B", "B", "C", "", ""]
CLASES = ["111111", "111111", "111111", "112222", "211111", "211112", "311111", "111111", "211111"]


def test_conteos_y_techo():
    conteos = conteos_por_llave(LLAVES, CLASES)
    assert conteos["A"] == Counter({"111111": 3, "112222": 1})
    # Llaves A (4 filas, 3 mayoritarias) y B (2 filas, 1); C es de 1 fila y "" no cuenta.
    assert techo(conteos) == {"llaves": 2, "filas": 6, "filas_clase_mayoritaria": 4, "pct": 66.67}
    assert techo(conteos, minimo_filas=1)["filas"] == 7


def test_techo_por_nivel_con_prefijos():
    conteos = conteos_por_llave(LLAVES, [c[:2] for c in CLASES])
    assert techo(conteos)["pct"] == 100.0  # A: todas en 11; B: todas en 21


def test_clases_por_llave():
    res = clases_por_llave(conteos_por_llave(LLAVES, CLASES))
    assert res["2"] == {"llaves": 2, "filas": 6}
    assert res["1"] == {"llaves": 0, "filas": 0}


def test_pureza_grupos():
    conteos = {
        "A": Counter({"x": 100}),
        "B": Counter({"x": 60, "y": 40}),
        "C": Counter({"x": 45, "y": 30, "z": 25}),
        "D": Counter({"x": 95, "y": 5}),
        "E": Counter({"x": 50}),  # menos de 100 filas
        "": Counter({"x": 300}),  # llave vacía
    }
    res = pureza_grupos(conteos, minimo=100)
    assert res["grupos"] == 4
    assert res["tramos"]["1"] == {"grupos": 1, "filas": 100}
    assert res["tramos"]["0.9_a_menos_de_1"]["grupos"] == 1
    assert res["tramos"]["0.5_a_0.7"]["grupos"] == 1
    assert res["tramos"]["menos_de_0.5"]["grupos"] == 1
    assert (res["minimo"], res["maximo"]) == (0.45, 1.0)


# --- Regla 7 ---


def _exclusion_exacta(tamanos, m):
    """Promedio de filas excluidas sobre todas las muestras posibles de m filas."""
    ids = [g for g, n in enumerate(tamanos) for _ in range(n)]
    totales = []
    for muestra in itertools.combinations(range(len(ids)), m):
        tocadas = Counter(ids[i] for i in muestra)
        totales.append(sum(tamanos[g] - k for g, k in tocadas.items()))
    return sum(totales) / len(totales)


@pytest.mark.parametrize("tamanos,m", [([2, 1, 1, 1, 1], 2), ([3, 2, 1], 2), ([4, 1, 1], 3)])
def test_exclusion_analitica_contra_enumeracion(tamanos, m):
    assert exclusion_esperada(tamanos, m).sum() == pytest.approx(_exclusion_exacta(tamanos, m))


def test_prob_sin_tocar():
    # 6 filas, muestra de 2: un grupo de 2 queda sin tocar con C(4,2)/C(6,2) = 6/15.
    assert prob_sin_tocar([2], 6, 2)[0] == pytest.approx(6 / 15)
    assert prob_sin_tocar([5], 6, 2)[0] == 0.0  # no caben 2 filas fuera del grupo


def test_simulacion_reproducible_y_acotada():
    tamanos = np.array([5, 3, 1, 1])
    ids = np.repeat(np.arange(4), tamanos)
    a = simular_exclusion(ids, tamanos, 3, 1000)
    b = simular_exclusion(ids, tamanos, 3, 1000)
    assert (a == b).all()
    assert a.sum() >= 0 and a.sum() <= tamanos.sum() - 3
    assert (a <= tamanos - 1).all()


# --- Topes y efectos ---


def test_topes_por_clase():
    res = topes_por_clase({"A": 50, "B": 150, "C": 1200}, topes=(100, 1000))
    assert res["100"] == {"filas_que_quedan": 250, "pct_que_queda": 17.86, "filas_recortadas": 1150, "clases_recortadas": 2}
    assert res["1000"]["filas_que_quedan"] == 1200 and res["1000"]["clases_recortadas"] == 1


def test_efecto_de_excluir():
    llaves = ["A", "A", "B", "B", "C", "D"]
    clases = ["x", "x", "y", "z", "z", "w"]
    mascara = [False, False, True, True, False, True]
    res = efecto_de_excluir(llaves, clases, mascara, muestra=2)
    assert res["filas"] == 3 and res["llaves"] == 2 and res["clases_afectadas"] == 3
    assert res["clases_que_se_quedan_sin_filas"] == 2  # y y w
    assert res["filas_esperadas_en_muestra_de_2"] == 1.0
    assert res["techo_clase_llaves_repetidas_pct"] == {"con_la_categoria": 75.0, "sin_la_categoria": 100.0}


# --- Revisión y privacidad ---


def test_filas_de_revision_y_bom(tmp_path):
    conteos = {"TALLER PATITO SIN NOMBRE": Counter({"811111": 3, "811112": 3}), "LA ESQUINITA": Counter({"811112": 2}), "": Counter({"811111": 9})}
    catalogo = {"811111": "Título uno", "811112": "Título dos"}
    marcas = {"LA ESQUINITA": {"su_clase": 0, "otra_clase": 1, "nombre_act": 2}}
    filas = filas_de_revision(conteos, catalogo, marcas, frozenset({"TALLER"}), top=5)
    assert [f["llave"] for f in filas] == ["TALLER PATITO SIN NOMBRE", "LA ESQUINITA"]  # sin la vacía
    assert filas[0]["clase_mayoritaria"] == "811111"  # empate: el código menor
    assert filas[0]["proporcion_mayoritaria"] == "0.5000" and filas[0]["marcador"] == "SI"
    assert filas[1]["filas_igual_nombre_act"] == 2 and filas[1]["marcador"] == "NO"
    ruta = tmp_path / "revision.csv"
    escribir_revision(filas, ruta)
    datos = ruta.read_bytes()
    assert datos[:3] == b"\xef\xbb\xbf"
    assert datos[3:].decode("utf-8").splitlines()[0] == ",".join(COLUMNAS_REVISION)


def test_verificar_privacidad():
    llaves = {"LA ESQUINITA", "TALLER DE PRUEBA", "123"}
    permitidos = {"TALLER DE PRUEBA"}
    verificar_privacidad({"a": "Taller de prueba", "b": ["123", "2026-11-20"], "c": 5}, llaves, permitidos)
    with pytest.raises(ErrorPrivacidad) as e:
        verificar_privacidad({"a": {"b": ["ok", "La Esquinita."]}}, llaves, permitidos)
    assert "$.a.b[1]" in str(e.value) and "ESQUINITA" not in str(e.value).upper()
