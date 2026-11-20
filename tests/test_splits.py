"""Pruebas del split del clasificador (paso 5).

Solo usan nombres, ids, códigos y títulos sintéticos inventados; nunca leen el DENUE.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.splits import (
    CAMPOS,
    COLUMNAS_LLAVES,
    ErrorRegla7,
    ErrorSplit,
    Particion,
    candado_regla7,
    clases_fuera_de_train,
    conteos,
    contenidos,
    escribir_archivos,
    esta_ignorado,
    partir,
    pasa_del_limite,
    preparar,
    regla7,
    registros,
    serializar,
    top_sin_llaves,
    universo,
    verificar_particion,
)
from src.subgrupos import titulos_normalizados

VOCAB = frozenset({"TIENDAS", "DE", "ABARROTES", "TALLER", "REPARACION"})
TITULOS = titulos_normalizados({"811111": "Reparación de prueba"})


def _df_sintetico(n_grupos: int = 60, semilla: int = 7) -> pd.DataFrame:
    """DENUE de juguete: grupos de tamaños variados, ids desordenados y nombres vacíos."""
    rng = np.random.default_rng(semilla)
    filas = []
    for g in range(n_grupos):
        tamano = int(rng.integers(1, 12)) if g % 10 else 40  # algunos grupos grandes
        for _ in range(tamano):
            filas.append((f"Negocio inventado {g}", f"{811110 + g % 7}", "Actividad inventada"))
    filas += [("", "811110", "Actividad inventada"), ("...", "811111", "Actividad inventada")]
    ids = rng.permutation(np.arange(5, 5 + len(filas)) * 37)  # ids de varios dígitos, sin orden
    return pd.DataFrame(
        {
            "id": [str(i) for i in ids],
            "nom_estab": [f[0] for f in filas],
            "codigo_act": [f[1] for f in filas],
            "nombre_act": [f[2] for f in filas],
        }
    )


def _construir(semilla=42, n_test=20, n_val=10):
    u = preparar(_df_sintetico(), VOCAB, TITULOS)
    llaves = u["llave"].tolist()
    return u, llaves, partir(llaves, semilla, n_test, n_val)


# --- Universo ---


def test_universo_sin_llave_vacia_y_ordenado_por_id_numerico():
    df = pd.DataFrame(
        {
            "id": ["10", "9", "100", "8", "1000"],
            "nom_estab": ["Café B", "Taller A", "...", "", "Taller A."],
            "codigo_act": ["811111"] * 5,
            "nombre_act": [""] * 5,
        }
    )
    u = universo(df)
    assert u["id"].tolist() == ["9", "10", "1000"]  # como texto, "10" iría antes que "9"
    assert u["llave"].tolist() == ["TALLER A", "CAFE B", "TALLER A"]


@pytest.mark.parametrize("ids", [["1", "1"], ["1", "2a"], ["1", "٣"]])  # repetido, letra, dígito no ASCII
def test_universo_rechaza_ids_invalidos(ids):
    df = pd.DataFrame({"id": ids, "nom_estab": ["A", "B"], "codigo_act": ["811111"] * 2, "nombre_act": [""] * 2})
    with pytest.raises(ErrorSplit):
        universo(df)


def test_marcas_y_sin_nombre_act_en_el_universo():
    df = pd.DataFrame(
        {
            "id": ["1", "2", "3", "4"],
            "nom_estab": ["Taller Patito SIN NOMBRE", "Tienda de abarrotes", "Café inventado", "Reparación de prueba"],
            "codigo_act": ["811111"] * 4,
            "nombre_act": ["", "", "Café inventado.", ""],
        }
    )
    u = preparar(df, VOCAB, TITULOS)
    assert "nombre_act" not in u.columns
    assert u["sin_nombre"].tolist() == [True, False, False, False]
    assert u["generico"].tolist() == [False, True, False, False]  # PRUEBA no está en VOCAB
    assert u["igual_titulo"].tolist() == [False, False, True, True]  # por nombre_act y por título


# --- Partición ---


def test_tamanos_exactos():
    _, _, p = _construir()
    assert len(p.test) == 20 and len(p.validacion) == 10


def test_llaves_no_se_cruzan_y_cada_fila_en_un_grupo():
    _, llaves, p = _construir()
    verificar_particion(llaves, p, 20, 10)
    por = {n: {llaves[i] for i in getattr(p, n)} for n in ("test", "validacion", "train_candidatos")}
    assert not por["test"] & por["validacion"]
    assert not por["test"] & por["train_candidatos"]
    assert not por["validacion"] & por["train_candidatos"]
    todos = np.sort(np.concatenate(list(p.conjuntos().values())))
    assert np.array_equal(todos, np.arange(len(llaves)))
    assert {llaves[i] for i in p.excluidas} <= por["test"] | por["validacion"]


def test_validacion_se_salta_las_llaves_de_test():
    _, llaves, p = _construir()
    perm = np.random.default_rng(42).permutation(len(llaves))
    llaves_test = {llaves[i] for i in perm[:20]}
    esperada = [int(i) for i in perm[20:] if llaves[i] not in llaves_test][:10]
    assert sorted(perm[:20].tolist()) == p.test.tolist()
    assert sorted(esperada) == p.validacion.tolist()


def test_verificar_particion_detecta_cruces():
    llaves = ["A", "A", "B", "C"]
    mala = Particion(np.array([0]), np.array([1]), np.array([2, 3]), np.array([], dtype=np.int64))
    with pytest.raises(ErrorSplit):
        verificar_particion(llaves, mala, 1, 1)


def test_misma_semilla_mismos_bytes_otra_semilla_otro_test():
    u, llaves, p = _construir()
    _, _, p2 = _construir()
    assert contenidos(u, p) == contenidos(u, p2)
    _, _, p3 = _construir(semilla=43)
    assert p.test.tolist() != p3.test.tolist()


def test_partir_rechaza_llave_vacia():
    with pytest.raises(ErrorSplit):
        partir(["A", "", "B"], 42, 1, 1)


def test_partir_falla_si_no_alcanzan_las_filas():
    with pytest.raises(ErrorSplit):
        partir(["A", "B"], 42, 2, 1)
    with pytest.raises(ErrorSplit):  # todas las filas comparten llave con test
        partir(["A"] * 10, 42, 2, 1)


# --- Regla 7 ---


def test_limite_de_10_por_ciento():
    assert not pasa_del_limite(10, 100)  # 10 % exacto no pasa
    assert pasa_del_limite(11, 100)
    assert pasa_del_limite(1001, 10000)  # 10.01 %
    candado_regla7(10, 100)
    with pytest.raises(ErrorRegla7):
        candado_regla7(11, 100)


def test_grupo_enorme_dispara_el_candado():
    llaves = ["GRANDE"] * 900 + [f"UNICA {i}" for i in range(100)]
    p = partir(llaves, 42, 10, 5)
    r = regla7(llaves, p)
    assert r["pasa_del_limite"] and r["pct_filas_candidatas"] > 10
    with pytest.raises(ErrorRegla7):
        candado_regla7(r["filas_excluidas"], r["filas_candidatas_train"])


def test_regla7_conteos_y_ranking_con_empates():
    llaves = ["B", "B", "B", "A", "A", "A", "C", "D", "D"]
    p = Particion(np.array([0]), np.array([3]), np.array([6, 7, 8]), np.array([1, 2, 4, 5]))
    r = regla7(llaves, p)
    assert r["filas_candidatas_train"] == 7 and r["filas_excluidas"] == 4
    assert r["pct_filas_candidatas"] == 57.14
    assert (r["por_llaves_de_test"], r["por_llaves_de_validacion"]) == (2, 2)
    assert [t["llave"] for t in r["top"]] == ["A", "B"]  # empate: orden alfabético
    assert r["top"][0] == {
        "rango": 1, "llave": "A", "conjunto": "validacion",
        "filas_del_grupo": 3, "filas_en_el_conjunto": 1, "filas_excluidas": 2,
    }
    assert all("llave" not in t for t in top_sin_llaves(r["top"]))
    assert set(COLUMNAS_LLAVES) - {"llave"} == set(top_sin_llaves(r["top"])[0])


# --- Conteos ---


def test_conteos_y_clases_fuera_de_train():
    u = pd.DataFrame(
        {
            "llave": ["A", "A", "B", "C", "D"],
            "codigo_act": ["111111", "111111", "222222", "111111", "333333"],
            "sin_nombre": [True, True, False, False, False],
            "generico": [False, False, True, False, False],
            "igual_titulo": [False, False, False, False, True],
        }
    )
    p = Particion(np.array([0]), np.array([2]), np.array([3, 4]), np.array([1]))
    c = conteos(u, p)
    assert c["test"] == {"filas": 1, "llaves": 1, "clases": 1, "sin_nombre": 1, "generico": 0, "igual_titulo": 0}
    assert c["excluidas"]["filas"] == 1 and c["train_candidatos"]["igual_titulo"] == 1
    assert clases_fuera_de_train(u, p) == {"test": {"filas": 0, "clases": 0}, "validacion": {"filas": 1, "clases": 1}}


# --- Archivos ---


def test_registros_con_claves_exactas_y_sin_columnas_prohibidas():
    u, _, p = _construir()
    regs = registros(u, p.test)
    assert all(tuple(r) == CAMPOS for r in regs)
    assert CAMPOS == ("id", "nom_estab", "codigo_act", "llave", "sin_nombre", "generico", "igual_titulo")
    assert all(isinstance(r["id"], str) and isinstance(r["generico"], bool) for r in regs)
    assert [int(r["id"]) for r in regs] == sorted(int(r["id"]) for r in regs)


def test_serializar_bytes():
    regs = [
        {"id": "9", "nom_estab": "Café Ñandú", "codigo_act": "811111", "llave": "CAFE NANDU",
         "sin_nombre": False, "generico": False, "igual_titulo": False},
        {"id": "10", "nom_estab": "Taller\ninventado", "codigo_act": "811112", "llave": "TALLER INVENTADO",
         "sin_nombre": False, "generico": True, "igual_titulo": False},
    ]
    datos = serializar(regs)
    assert not datos.startswith(b"\xef\xbb\xbf") and b"\r" not in datos and datos.endswith(b"\n")
    assert "Café Ñandú".encode("utf-8") in datos  # sin escapar los acentos
    lineas = datos.decode("utf-8").split("\n")
    assert len(lineas) == 3 and lineas[-1] == ""  # un salto en el texto no rompe la línea
    assert json.loads(lineas[1])["nom_estab"] == "Taller\ninventado"
    with pytest.raises(ErrorSplit):
        serializar([{"nom_estab": "x", "id": "1"}])


def test_escribir_archivos_no_sobrescribe(tmp_path):
    rutas = escribir_archivos({"test": b"uno\n", "validacion": b"dos\n"}, tmp_path / "splits")
    assert rutas["test"].read_bytes() == b"uno\n" and rutas["validacion"].name == "val.jsonl"
    with pytest.raises(ErrorSplit):
        escribir_archivos({"test": b"otro\n"}, tmp_path / "splits")
    assert rutas["test"].read_bytes() == b"uno\n"


def test_esta_ignorado():
    assert esta_ignorado("data/splits/x.jsonl")
    assert esta_ignorado("data/2026-01-01_fase1_regla7-llaves.csv")
    assert not esta_ignorado("src/x.py")
    assert not esta_ignorado("data/scian_2023_vocabulario.csv")  # excepción (!) en .gitignore
