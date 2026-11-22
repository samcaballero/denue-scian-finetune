"""Pruebas del hard set del clasificador (Fase 1, paso 7).

Solo usan nombres, llaves, ids y códigos inventados, en memoria o en carpetas temporales;
nunca leen el DENUE ni los splits reales.
"""

import builtins
import csv
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import src.hard_set as hs
from src.hard_set import (
    CAMPOS_HARD,
    COLUMNAS_CSV,
    MAX_FILAS_LLAVE,
    N_HARD,
    N_HARD_PLANEADO,
    PALABRAS_GRAMATICALES,
    PALABRAS_GRAMATICALES_AGREGADAS,
    PALABRAS_GRAMATICALES_ORIGINALES,
    Candidato,
    Entradas,
    ErrorConteo,
    ErrorCuracion,
    ErrorHardSet,
    Marca,
    agregar_entradas,
    agrupar,
    aplicar_ajustes,
    candado_hashes,
    cargar_entradas,
    clases_mas_comunes,
    clases_mezcladas,
    contenidos,
    elegibles,
    escribir_nuevo,
    escribir_splits,
    exclusion_regla7,
    filas_curacion,
    generar_candidatos,
    intercalar,
    leer_ajustes,
    leer_archivos,
    leer_curacion,
    otras_iguales,
    peor_caso,
    registros_hard,
    repartir,
    resumen_candidatos,
    resumen_curacion,
    resumen_hard,
    serializar_csv,
    serializar_hard,
    sin_vocabulario,
    sortear,
    texto_excel,
    tramo_filas,
    validar_curacion,
)
from src.muestra_humo import ErrorMuestra
from src.splits import CAMPOS
from src.splits import serializar as serializar_split

CODIGOS = ("461110", "461121", "722511", "722513", "811111", "811112", "311811", "431110", "541110", "621111")
CATALOGO = {c: f"Clase inventada {c}" for c in CODIGOS}
VOCAB = frozenset({"TALLER", "COMERCIO", "ABARROTES", "PAN", "BAR", "DE", "LA", "SOBRE", "SU", "QUE", "SERVICIOS"})
RONDA_CHICA = ({"i": 5, "ii": 5},)


def _reg(i: int, llave: str, codigo: str, nombre: str | None = None, **marcas) -> dict:
    """Registro con los campos y el orden de src/splits.py."""
    return {
        "id": str(i),
        "nom_estab": nombre if nombre is not None else llave.title(),
        "codigo_act": codigo,
        "llave": llave,
        "sin_nombre": marcas.get("sin_nombre", False),
        "generico": marcas.get("generico", False),
        "igual_titulo": marcas.get("igual_titulo", False),
    }


def _palabra(k: int) -> str:
    """Palabra inventada sin vocales (nunca está en VOCAB)."""
    letras = "BCDFGHJKMNPQRSTVWXZ"
    s, k = "", k + 1
    while k:
        k, r = divmod(k - 1, len(letras))
        s = letras[r] + s
    return s


def _datos(n_i: int = 40, n_ii: int = 40) -> list[dict]:
    """Train candidato inventado: llaves de clases mezcladas, sin vocabulario y no elegibles."""
    regs, i = [], 1
    for k in range(n_i):
        for j in range(2 + k % 4):  # de 2 a 5 filas, todas de clases distintas
            regs.append(_reg(i, f"MEZCLA {_palabra(k)}", CODIGOS[(k + j) % len(CODIGOS)]))
            i += 1
    for k in range(n_ii):
        regs.append(_reg(i, f"ZORRA {_palabra(k)}", CODIGOS[k % len(CODIGOS)]))
        i += 1
    for k in range(10):
        regs.append(_reg(i, f"TALLER {_palabra(k)}", "811111"))  # tiene vocabulario
        regs.append(_reg(i + 1, f"PURA {_palabra(k)}", "461110"))  # una sola clase
        regs.append(_reg(i + 2, f"PURA {_palabra(k)}", "461110"))
        i += 3
    return regs


def _ent(regs, llaves_val=(), previas: int = 5) -> Entradas:
    return Entradas(regs, agrupar(regs), set(llaves_val), len(regs) + previas, previas, "0" * 64,
                    {"train_candidatos": "1" * 64, "validacion": "2" * 64})


def _sha(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def _repo(tmp_path: Path):
    """Repo de juguete con train candidato, validación y test inventados, y su manifest."""
    raiz = tmp_path / "repo"
    splits = raiz / "data" / "splits"
    splits.mkdir(parents=True)
    archivos = {
        "train_candidatos.jsonl": _datos(),
        "val.jsonl": [_reg(10000 + k, f"VALIDACION {_palabra(k)}", "461110") for k in range(8)],
        "test.jsonl": [_reg(20000 + k, f"PRUEBA {_palabra(k)}", "811111") for k in range(10)],
    }
    shas = {}
    for nombre, regs in archivos.items():
        datos = serializar_split(regs)
        (splits / nombre).write_bytes(datos)
        shas[nombre] = _sha(datos)
    entradas = {"test.jsonl": "clasificador_test", "val.jsonl": "clasificador_validacion",
                "train_candidatos.jsonl": "clasificador_train_candidatos"}
    manifest = {"splits": {e: {"ruta": f"data/splits/{n}", "sha256": shas[n], "filas": len(archivos[n])}
                           for n, e in entradas.items()}}
    universo = sum(len(r) for r in archivos.values()) + 7
    return raiz, manifest, universo


def _marcar(filas, marcas: dict[int, tuple[str, str]]) -> list[dict]:
    """Copia de las filas del CSV con incluir y tipo en las posiciones dadas."""
    salida = []
    for k, f in enumerate(filas):
        f = dict(f)
        if k in marcas:
            f["incluir"], f["tipo"] = marcas[k]
        salida.append(f)
    return salida


# --- Palabras gramaticales y fuentes ---


def test_palabras_gramaticales_fijas():
    assert PALABRAS_GRAMATICALES_ORIGINALES == frozenset(
        "A AL CON DE DEL E EL EN LA LAS LO LOS O PARA POR SIN U UN UNA UNOS UNAS Y".split()
    )
    assert PALABRAS_GRAMATICALES_AGREGADAS == {"MEDIANTE", "QUE", "SOBRE", "SU", "SUS"}
    assert not PALABRAS_GRAMATICALES_ORIGINALES & PALABRAS_GRAMATICALES_AGREGADAS
    assert PALABRAS_GRAMATICALES == PALABRAS_GRAMATICALES_ORIGINALES | PALABRAS_GRAMATICALES_AGREGADAS
    assert {"BAR", "PAN", "GAS", "VIA", "ESTOS"}.isdisjoint(PALABRAS_GRAMATICALES)  # con contenido o fuera de la lista


def _filas(*codigos):
    return [_reg(i, "X", c) for i, c in enumerate(codigos, 1)]


def test_clases_mezcladas_frontera_del_60_y_tope_de_tamano():
    assert clases_mezcladas(_filas("461110", "722511"))  # 50 %
    assert not clases_mezcladas(_filas("461110", "461110"))
    assert clases_mezcladas(_filas(*["461110"] * 3, *["722511"] * 2))  # 3/5 = 60 %
    assert not clases_mezcladas(_filas("461110", "461110", "722511"))  # 2/3
    assert not clases_mezcladas(_filas("461110"))  # una sola fila
    veinte = ["461110"] * 10 + ["722511"] * 10
    assert MAX_FILAS_LLAVE == 20
    assert clases_mezcladas(_filas(*veinte))
    assert not clases_mezcladas(_filas(*veinte, "811111"))  # 21 filas


def test_sin_vocabulario_ignora_solo_las_palabras_gramaticales():
    assert sin_vocabulario("LA ZORRA", VOCAB)  # LA está en el vocabulario, pero es gramatical
    assert sin_vocabulario("SU ZORRA SOBRE QUE", VOCAB)
    assert not sin_vocabulario("SU ZORRA", VOCAB, PALABRAS_GRAMATICALES_ORIGINALES)  # sin las agregadas, SU bloquea
    assert not sin_vocabulario("PAN DE ZORRA", VOCAB)
    assert not sin_vocabulario("PANES ZORRA", VOCAB)  # tolerancia de S/ES de en_vocabulario
    assert not sin_vocabulario("BAR ZORRA", VOCAB)  # palabra corta con contenido: cuenta
    assert not sin_vocabulario("DE LA", VOCAB)  # solo palabras gramaticales
    assert sin_vocabulario("ZORRA", VOCAB, frozenset())  # sin lista también entra


def test_elegibles_excluye_por_llave_y_cuenta():
    regs = [
        _reg(1, "MEZCLA A", "461110"), _reg(2, "MEZCLA A", "722511", igual_titulo=True),
        _reg(3, "MEZCLA B", "461110", sin_nombre=True), _reg(4, "MEZCLA B", "722511", sin_nombre=True),
        _reg(5, "MEZCLA C", "461110", generico=True), _reg(6, "MEZCLA C", "722511", generico=True),
        _reg(7, "ZORRA", "811111"),
        _reg(8, "TALLER ZORRA", "811111"),
        _reg(9, "DE LA", "811111"),
        _reg(10, "SU ZORRA", "811111"),
        *[_reg(100 + k, "GRANDE", CODIGOS[k % 2]) for k in range(21)],
    ]
    el = elegibles(agrupar(regs), VOCAB)
    assert el.i == ["MEZCLA C"]
    assert el.ii == ["SU ZORRA", "ZORRA"]
    c = el.conteos
    assert (c["excluidas_sin_nombre"], c["excluidas_igual_titulo"]) == (1, 1)
    assert (c["i_genericas"], c["i_fuera_por_mas_de_20_filas"]) == (1, 1)
    assert (c["ii"], c["ii_con_la_lista_original"], c["ii_sin_ignorar_palabras"]) == (2, 1, 1)
    assert c["ii_solo_palabras_gramaticales"] == 1


def test_genericas_no_pueden_estar_en_la_fuente_ii():
    with pytest.raises(ErrorHardSet, match="genérico"):
        elegibles(agrupar([_reg(1, "ZORRA", "461110", generico=True)]), VOCAB)


# --- Sorteo y reparto ---


def test_sorteo_determinista_e_independiente_del_orden_de_entrada():
    g = agrupar(_datos())
    el = elegibles(g, VOCAB)
    a = sortear(el.i, el.ii, g)
    assert a == sortear(list(reversed(el.i)), list(reversed(el.ii)), g)
    assert a != sortear(el.i, el.ii, g, semilla=43)
    rng = np.random.default_rng(42)
    perm_i = rng.permutation(len(el.i))
    esperado_i = [g[el.i[j]][int(rng.integers(len(g[el.i[j]])))] for j in perm_i]
    esperado_ii = [g[el.ii[j]][0] for j in rng.permutation(len(el.ii))]
    assert a == {"i": esperado_i, "ii": esperado_ii}
    assert len({r["llave"] for r in a["i"]}) == len(el.i)  # una fila por llave


def test_repartir_respeta_el_tope_y_devuelve_donde_sigue():
    sec = [{"codigo_act": c} for c in ("461110", "461121", "461110", "722511", "461110", "811111")]
    assert repartir(sec, 0, 3, tope=2) == ([0, 1, 3], 4)
    assert repartir(sec, 4, 2, tope=2) == ([4, 5], 6)
    with pytest.raises(ErrorHardSet):
        repartir(sec, 0, 6, tope=2)


def test_rondas_siguen_la_permutacion_sin_repetir():
    g = agrupar(_datos())
    r1 = generar_candidatos(g, VOCAB, rondas=RONDA_CHICA, tope=3)
    r2 = generar_candidatos(g, VOCAB, rondas=(*RONDA_CHICA, {"i": 4, "ii": 4}), tope=3)
    assert [c.id for c in r2.de_ronda(1)] == [c.id for c in r1.candidatos]
    ids1, ids2 = {c.id for c in r2.de_ronda(1)}, {c.id for c in r2.de_ronda(2)}
    assert len(ids2) == 8 and not ids1 & ids2
    for f in ("i", "ii"):
        assert min(c.posicion for c in r2.de_ronda(2) if c.fuente == f) >= r1.recorridas[f]
    for ronda in (1, 2):
        for f in ("i", "ii"):
            assert max(Counter(c.sector for c in r2.de_ronda(ronda) if c.fuente == f).values()) <= 3
    assert len({c.llave for c in r2.candidatos}) == len(r2.candidatos)


# --- Regla 7 ---


def test_peor_caso_y_limite_de_la_regla7():
    assert peor_caso([1, 5, 3, 2], n=2) == 8
    assert peor_caso([1, 2], n=5) == 3
    assert not exclusion_regla7(41372, 459520, 4580)["pasa_del_limite"]  # exactamente 10 %
    assert exclusion_regla7(41372, 459520, 4581)["pasa_del_limite"]
    cota = exclusion_regla7(41372, 459520, N_HARD_PLANEADO * MAX_FILAS_LLAVE)
    assert (cota["filas_excluidas"], cota["pct_filas_candidatas"], cota["pasa_del_limite"]) == (44372, 9.66, False)


def test_tramos_de_filas():
    assert [tramo_filas(n) for n in (1, 2, 3, 5, 6, 20)] == ["1", "2", "3", "4-5", "6-10", "11-20"]
    with pytest.raises(ErrorHardSet):
        tramo_filas(21)


def test_resumen_de_candidatos():
    regs = _datos()
    ent = _ent(regs)
    gen = generar_candidatos(ent.grupos, VOCAB, rondas=RONDA_CHICA, tope=3)
    res = resumen_candidatos(gen, ent)
    assert res["ofrecidos"] == {"i": 5, "ii": 5}
    assert sum(c["i"] + c["ii"] for c in res["por_sector"].values()) == 10
    assert sum(res["filas_por_llave"].values()) == 10
    peor = sum(c.filas_llave for c in gen.candidatos)  # 10 candidatos < 150: entran todos
    assert res["peor_caso"]["excluidas_por_hard_set"] == peor
    assert res["palabras_gramaticales"]["agregadas"] == ["MEDIANTE", "QUE", "SOBRE", "SU", "SUS"]


# --- CSV de curación ---


def test_texto_excel():
    assert [texto_excel(t) for t in ("=A", "+A", "-A", "@A", "A=", "")] == [" =A", " +A", " -A", " @A", "A=", ""]


def test_clases_mas_comunes():
    filas = _filas("722511", "461110", "461110", "811111", "722511", "461110")
    assert clases_mas_comunes(filas, CATALOGO) == [
        "461110 (3) Clase inventada 461110", "722511 (2) Clase inventada 722511", "811111 (1) Clase inventada 811111",
    ]
    assert clases_mas_comunes(_filas("461110"), CATALOGO) == ["461110 (1) Clase inventada 461110", "", ""]


def test_intercalar_alterna_las_fuentes_por_ronda():
    def c(ronda, fuente, pos):
        return Candidato(ronda, fuente, pos, _reg(100 * ronda + pos + (50 if fuente == "ii" else 0), "X", "461110"), 1)

    cands = [c(1, "ii", 7), c(1, "i", 3), c(1, "i", 1), c(2, "i", 9), c(2, "ii", 8), c(1, "i", 5)]
    orden = [(x.ronda, x.fuente, x.posicion) for x in intercalar(cands)]
    assert orden == [(1, "i", 1), (1, "ii", 7), (1, "i", 3), (1, "i", 5), (2, "i", 9), (2, "ii", 8)]


def test_csv_de_curacion_bom_id_y_formulas():
    regs = [
        _reg(1, "MEZCLA A", "461110"), _reg(2, "MEZCLA A", "722511"), _reg(3, "MEZCLA A", "461110"),
        _reg(4, "ZORRA", "811111", nombre="=Zorra inventada"),
        _reg(5, "MEZCLA B", "311811", nombre="-Guion, inventado"), _reg(6, "MEZCLA B", "431110"),
    ]
    g = agrupar(regs)
    cands = [Candidato(1, "i", 0, g["MEZCLA A"][2], 3), Candidato(1, "ii", 0, g["ZORRA"][0], 1),
             Candidato(1, "i", 1, g["MEZCLA B"][0], 2)]
    filas, protegidas = filas_curacion(cands, g, CATALOGO)
    assert protegidas == {"nom_estab": 2}
    datos = serializar_csv(filas)
    assert datos.startswith(b"\xef\xbb\xbf") and b"\r" not in datos
    texto = datos.decode("utf-8-sig")
    assert texto.split("\n", 1)[0] == ",".join(COLUMNAS_CSV)
    leidas = list(csv.DictReader(io.StringIO(texto, newline="")))
    assert [f["id"] for f in leidas] == ["id-3", "id-4", "id-5"]
    assert [f["fuente"] for f in leidas] == ["i_clases_mezcladas", "ii_sin_vocabulario", "i_clases_mezcladas"]
    assert leidas[0]["clase_1"] == "461110 (2) Clase inventada 461110"
    assert leidas[0]["clase_3"] == "" and leidas[0]["filas_llave"] == "3"
    assert leidas[0]["descripcion_scian"] == "Clase inventada 461110"
    assert leidas[1]["nom_estab"] == " =Zorra inventada" and leidas[2]["nom_estab"] == " -Guion, inventado"
    assert all(f["incluir"] == f["tipo"] == f["nota"] == "" for f in leidas)


def test_escribir_nuevo_exige_ignorado_y_no_sobrescribe(tmp_path, monkeypatch):
    monkeypatch.setattr(hs, "esta_ignorado", lambda r: True)
    ruta = tmp_path / "a.csv"
    escribir_nuevo(b"x", ruta)
    with pytest.raises(FileExistsError):
        escribir_nuevo(b"y", ruta)
    assert ruta.read_bytes() == b"x"
    monkeypatch.setattr(hs, "esta_ignorado", lambda r: False)
    with pytest.raises(ErrorHardSet):
        escribir_nuevo(b"x", tmp_path / "b.csv")
    assert not (tmp_path / "b.csv").exists()


# --- Lectura de vuelta ---


def _csv_curado(marcas: list[tuple[str, str, str]], delim=",", fin="\n", encoding="utf-8-sig") -> bytes:
    """CSV con todas las columnas; marcas = (id, incluir, tipo)."""
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=delim, lineterminator=fin)
    w.writerow(COLUMNAS_CSV)
    for k, (id_, inc, tipo) in enumerate(marcas):
        w.writerow([id_, f"Café ñandú inventado {k}", "461110", "Clase", "i", "2", "a", "b", "c", inc, tipo, "nota"])
    return buf.getvalue().encode(encoding)


MARCAS_EXCEL = [("id-5", "Sí", "Ambiguo "), ("id-3", " s ", "INDIRECTO"), ("id-9", "no", ""),
                ("7", "", "ambiguo"), ("ID-11", "SI", "indirécto")]


def _resumen(marcas):
    return sorted((m.id, m.incluir, m.tipo, m.tipo_en_fila_n) for m in marcas)


ESPERADO_EXCEL = [("11", True, "indirecto", False), ("3", True, "indirecto", False), ("5", True, "ambiguo", False),
                  ("7", False, None, True), ("9", False, None, False)]


@pytest.mark.parametrize(
    "delim,fin,encoding",
    [(",", "\n", "utf-8-sig"), (",", "\r\n", "utf-8"), (";", "\r\n", "cp1252"), ("\t", "\r\n", "utf-16")],
)
def test_leer_tolera_lo_que_excel_guarda(delim, fin, encoding):
    datos = _csv_curado(MARCAS_EXCEL, delim, fin, encoding)
    assert _resumen(leer_curacion(datos, "x.csv")) == ESPERADO_EXCEL


def test_leer_ignora_filas_vacias_y_orden():
    datos = _csv_curado(list(reversed(MARCAS_EXCEL))) + b",,,,,,,,,,,\n\n"
    assert _resumen(leer_curacion(datos)) == ESPERADO_EXCEL


@pytest.mark.parametrize(
    "id_,incluir,tipo,columna",
    [("id-1", "X", "ambiguo", "incluir"), ("id-1", "S", "", "tipo"), ("id-1", "S", "otro", "tipo"),
     ("1.2E+06", "S", "ambiguo", "id"), ("id-", "N", "", "id")],
)
def test_leer_rechaza_valores_invalidos_sin_mostrar_nombres(id_, incluir, tipo, columna):
    with pytest.raises(ErrorCuracion, match=f"registro 1: columna {columna}") as e:
        leer_curacion(_csv_curado([(id_, incluir, tipo)]), "x.csv")
    assert "ñandú" not in str(e.value) and "Café" not in str(e.value)


def test_leer_exige_las_columnas():
    with pytest.raises(ErrorCuracion, match="columnas"):
        leer_curacion("id,incluir\nid-1,S\n".encode())
    with pytest.raises(ErrorCuracion, match="repetidas"):
        leer_curacion("id,incluir,tipo,tipo\nid-1,S,ambiguo,x\n".encode())


NOMBRE_GOOGLE = "2026-11-22_fase1_hard-candidatos-r1 - Hoja 1-curado.csv"


def _csv_google(marcas: list[tuple[str, str, str]], fin: str = "\r\n", salto_final: bool = False) -> bytes:
    """Como lo descarga Google Sheets: UTF-8 sin BOM, comas, comillas solo donde hacen falta."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator=fin)
    w.writerow(COLUMNAS_CSV)
    for k, (id_, inc, tipo) in enumerate(marcas):
        nombre = f'Café "La Ñandú", inventado {k}'  # comas y comillas dentro de la celda
        nota = 'dice "sí",\nsegunda línea' if k == 0 else ""  # nota de varias líneas (Ctrl+Enter)
        w.writerow([id_, nombre, "461110", "Clase, inventada", "i_clases_mezcladas", "2",
                    "461110 (1) Clase", "722511 (1) Otra", "", inc, tipo, nota])
    texto = buf.getvalue()
    return (texto if salto_final else texto[: -len(fin)]).encode("utf-8")


@pytest.mark.parametrize("fin,salto_final", [("\r\n", False), ("\r\n", True), ("\n", False), ("\n", True)])
def test_leer_tolera_lo_que_descarga_google_sheets(tmp_path, fin, salto_final):
    datos = _csv_google(MARCAS_EXCEL, fin, salto_final)
    assert not datos.startswith(b"\xef\xbb\xbf") and b'"Caf' in datos and b'""' in datos
    ruta = tmp_path / "data" / NOMBRE_GOOGLE  # nombre con espacios y " - Hoja 1"
    ruta.parent.mkdir()
    ruta.write_bytes(datos)
    marcas, archivos = leer_archivos([ruta])
    assert _resumen(marcas) == ESPERADO_EXCEL
    # El registro 1 ocupa dos líneas (nota con salto); la numeración es por registro, no por línea.
    assert {m.id: m.registro for m in marcas} == {"5": 1, "3": 2, "9": 3, "7": 4, "11": 5}
    assert archivos == [{"ruta": NOMBRE_GOOGLE, "sha256": _sha(datos), "tamano_bytes": len(datos)}]


def test_ruta_del_csv_curado(tmp_path, monkeypatch):
    datos_dir = tmp_path / "data"
    datos_dir.mkdir()
    for nombre in (NOMBRE_GOOGLE, "x - Hoja 1.csv", "x-curado.csv"):
        (datos_dir / nombre).write_bytes(b"id,incluir,tipo\n")
    (tmp_path / "fuera-curado.csv").write_bytes(b"id,incluir,tipo\n")
    monkeypatch.setattr(hs, "esta_ignorado", lambda r: True)
    hs.verificar_ruta_curada(datos_dir / NOMBRE_GOOGLE, datos_dir)
    with pytest.raises(ErrorCuracion, match="-curado.csv"):
        hs.verificar_ruta_curada(datos_dir / "x - Hoja 1.csv", datos_dir)
    with pytest.raises(ErrorCuracion, match="data/"):
        hs.verificar_ruta_curada(tmp_path / "fuera-curado.csv", datos_dir)
    with pytest.raises(ErrorCuracion, match="No existe"):
        hs.verificar_ruta_curada(datos_dir / "otro-curado.csv", datos_dir)
    monkeypatch.setattr(hs, "esta_ignorado", lambda r: False)
    with pytest.raises(ErrorCuracion, match="ignorado"):
        hs.verificar_ruta_curada(datos_dir / "x-curado.csv", datos_dir)


def test_el_nombre_de_google_queda_ignorado_por_git_en_data():
    from src.splits import esta_ignorado

    assert esta_ignorado(f"data/{NOMBRE_GOOGLE}")
    assert esta_ignorado("data/2026-11-22_fase1_hard-candidatos-r1-curado.csv")


def _marcas(cands, n_si: int, tipo="ambiguo"):
    return [Marca("x.csv", k, c.id, k <= n_si, tipo if k <= n_si else None, False) for k, c in enumerate(cands, 1)]


def test_validar_curacion():
    g = agrupar(_datos())
    cands = generar_candidatos(g, VOCAB, rondas=RONDA_CHICA, tope=3).candidatos
    marcas = _marcas(cands, 4)
    inc = validar_curacion(marcas, cands, set(), n=4)
    assert [int(c.id) for c, _ in inc] == sorted(int(c.id) for c in cands[:4])
    assert {t for _, t in inc} == {"ambiguo"}
    with pytest.raises(ErrorConteo) as e:
        validar_curacion(marcas, cands, set(), n=5)
    assert (e.value.encontradas, e.value.esperadas, sum(e.value.por_fuente.values())) == (4, 5, 4)
    with pytest.raises(ErrorConteo):
        validar_curacion(marcas, cands, set(), n=3)
    con_cero = [Marca("x.csv", 1, "0" + cands[0].id, True, "ambiguo", False), *marcas[1:]]
    assert len(validar_curacion(con_cero, cands, set(), n=4)) == 4  # el id se compara como número


@pytest.mark.parametrize("caso", ["ajeno", "repetido", "en_val"])
def test_validar_curacion_rechaza(caso):
    g = agrupar(_datos())
    cands = generar_candidatos(g, VOCAB, rondas=RONDA_CHICA, tope=3).candidatos
    marcas, val = _marcas(cands, 4), set()
    if caso == "ajeno":
        marcas.append(Marca("x.csv", 99, "999999", False, None, False))
    elif caso == "repetido":
        marcas.append(marcas[0])
    else:
        val = {cands[0].llave}
    with pytest.raises(ErrorCuracion) as e:
        validar_curacion(marcas, cands, val, n=4)
    assert not isinstance(e.value, ErrorConteo)


def test_tamano_aprobado_del_hard_set():
    # CAMBIO DE DISEÑO del 2026-11-22: leer y congelar piden 105; la PARADA 1 sigue con las 150 planeadas.
    assert (N_HARD, N_HARD_PLANEADO) == (105, 150)
    g = agrupar(_datos(n_i=60, n_ii=60))
    cands = generar_candidatos(g, VOCAB, rondas=({"i": 60, "ii": 60},), tope=60).candidatos
    assert len(validar_curacion(_marcas(cands, 105), cands, set())) == 105  # meta por omisión: N_HARD
    for n_si in (104, 106):
        with pytest.raises(ErrorConteo) as e:
            validar_curacion(_marcas(cands, n_si), cands, set())
        assert (e.value.encontradas, e.value.esperadas) == (n_si, 105)
    assert resumen_candidatos(generar_candidatos(g, VOCAB, rondas=RONDA_CHICA, tope=3), _ent(_datos()))[
        "cota_garantizada"]["filas"] == 150 * MAX_FILAS_LLAVE


# --- Ajustes del chat encima de la curación de la IA ---


def _aj(id_, incluir="N", tipo=None, motivo="motivo inventado"):
    return {"id": id_, "incluir": incluir, "tipo": tipo, "motivo": motivo}


def _json_ajustes(ajustes: list) -> bytes:
    obj = {"descripcion": "Ajustes inventados.", "ajustes": ajustes}
    return (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _m(id_: str, incluir: bool, tipo: str | None = None) -> Marca:
    return Marca("x.csv", int(id_), id_, incluir, tipo if incluir else None, False)


MARCAS_IA = [_m("1", True, "ambiguo"), _m("2", True, "indirecto"), _m("3", False), _m("4", True, "indirecto")]


def test_archivo_de_ajustes_en_data_ignorado_y_con_hash(tmp_path, monkeypatch):
    from src.splits import esta_ignorado

    # Lleva ids del DENUE: vive en data/, ignorado por git; el código solo versiona su SHA-256.
    assert hs.AJUSTES.startswith("data/") and esta_ignorado(hs.AJUSTES)
    assert len(hs.AJUSTES_SHA256) == 64 and set(hs.AJUSTES_SHA256) <= set("0123456789abcdef")
    ruta = tmp_path / "ajustes.json"
    datos = _json_ajustes([_aj("id-5")])
    ruta.write_bytes(datos)
    monkeypatch.setattr(hs, "esta_ignorado", lambda r: True)
    leidos, info = hs.leer_archivo_ajustes(ruta, _sha(datos))
    assert leidos == datos
    assert info == {"sha256": _sha(datos), "tamano_bytes": len(datos), "sha256_coincide_con_el_codigo": True}
    assert not hs.leer_archivo_ajustes(ruta, "0" * 64)[1]["sha256_coincide_con_el_codigo"]
    with pytest.raises(ErrorCuracion, match="No existe"):
        hs.leer_archivo_ajustes(tmp_path / "otro.json")
    monkeypatch.setattr(hs, "esta_ignorado", lambda r: False)
    with pytest.raises(ErrorCuracion, match="ignorado"):
        hs.leer_archivo_ajustes(ruta)


def test_leer_ajustes():
    datos = _json_ajustes([_aj("id-5"), _aj("7", None, "ambiguo", "otro motivo"), _aj("ID-9", None, "indirecto")])
    aj = leer_ajustes(datos)
    assert [(a.numero, a.id, a.a_n, a.tipo) for a in aj] == [
        (1, "5", True, None), (2, "7", False, "ambiguo"), (3, "9", False, "indirecto")]
    assert aj[1].motivo == "otro motivo"


@pytest.mark.parametrize("ajustes,error", [
    ([{"id": "id-5", "incluir": "N", "tipo": None}], "claves"),
    ([{"incluir": "N", "id": "id-5", "tipo": None, "motivo": "m"}], "claves"),
    ([_aj("x-5")], "id con formato"),
    ([_aj(5)], "id con formato"),
    ([_aj("id-5"), _aj("5", None, "ambiguo")], "id repetido"),
    ([_aj("id-5", "N", "ambiguo")], "incluir N con tipo null"),
    ([_aj("id-5", None, None)], "incluir N con tipo null"),
    ([_aj("id-5", "S", None)], "incluir N con tipo null"),
    ([_aj("id-5", None, "otro")], "incluir N con tipo null"),
    ([_aj("id-5", motivo=" ")], "falta el motivo"),
])
def test_leer_ajustes_rechaza(ajustes, error):
    with pytest.raises(ErrorCuracion, match=error):
        leer_ajustes(_json_ajustes(ajustes))


def test_leer_ajustes_exige_json_con_sus_claves():
    for datos in (b"{", b"\xef\xbb\xbf" + _json_ajustes([]), b"[]", b'{"ajustes": []}',
                  b'{"descripcion": "x", "ajustes": [], "otra": 1}'):
        with pytest.raises(ErrorCuracion):
            leer_ajustes(datos)
    assert leer_ajustes(_json_ajustes([])) == []


def test_aplicar_ajustes():
    aj = leer_ajustes(_json_ajustes([_aj("id-1"), _aj("id-02", None, "ambiguo")]))  # el id se compara como número
    salida = aplicar_ajustes(MARCAS_IA, aj)
    assert [(m.id, m.incluir, m.tipo) for m in salida] == [
        ("1", False, None), ("2", True, "ambiguo"), ("3", False, None), ("4", True, "indirecto")]
    assert MARCAS_IA[0].incluir and MARCAS_IA[1].tipo == "indirecto"  # no modifica la entrada
    assert [(m.archivo, m.registro) for m in salida] == [(m.archivo, m.registro) for m in MARCAS_IA]
    assert aplicar_ajustes(MARCAS_IA, []) == MARCAS_IA


@pytest.mark.parametrize("ajuste,error", [
    (_aj("id-3"), "no está en S"),
    (_aj("id-3", None, "ambiguo"), "no está en S"),
    (_aj("id-4", None, "indirecto"), "no cambia nada"),
    (_aj("id-99"), "exactamente una vez"),
])
def test_aplicar_ajustes_se_detiene(ajuste, error):
    with pytest.raises(ErrorCuracion, match=error):
        aplicar_ajustes(MARCAS_IA, leer_ajustes(_json_ajustes([ajuste])))


def test_aplicar_ajustes_rechaza_id_repetido_en_los_csv():
    with pytest.raises(ErrorCuracion, match="exactamente una vez"):
        aplicar_ajustes([*MARCAS_IA, _m("1", True, "ambiguo")], leer_ajustes(_json_ajustes([_aj("id-1")])))


def test_resumen_curacion():
    aj = leer_ajustes(_json_ajustes([_aj("id-1", motivo="b"), _aj("id-4", motivo="b"), _aj("id-2", None, "ambiguo", "a")]))
    assert resumen_curacion(MARCAS_IA, aj) == {
        "filas_con_s_en_los_csv": 3,
        "por_tipo_en_los_csv": {"ambiguo": 1, "indirecto": 2},
        "ajustes": 3,
        "ajustes_a_n": 2,
        "ajustes_de_tipo": {"ambiguo": 1, "indirecto": 0},
        "por_motivo": [{"cambio": "a N", "motivo": "b", "filas": 2}, {"cambio": "tipo ambiguo", "motivo": "a", "filas": 1}],
    }


def test_ajustes_y_validacion_dan_la_meta():
    g = agrupar(_datos())
    cands = generar_candidatos(g, VOCAB, rondas=RONDA_CHICA, tope=3).candidatos
    marcas = _marcas(cands, 6, tipo="indirecto")
    aj = leer_ajustes(_json_ajustes([_aj(f"id-{cands[0].id}"), _aj(f"id-{cands[1].id}", None, "ambiguo")]))
    inc = validar_curacion(aplicar_ajustes(marcas, aj), cands, set(), n=5)
    assert int(cands[0].id) not in {int(c.id) for c, _ in inc}
    assert Counter(t for _, t in inc) == {"indirecto": 4, "ambiguo": 1}


# --- PARADA 2 y congelado ---


def _incluidas_de_prueba():
    regs = [
        _reg(1, "MEZCLA A", "111110"), _reg(2, "MEZCLA A", "111110"), _reg(3, "MEZCLA A", "722511"),
        _reg(4, "ZORRA", "999999"),
        _reg(5, "OTRA", "722511"), _reg(6, "OTRA", "461110"), _reg(7, "MAS", "461110", generico=True),
    ]
    ent = _ent(regs, previas=5)
    g = ent.grupos
    incluidas = [(Candidato(1, "ii", 0, g["ZORRA"][0], 1), "indirecto"),
                 (Candidato(1, "i", 0, g["MEZCLA A"][2], 3), "ambiguo")]
    return regs, ent, incluidas


def test_resumen_hard():
    _, ent, incluidas = _incluidas_de_prueba()
    r = resumen_hard(incluidas, ent)
    assert r["por_tipo"] == {"ambiguo": 1, "indirecto": 1}
    assert r["por_fuente_y_tipo"] == {"i": {"ambiguo": 1, "indirecto": 0}, "ii": {"ambiguo": 0, "indirecto": 1}}
    assert r["por_sector"] == {"72": 1, "99": 1} and r["por_clase"] == {"722511": 1, "999999": 1}
    assert r["filas_que_salen_de_train"] == 4
    assert (r["regla7"]["filas_excluidas"], r["regla7"]["filas_candidatas_train"]) == (9, 12)
    assert r["train_candidatos_sin_hard"] == {"filas": 3, "clases": 2}
    assert r["clases_de_train_sin_ejemplos"] == ["111110", "999999"]
    assert r["clases_del_hard_sin_ejemplos_en_train"] == ["999999"]
    assert r["tipos_con_menos_de_50"] == ["ambiguo", "indirecto"]


def test_contenidos_del_congelado():
    regs, ent, incluidas = _incluidas_de_prueba()
    datos = contenidos(incluidas, ent)
    assert set(datos) == {"hard.jsonl", "train_candidatos_sin_hard.jsonl"}
    hard = [json.loads(l) for l in datos["hard.jsonl"].decode("utf-8").splitlines()]
    assert [tuple(r) for r in hard] == [CAMPOS_HARD] * 2
    assert [(r["id"], r["tipo"]) for r in hard] == [("3", "ambiguo"), ("4", "indirecto")]  # por int(id)
    assert datos["train_candidatos_sin_hard.jsonl"] == serializar_split([r for r in regs if r["id"] in {"5", "6", "7"}])
    assert serializar_split(regs) == "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in regs).encode("utf-8")
    assert contenidos(incluidas, ent) == datos
    assert not datos["hard.jsonl"].startswith(b"\xef\xbb\xbf") and b"\r" not in datos["hard.jsonl"]


def test_serializar_hard_rechaza_tipo_o_claves():
    r = {**_reg(1, "X", "461110"), "tipo": "ambiguo"}
    serializar_hard([r])
    with pytest.raises(ErrorHardSet):
        serializar_hard([{**r, "tipo": "otro"}])
    with pytest.raises(ErrorHardSet):
        serializar_hard([{"tipo": "ambiguo", **_reg(1, "X", "461110")}])
    assert registros_hard([(Candidato(1, "i", 0, _reg(1, "X", "461110"), 1), "ambiguo")]) == [r]


def test_candado_hashes():
    m = {"splits": {"clasificador_test": {"sha256": "a" * 64}}, "otra": {"x_sha256": "c" * 64}}
    candado_hashes({"hard.jsonl": "b" * 64}, m)
    for h in ("a" * 64, "c" * 64):
        with pytest.raises(ErrorHardSet):
            candado_hashes({"hard.jsonl": h}, m)


def test_agregar_entradas_no_toca_lo_demas():
    m = {"version": 1, "splits": {"a": {"x": 1}}, "catalogos": {"c": {"y": 2}}}
    nuevo = agregar_entradas(m, {"clasificador_hard": {"z": 3}})
    assert nuevo["splits"] == {"a": {"x": 1}, "clasificador_hard": {"z": 3}}
    assert m == {"version": 1, "splits": {"a": {"x": 1}}, "catalogos": {"c": {"y": 2}}}
    with pytest.raises(ErrorHardSet):
        agregar_entradas(nuevo, {"clasificador_hard": {}})
    with pytest.raises(ErrorHardSet):
        otras_iguales(m, {**m, "version": 2}, set())
    with pytest.raises(ErrorHardSet):
        otras_iguales(m, {**m, "splits": {"a": {"x": 9}, "clasificador_hard": {}}}, {"clasificador_hard"})


def test_escribir_splits_no_sobrescribe(tmp_path, monkeypatch):
    monkeypatch.setattr(hs, "esta_ignorado", lambda r: True)
    datos = {"hard.jsonl": b"a\n", "train_candidatos_sin_hard.jsonl": b"b\n"}
    rutas = escribir_splits(datos, tmp_path)
    assert {n: r.read_bytes() for n, r in rutas.items()} == datos
    with pytest.raises(ErrorHardSet):
        escribir_splits(datos, tmp_path)


# --- Entradas y test nunca abierto ---


def test_cargar_entradas_y_regla7(tmp_path):
    raiz, manifest, universo = _repo(tmp_path)
    ent = cargar_entradas(manifest, raiz, universo)
    assert len(ent.regs) == manifest["splits"]["clasificador_train_candidatos"]["filas"]
    assert ent.candidatas == universo - 10 - 8
    assert ent.previas == 7
    assert ent.sha_test == manifest["splits"]["clasificador_test"]["sha256"]


def test_cargar_entradas_no_usa_la_ruta_de_test(tmp_path):
    raiz, manifest, universo = _repo(tmp_path)
    del manifest["splits"]["clasificador_test"]["ruta"]
    (raiz / "data" / "splits" / "test.jsonl").unlink()
    assert cargar_entradas(manifest, raiz, universo).previas == 7


def test_cargar_entradas_verifica_hash_y_llaves_de_val(tmp_path):
    raiz, manifest, universo = _repo(tmp_path)
    manifest["splits"]["clasificador_validacion"]["sha256"] = "f" * 64
    with pytest.raises(ErrorMuestra):
        cargar_entradas(manifest, raiz, universo)
    raiz2, manifest2, universo2 = _repo(tmp_path / "otro")
    val = [_reg(10000, "MEZCLA B", "461110")]  # llave que también está en train candidato
    datos = serializar_split(val)
    (raiz2 / "data" / "splits" / "val.jsonl").write_bytes(datos)
    manifest2["splits"]["clasificador_validacion"].update(sha256=_sha(datos), filas=1)
    with pytest.raises(ErrorHardSet, match="validación"):
        cargar_entradas(manifest2, raiz2, universo2)


def test_el_flujo_completo_nunca_abre_test(tmp_path, monkeypatch):
    raiz, manifest, universo = _repo(tmp_path)
    monkeypatch.setattr(hs, "esta_ignorado", lambda r: True)
    abiertos: list[str] = []
    original = io.open

    def espia(archivo, *args, **kwargs):
        if not isinstance(archivo, int):
            abiertos.append(Path(os.fspath(archivo)).name)
        return original(archivo, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", espia)
    monkeypatch.setattr(io, "open", espia)
    ent = cargar_entradas(manifest, raiz, universo)
    gen = generar_candidatos(ent.grupos, VOCAB, rondas=RONDA_CHICA, tope=3)
    filas, _ = filas_curacion(gen.candidatos, ent.grupos, CATALOGO)
    curadas = _marcar(filas, {k: ("S", "ambiguo" if k % 2 else "indirecto") for k in range(6)})
    ruta_csv = raiz / "data" / "curado.csv"
    escribir_nuevo(serializar_csv(curadas), ruta_csv)
    marcas, archivos = leer_archivos([ruta_csv])
    ruta_ajustes = raiz / "evals" / "ajustes.json"
    ruta_ajustes.parent.mkdir()
    ruta_ajustes.write_bytes(_json_ajustes([_aj(f"id-{marcas[0].id}")]))
    marcas = aplicar_ajustes(marcas, leer_ajustes(ruta_ajustes.read_bytes()))
    incluidas = validar_curacion(marcas, gen.candidatos, ent.llaves_val, n=5)
    resumen_hard(incluidas, ent)
    datos = contenidos(incluidas, ent)
    candado_hashes({n: _sha(b) for n, b in datos.items()}, manifest)
    escribir_splits(datos, raiz / "data" / "splits")
    monkeypatch.undo()
    assert {"train_candidatos.jsonl", "val.jsonl", "curado.csv", "ajustes.json", "hard.jsonl"} <= set(abiertos)  # el espía sí ve
    assert "test.jsonl" not in abiertos
    assert archivos[0]["sha256"] == _sha(ruta_csv.read_bytes())
    hard = (raiz / "data" / "splits" / "hard.jsonl").read_bytes().decode("utf-8").splitlines()
    assert len(hard) == 5 and {json.loads(l)["llave"] for l in hard}.isdisjoint(ent.llaves_val)


def test_el_hard_set_sale_completo_de_train(tmp_path):
    raiz, manifest, universo = _repo(tmp_path)
    ent = cargar_entradas(manifest, raiz, universo)
    gen = generar_candidatos(ent.grupos, VOCAB, rondas=RONDA_CHICA, tope=3)
    incluidas = validar_curacion(_marcas(gen.candidatos, 6), gen.candidatos, ent.llaves_val, n=6)
    datos = contenidos(incluidas, ent)
    llaves_hard = {c.llave for c, _ in incluidas}
    sin_hard = [json.loads(l) for l in datos["train_candidatos_sin_hard.jsonl"].decode("utf-8").splitlines()]
    assert not {r["llave"] for r in sin_hard} & llaves_hard
    assert len(sin_hard) == len(ent.regs) - sum(len(ent.grupos[k]) for k in llaves_hard)
    assert [tuple(r) for r in sin_hard[:1]] == [CAMPOS]
