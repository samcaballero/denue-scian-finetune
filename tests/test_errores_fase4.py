"""Pruebas del paso 18 (src/errores_fase4.py). Todo es inventado: nombres, ids, códigos, títulos y hashes."""

from __future__ import annotations

import copy
import csv
import io
import json
import random
from pathlib import Path

import pytest

from src import errores_fase4 as ef
from src.baselines_fase4 import EstadoGit
from src.evaluacion_final_fase4 import COLUMNAS_ERRORES, ErrorFinal
from src.exploracion_clasificador import ErrorPrivacidad
from src.muestra_humo import ErrorMuestra
from src.splits import sha256_bytes

HASH_TEST = "9" * 64
GIT = EstadoGit(commit="a" * 40, arbol_limpio=True, versionado=True)
FECHA = "2026-11-26"
NOTA_INVENTADA = "Parece la Fonda Secreta del centro"


def fila(i, nombre, act, resp, cat, nivel, sector):
    existente = cat == "codigo_existente"
    return {
        "id": i, "nom_estab": nombre, "codigo_act": act, "titulo_act": f"Título {act}", "respuesta": resp,
        "categoria_regla5": cat, "codigo_respondido": resp if len(resp) == 6 and resp.isdigit() else "",
        "titulo_respondido": f"Título {resp}" if existente else "", "nivel_mas_fino_acertado": nivel,
        "acierta_sector": sector, "sin_nombre": False, "generico": False, "igual_titulo": False,
    }


# En orden de int(id); los automáticos esperados: a, b (subrama), b (rama), juicio, juicio, a (formato inválido).
FILAS = [
    fila("907", "Fonda Azul", "722511", "999999", "codigo_inexistente", "ninguno", False),
    fila("1014", "Tlapalería Norte", "467111", "467112", "codigo_existente", "subrama", True),
    fila("1021", "Papelería Sol", "461110", "461121", "codigo_existente", "rama", True),
    fila("1028", "Café Luna", "722515", "461110", "codigo_existente", "ninguno", False),
    fila("2035", "Estética Mar", "812110", "811111", "codigo_existente", "sector", True),
    fila("10042", "Taller Río", "811111", "no sé", "formato_invalido", "ninguno", False),
]
AUTOS = ["codigo_inexistente", "clases_hermanas", "clases_hermanas", None, None, "codigo_inexistente"]
PROPUESTAS = [
    {"posicion": 1, "tipo": "codigo_inexistente", "razon": "Respuesta fuera del catálogo.", "candidato_charla": True},
    {"posicion": 2, "tipo": "clases_hermanas", "razon": "Clase vecina de la misma rama.", "candidato_charla": False},
    {"posicion": 3, "tipo": "clases_hermanas", "razon": "Clase vecina de la misma rama.", "candidato_charla": True},
    {"posicion": 4, "tipo": "etiqueta_dudosa", "razon": "El giro del nombre no es el de la clase.",
     "candidato_charla": True},
    {"posicion": 5, "tipo": "nombre_ambiguo", "razon": "Sin giro claro en el nombre.", "candidato_charla": False},
    {"posicion": 6, "tipo": "codigo_inexistente", "razon": "Formato inválido.", "candidato_charla": False},
]
REFS = {
    "test": [{"id": f["id"], "nom_estab": f["nom_estab"]} for f in FILAS],
    "train_final": [{"id": "5", "nom_estab": "Abarrotes Cielo", "codigo_act": "461110"}],
}
MANIFEST = {"splits": {"clasificador_test": {"sha256": HASH_TEST}}}
SIN_AVANCE = {"ignorado": lambda p: True, "versionado": lambda r: True, "avance": lambda s: None}


def jsonl(filas) -> bytes:
    return "".join(json.dumps(f, ensure_ascii=False) + "\n" for f in filas).encode("utf-8")


def escribir_errores(raiz: Path, filas) -> None:
    datos = jsonl(filas)
    (raiz / ef.ERRORES).write_bytes(datos)
    res = {"hash_split": {"test": HASH_TEST},
           "errores_para_analisis": {"modelo": ef.MODELO, "conjunto": "test", "n_errores": 60, "n": len(filas),
                                     "archivos": {"jsonl": {"ruta": ef.ERRORES, "sha256": sha256_bytes(datos)}}}}
    (raiz / ef.RESULTADO_FINAL).write_text(json.dumps(res), encoding="utf-8")


def escribir_propuestas(raiz: Path, filas=PROPUESTAS, sha=None) -> None:
    sha = sha or ef.sha_criterios((raiz / ef.RUTA_CRITERIOS).read_bytes())
    (raiz / ef.PROPUESTAS).write_text(json.dumps({"criterios_sha256": sha, "filas": filas}, ensure_ascii=False),
                                      encoding="utf-8")


@pytest.fixture
def raiz(tmp_path: Path) -> Path:
    for d in ("results", "data/fase4", "docs"):
        (tmp_path / d).mkdir(parents=True)
    (tmp_path / ef.RUTA_CRITERIOS).write_bytes(b"# Criterios de prueba\r\nTipos a, b, c, d y otro.\r\n")
    escribir_errores(tmp_path, FILAS)
    return tmp_path


@pytest.fixture
def con_csv(raiz: Path) -> Path:
    escribir_propuestas(raiz)
    ef.correr_proponer(raiz, **SIN_AVANCE)
    return raiz


def filas_csv(datos: bytes) -> list[dict]:
    return list(csv.DictReader(io.StringIO(datos.decode("utf-8-sig"), newline="")))


def revisado(raiz: Path, mutar=None, *, bom=False, crlf=True, sheets=True, barajar=True) -> Path:
    """Simula el CSV exportado de Google Sheets: sin BOM, CRLF, booleanos en mayúsculas, columna extra y filas
    reordenadas."""
    filas = filas_csv((raiz / ef.PARA_REVISION).read_bytes())
    if sheets:
        for r in filas:
            for c in ("acierta_sector", "sin_nombre", "generico", "igual_titulo"):
                r[c] = r[c].upper()
            r["comentario_extra"] = ""
    if mutar:
        filas = mutar(filas)
    if barajar:
        random.Random(3).shuffle(filas)
    texto = io.StringIO()
    campos = list(filas[0].keys()) if filas else list(ef.COLUMNAS_REVISION)
    w = csv.DictWriter(texto, fieldnames=campos, lineterminator="\r\n" if crlf else "\n")
    w.writeheader()
    w.writerows(filas)
    ruta = raiz / "revisado.csv"
    ruta.write_bytes(texto.getvalue().encode("utf-8-sig" if bom else "utf-8"))
    return ruta


def cambiar(cambios: dict[int, dict]):
    def f(filas):
        for r in filas:
            r.update(cambios.get(int(r["posicion"]), {}))
        return filas
    return f


def contar(raiz: Path, ruta: Path, **kw):
    base = dict(raiz=raiz, fecha=FECHA, git=lambda: GIT, manifest=MANIFEST, referencias=lambda r: REFS,
                versionado=lambda r: True, commit_criterios=lambda: "b" * 40, avance=lambda s: None)
    base.update(kw)
    return ef.correr_contar(ruta, **base)


def salida(raiz: Path) -> Path:
    return raiz / "results" / f"{FECHA}_fase4_errores-por-tipo.json"


# --- Entradas y tipos automáticos ---


def test_columnas_de_revision_cambian_id_por_posicion():
    assert ef.COLUMNAS_REVISION[0] == "posicion"
    assert "id" not in ef.COLUMNAS_REVISION
    assert ef.COLUMNAS_REVISION[1:13] == COLUMNAS_ERRORES[1:]
    assert ef.COLUMNAS_REVISION[13:] == ("tipo_propuesto", "razon", "tipo_samuel", "nota_samuel", "candidato_charla")


def test_sha_criterios_ignora_crlf():
    assert ef.sha_criterios(b"a\r\nb\r\n") == ef.sha_criterios(b"a\nb\n") == sha256_bytes(b"a\nb\n")


@pytest.mark.parametrize("i", range(len(FILAS)))
def test_tipo_automatico(i):
    assert ef.tipo_automatico(FILAS[i], i + 1) == AUTOS[i]


def test_tipo_automatico_se_detiene_si_el_nivel_no_coincide_con_la_rama():
    f = {**FILAS[2], "nivel_mas_fino_acertado": "subsector"}
    with pytest.raises(ef.ErrorErrores, match="Posición 3"):
        ef.tipo_automatico(f, 3)
    g = {**FILAS[3], "nivel_mas_fino_acertado": "rama"}
    with pytest.raises(ef.ErrorErrores):
        ef.tipo_automatico(g, 4)


def test_tipo_automatico_se_detiene_si_la_respuesta_es_la_etiqueta():
    with pytest.raises(ef.ErrorErrores):
        ef.tipo_automatico({**FILAS[3], "codigo_respondido": "722515"}, 4)


def test_cargar_errores_verifica_el_sha256(raiz):
    (raiz / ef.ERRORES).write_bytes(jsonl(FILAS[:-1]))
    with pytest.raises(ErrorMuestra):
        ef.cargar_errores(raiz)


def test_cargar_errores_ordena_por_int_id(raiz):
    escribir_errores(raiz, list(reversed(FILAS)))
    filas, info = ef.cargar_errores(raiz)
    assert [f["id"] for f in filas] == [f["id"] for f in FILAS]
    assert info["n"] == 6 and info["n_errores"] == 60 and info["hash_split_test"] == HASH_TEST


def test_cargar_errores_exige_las_columnas_del_paso_17(raiz):
    escribir_errores(raiz, [{**f, "extra": 1} for f in FILAS])
    with pytest.raises(ef.ErrorErrores):
        ef.cargar_errores(raiz)


# --- Vista ---


def test_vista_sin_ids_con_tipos_automaticos(raiz):
    c = ef.correr_vista(raiz, ignorado=lambda p: True, avance=lambda s: None)
    texto = (raiz / ef.VISTA).read_text(encoding="utf-8")
    for f in FILAS:
        assert f["id"] not in texto
        assert f["nom_estab"] in texto
    assert [f"## {i}" in texto for i in range(1, 7)] == [True] * 6
    assert c == {"codigo_inexistente": 2, "clases_hermanas": 2, "por_decidir": 2}
    assert texto.count("por decidir") == 2


def test_vista_no_se_escribe_si_no_esta_ignorada(raiz):
    with pytest.raises(ef.ErrorErrores):
        ef.correr_vista(raiz, ignorado=lambda p: False, avance=lambda s: None)
    assert not (raiz / ef.VISTA).exists()


# --- Propuestas y CSV para revisión ---


def test_proponer_escribe_el_csv_para_samuel(raiz):
    escribir_propuestas(raiz)
    r = ef.correr_proponer(raiz, **SIN_AVANCE)
    datos = (raiz / ef.PARA_REVISION).read_bytes()
    assert datos.startswith(b"\xef\xbb\xbf") and b"\r" not in datos
    texto = datos.decode("utf-8-sig")
    assert texto.split("\n")[0] == ",".join(ef.COLUMNAS_REVISION)
    for f in FILAS:
        assert f["id"] not in texto
    filas = filas_csv(datos)
    assert [x["posicion"] for x in filas] == ["1", "2", "3", "4", "5", "6"]
    assert [x["nom_estab"] for x in filas] == [f["nom_estab"] for f in FILAS]
    assert [x["tipo_propuesto"] for x in filas] == [p["tipo"] for p in PROPUESTAS]
    assert all(x["tipo_samuel"] == "" and x["nota_samuel"] == "" for x in filas)
    assert [x["candidato_charla"] for x in filas] == ["sí", "", "sí", "sí", "", ""]
    assert r["candidatos"] == 3 and r["sha256"] == sha256_bytes(datos)
    assert r["por_tipo_propuesto"] == {"codigo_inexistente": 2, "clases_hermanas": 2, "etiqueta_dudosa": 1,
                                       "nombre_ambiguo": 1, "otro": 0}


def _mutada(i, **cambios):
    filas = copy.deepcopy(PROPUESTAS)
    filas[i].update(cambios)
    return filas


@pytest.mark.parametrize("filas", [
    PROPUESTAS[:-1],                                                     # falta una fila
    [PROPUESTAS[1], PROPUESTAS[0], *PROPUESTAS[2:]],                     # fuera de orden
    _mutada(0, posicion=True),                                           # posición que no es int
    _mutada(4, tipo="ninguno"),                                          # tipo inválido
    _mutada(1, tipo="otro"),                                             # automática con tipo de juicio
    _mutada(3, tipo="clases_hermanas"),                                  # de juicio con tipo automático
    _mutada(4, razon=" ".join(["palabra"] * 16)),                        # más de 15 palabras
    _mutada(4, razon="   "),                                             # razón vacía
    _mutada(4, razon="Dos\nlíneas"),                                     # razón de dos líneas
    _mutada(3, razon="Parece café luna, no comercio"),                   # repite la llave de su fila
    _mutada(4, candidato_charla="sí"),                                   # candidato que no es booleano
    _mutada(1, candidato_charla=True),                                   # dos candidatos del mismo tipo
    _mutada(4, extra=1),                                                 # clave de más
])
def test_proponer_rechaza(raiz, filas):
    escribir_propuestas(raiz, filas)
    with pytest.raises(ef.ErrorErrores) as e:
        ef.correr_proponer(raiz, **SIN_AVANCE)
    assert not (raiz / ef.PARA_REVISION).exists()
    for f in FILAS:
        assert f["nom_estab"] not in str(e.value) and f["id"] not in str(e.value).split()


def test_proponer_rechaza_mas_de_seis_candidatos():
    filas = [{**f, "id": str(5000 + i), "nom_estab": f"Negocio {i}", "codigo_act": "722515", "codigo_respondido": "461110",
              "categoria_regla5": "codigo_existente", "nivel_mas_fino_acertado": "ninguno"} for i, f in
             enumerate([FILAS[3]] * 7)]
    props = [{"posicion": i, "tipo": "otro", "razon": "Error del modelo.", "candidato_charla": True} for i in range(1, 8)]
    with pytest.raises(ef.ErrorErrores, match="máximo es 6"):
        ef.validar_propuestas({"criterios_sha256": "x", "filas": props}, filas, "x")


def test_proponer_exige_los_criterios_de_las_propuestas(raiz):
    escribir_propuestas(raiz, sha="0" * 64)
    with pytest.raises(ef.ErrorErrores, match="criterios"):
        ef.correr_proponer(raiz, **SIN_AVANCE)


def test_proponer_exige_criterios_versionados(raiz):
    escribir_propuestas(raiz)
    with pytest.raises(ef.ErrorErrores, match="git"):
        ef.correr_proponer(raiz, **{**SIN_AVANCE, "versionado": lambda r: False})


def test_proponer_no_escribe_si_no_esta_ignorado(raiz):
    escribir_propuestas(raiz)
    with pytest.raises(ef.ErrorErrores):
        ef.correr_proponer(raiz, **{**SIN_AVANCE, "ignorado": lambda p: False})
    assert not (raiz / ef.PARA_REVISION).exists()


# --- Conteo ---


@pytest.mark.parametrize("texto,esperado", [
    ("", ""), ("  ", ""), ("otro", "otro"), (" Nombre ambiguo ", "nombre_ambiguo"),
    ("Código inexistente", "codigo_inexistente"), ("ETIQUETA-DUDOSA", "etiqueta_dudosa"),
    ("clases_hermanas", "clases_hermanas"), ("ok", "ok"),
])
def test_normalizar_tipo(texto, esperado):
    assert ef.normalizar_tipo(texto) == esperado


def test_leer_csv_omite_filas_vacias_y_columnas_extra():
    datos = "posicion,tipo_propuesto,tipo_samuel,nota_samuel,extra\r\n1,otro,,,x\r\n,,,,\r\n".encode("utf-8")
    assert ef.leer_csv(datos, ef.COLUMNAS_LEIDAS, "CSV") == [
        {"posicion": "1", "tipo_propuesto": "otro", "tipo_samuel": "", "nota_samuel": ""}]


def test_contar_camino_feliz_con_csv_de_sheets(con_csv):
    ruta = revisado(con_csv, cambiar({5: {"tipo_samuel": "Etiqueta dudosa"},
                                      4: {"tipo_samuel": "etiqueta_dudosa"},
                                      2: {"tipo_samuel": "clases_hermanas"},
                                      6: {"nota_samuel": NOTA_INVENTADA}}))
    r = contar(con_csv, ruta)
    c = r["conteos"]
    assert {t: x["filas"] for t, x in c["por_tipo_final"].items()} == {
        "codigo_inexistente": 2, "clases_hermanas": 2, "etiqueta_dudosa": 2, "nombre_ambiguo": 0, "otro": 0}
    assert c["por_tipo_final"]["etiqueta_dudosa"]["sector_acertado"] == 1
    assert c["por_tipo_final"]["clases_hermanas"]["sector_acertado"] == 2
    assert c["por_tipo_final"]["etiqueta_dudosa"]["pct"] == 33.33
    assert c["total"] == {"filas": 6, "sector_acertado": 3, "pct_sector_acertado": 50.0}
    assert c["revision_de_samuel"] == {"cambiadas": 1, "confirmadas_con_el_mismo_tipo": 2, "sin_tocar": 3,
                                       "con_nota": 1}
    assert c["propuesto_a_final"]["nombre_ambiguo"]["etiqueta_dudosa"] == 1
    assert c["propuesto_a_final"]["etiqueta_dudosa"]["etiqueta_dudosa"] == 1
    assert c["por_tipo_propuesto"]["nombre_ambiguo"] == 1
    datos = salida(con_csv).read_bytes()
    assert json.loads(datos) == r
    texto = datos.decode("utf-8")
    for f in FILAS:
        assert f["nom_estab"] not in texto and f'"{f["id"]}"' not in texto
    assert "Fonda Secreta" not in texto and "Clase vecina" not in texto
    for k in ("fecha", "commit", "modelo", "configuracion", "semilla", "split", "hash_split", "tamano_muestra",
              "horas_gpu", "autoria", "cambio_de_diseno", "advertencias"):
        assert k in r
    assert r["hash_split"] == {"test": HASH_TEST} and r["tamano_muestra"] == {"errores_del_modelo_en_test": 60,
                                                                             "muestra": 6}
    assert "D7" in r["cambio_de_diseno"] and "anclaje" in r["cambio_de_diseno"]
    assert r["configuracion"]["criterios"]["commit"] == "b" * 40
    assert "ruta" not in r["fuentes"]["revisado"]


def test_contar_sin_cambios_con_bom_y_lf(con_csv):
    r = contar(con_csv, revisado(con_csv, bom=True, crlf=False, sheets=False, barajar=False))
    assert r["conteos"]["revision_de_samuel"]["sin_tocar"] == 6
    assert r["conteos"]["por_tipo_final"]["nombre_ambiguo"]["filas"] == 1


def _quitar(p):
    return lambda filas: [r for r in filas if r["posicion"] != str(p)]


def _duplicar(p):
    return lambda filas: filas + [dict(next(r for r in filas if r["posicion"] == str(p)))]


def _sin_columna(c):
    return lambda filas: [{k: v for k, v in r.items() if k != c} for r in filas]


@pytest.mark.parametrize("mutar", [
    _quitar(3),                                              # falta una posición
    _duplicar(2),                                            # posición repetida
    cambiar({4: {"posicion": "x"}}),                         # posición inválida
    cambiar({4: {"posicion": "7"}}),                         # posición fuera de rango
    cambiar({5: {"tipo_samuel": "valor_raro_inventado"}}),   # tipo inválido
    cambiar({5: {"tipo_propuesto": "otro"}}),                # tipo_propuesto editado
    cambiar({2: {"tipo_samuel": "otro"}}),                   # automática cambiada
    cambiar({4: {"tipo_samuel": "clases_hermanas"}}),        # automático asignado a una de juicio
    _sin_columna("tipo_samuel"),                             # falta una columna
])
def test_contar_rechaza(con_csv, mutar):
    with pytest.raises(ef.ErrorErrores) as e:
        contar(con_csv, revisado(con_csv, mutar))
    assert not salida(con_csv).exists()
    assert "valor_raro_inventado" not in str(e.value)
    for f in FILAS:
        assert f["nom_estab"] not in str(e.value)


def test_contar_se_niega_si_ya_existe(con_csv):
    (con_csv / "results" / "2026-11-25_fase4_errores-por-tipo.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ef.ErrorErrores, match="Ya existe"):
        contar(con_csv, revisado(con_csv))
    assert not salida(con_csv).exists()


@pytest.mark.parametrize("git", [EstadoGit("a" * 40, False, True), EstadoGit("a" * 40, True, False)])
def test_contar_exige_arbol_limpio_y_script_en_git(con_csv, git):
    with pytest.raises(ef.ErrorErrores, match="árbol limpio"):
        contar(con_csv, revisado(con_csv), git=lambda: git)
    assert not salida(con_csv).exists()


def test_contar_se_detiene_si_cambia_head(con_csv):
    estados = iter([GIT, EstadoGit("c" * 40, True, True)])
    with pytest.raises(Exception, match="cambió"):
        contar(con_csv, revisado(con_csv), git=lambda: next(estados))
    assert not salida(con_csv).exists()


def test_contar_candado_de_privacidad_ante_una_llave_plantada(con_csv):
    refs = {**REFS, "train_final": [{"id": "6", "nom_estab": "Test", "codigo_act": "461110"}]}
    with pytest.raises(ErrorPrivacidad):
        contar(con_csv, revisado(con_csv), referencias=lambda r: refs)
    assert not salida(con_csv).exists()


def test_contar_claves_fijas_no_forman_una_llave_corta_con_el_conteo(con_csv):
    # Regresión del paso 19: la clave fija "aciertan_sector" seguida de su conteo formaba la llave de dos palabras
    # "SECTOR 3" y la regla acotada detenía la escritura. Con "sector_acertado", el conteo ya no queda junto a SECTOR.
    refs = {**REFS, "train_final": [{"id": "6", "nom_estab": "Sector 3", "codigo_act": "461110"}]}
    r = contar(con_csv, revisado(con_csv), referencias=lambda r: refs)
    assert r["conteos"]["total"] == {"filas": 6, "sector_acertado": 3, "pct_sector_acertado": 50.0}
    assert salida(con_csv).exists()


def test_contar_candado_de_ids_ante_un_id_plantado(con_csv):
    refs = {**REFS, "train_final": [{"id": "a" * 40, "nom_estab": "Abarrotes Cielo", "codigo_act": "461110"}]}
    with pytest.raises(ErrorFinal):
        contar(con_csv, revisado(con_csv), referencias=lambda r: refs)
    assert not salida(con_csv).exists()


def test_contar_detecta_criterios_cambiados(con_csv):
    ruta = revisado(con_csv)
    (con_csv / ef.RUTA_CRITERIOS).write_bytes(b"# Otros criterios\n")
    with pytest.raises(ef.ErrorErrores, match="criterios"):
        contar(con_csv, ruta)


def test_contar_detecta_un_csv_para_revision_que_no_es_el_de_las_propuestas(con_csv):
    ruta = revisado(con_csv)
    filas = filas_csv((con_csv / ef.PARA_REVISION).read_bytes())
    filas[4]["tipo_propuesto"] = "otro"
    texto = io.StringIO()
    w = csv.DictWriter(texto, fieldnames=ef.COLUMNAS_REVISION, lineterminator="\n")
    w.writeheader()
    w.writerows(filas)
    (con_csv / ef.PARA_REVISION).write_bytes(texto.getvalue().encode("utf-8-sig"))
    with pytest.raises(ef.ErrorErrores, match="no corresponde"):
        contar(con_csv, ruta)


def test_contar_exige_el_hash_de_test_del_manifest(con_csv):
    with pytest.raises(ef.ErrorErrores, match="manifest"):
        contar(con_csv, revisado(con_csv), manifest={"splits": {"clasificador_test": {"sha256": "1" * 64}}})


# --- Terminal ---


def test_main_exige_un_modo():
    with pytest.raises(SystemExit) as e:
        ef.main([])
    assert e.value.code == 2


def test_main_no_imprime_el_mensaje_de_un_error_inesperado(monkeypatch, capsys):
    def falla(*a, **k):
        raise RuntimeError("Fonda Azul 907")
    monkeypatch.setattr(ef, "correr_vista", falla)
    with pytest.raises(SystemExit) as e:
        ef.main(["--vista"])
    assert e.value.code == 1
    assert "Fonda Azul" not in capsys.readouterr().out


def test_main_detiene_con_el_mensaje_de_un_error_propio(monkeypatch):
    def falla(*a, **k):
        raise ef.ErrorErrores("Posición 3: algo.")
    monkeypatch.setattr(ef, "correr_proponer", falla)
    with pytest.raises(SystemExit) as e:
        ef.main(["--proponer"])
    assert str(e.value.code) == "ME DETENGO: Posición 3: algo."
