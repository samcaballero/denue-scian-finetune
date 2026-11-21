"""Pruebas de la muestra de prueba para Kaggle (Fase 2, paso 6a).

Solo usan nombres, ids y códigos inventados en carpetas temporales; nunca leen el DENUE
ni los splits reales.
"""

import builtins
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.exploracion_clasificador import textos
from src.muestra_humo import (
    ARCHIVO_METADATA,
    ARCHIVO_TRAIN,
    ARCHIVO_VAL,
    CAMPOS_HUMO,
    CARPETA,
    ENTRADA,
    ID_KAGGLE,
    PERMITIDOS,
    SECCION,
    ErrorCandado,
    ErrorMuestra,
    agregar_entrada,
    candado_bytes,
    candado_carpeta,
    conteos,
    entrada_kaggle,
    escribir_carpeta,
    hashes_del_manifest,
    metadata_kaggle,
    muestrear,
    otras_iguales,
    parsear,
    preparar,
    seleccionar,
    serializar,
    verificar_fuera_del_repo,
)
from src.splits import CAMPOS as CAMPOS_ORIGEN

TAMANOS_PRUEBA = {ARCHIVO_TRAIN: 20, ARCHIVO_VAL: 5}


def _sha(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def _reg(i: int, nombre: str, codigo: str) -> dict:
    """Registro con los campos y el orden de src/splits.py."""
    return {
        "id": str(i),
        "nom_estab": nombre,
        "codigo_act": codigo,
        "llave": nombre.upper(),
        "sin_nombre": False,
        "generico": False,
        "igual_titulo": False,
    }


def _jsonl(regs) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in regs).encode("utf-8")


def _repo(tmp_path: Path, ids_train=None, ids_val=None):
    """Repo de juguete con train candidato, validación y test inventados, y su manifest."""
    raiz = tmp_path / "repo"
    splits = raiz / "data" / "splits"
    splits.mkdir(parents=True)
    ids_train = ids_train if ids_train is not None else range(5, 5 + 3 * 60, 3)  # 5, 8, 11, ...
    ids_val = ids_val if ids_val is not None else range(1000, 1020)
    archivos = {
        "train_candidatos.jsonl": [_reg(i, f"Negocio inventado {i}", f"{811110 + i % 5}") for i in ids_train],
        "val.jsonl": [_reg(i, f"Café inventado {i}", f"{811110 + i % 7}") for i in ids_val],
        "test.jsonl": [_reg(i, "Taller de prueba", "811111") for i in range(5000, 5010)],
    }
    shas = {}
    for nombre, regs in archivos.items():
        datos = _jsonl(regs)
        (splits / nombre).write_bytes(datos)
        shas[nombre] = _sha(datos)
    manifest = {
        "fuentes": {"denue_cdmx": {"ruta": "../x.csv", "sha256": "0" * 64}},
        "splits": {
            "clasificador_test": {"ruta": "data/splits/test.jsonl", "sha256": shas["test.jsonl"]},
            "clasificador_validacion": {"ruta": "data/splits/val.jsonl", "sha256": shas["val.jsonl"]},
            "clasificador_train_candidatos": {
                "ruta": "data/splits/train_candidatos.jsonl",
                "sha256": shas["train_candidatos.jsonl"],
            },
        },
    }
    return raiz, manifest


# --- Muestreo ---


def test_muestrear_es_el_prefijo_ordenado_de_la_permutacion():
    esperado = np.sort(np.random.default_rng(42).permutation(100)[:10])
    assert np.array_equal(muestrear(100, 10), esperado)
    assert np.array_equal(muestrear(100, 10), muestrear(100, 10))
    assert not np.array_equal(muestrear(100, 10), muestrear(100, 10, semilla=43))


@pytest.mark.parametrize("n, k", [(5, 6), (5, 0), (0, 1)])
def test_muestrear_rechaza_tamanos_invalidos(n, k):
    with pytest.raises(ErrorMuestra):
        muestrear(n, k)


def test_un_generador_nuevo_por_archivo(tmp_path):
    raiz, manifest = _repo(tmp_path)
    _, regs, filas = preparar(manifest, raiz, TAMANOS_PRUEBA)
    origen_val = parsear((raiz / "data/splits/val.jsonl").read_bytes())
    esperados = [origen_val[i]["id"] for i in muestrear(filas[ARCHIVO_VAL], TAMANOS_PRUEBA[ARCHIVO_VAL])]
    assert [r["id"] for r in regs[ARCHIVO_VAL]] == esperados  # no depende de haber muestreado train antes


def test_preparar_mismos_bytes_y_tamanos(tmp_path):
    raiz, manifest = _repo(tmp_path)
    datos, regs, filas = preparar(manifest, raiz, TAMANOS_PRUEBA)
    datos2, _, _ = preparar(manifest, raiz, TAMANOS_PRUEBA)
    assert datos == datos2
    assert set(datos) == PERMITIDOS
    assert filas == {ARCHIVO_TRAIN: 60, ARCHIVO_VAL: 20}
    assert {a: len(r) for a, r in regs.items()} == TAMANOS_PRUEBA
    for rs in regs.values():
        ids = [int(r["id"]) for r in rs]
        assert ids == sorted(ids)


def test_seleccionar_ordena_por_id_numerico():
    regs = [_reg(9, "A", "811111"), _reg(10, "B", "811111"), _reg(100, "C", "811111")]
    salida = seleccionar(regs, [2, 0, 1])
    assert [r["id"] for r in salida] == ["9", "10", "100"]  # como texto, "10" iría antes que "9"


def test_ids_compartidos_entre_muestras_se_detiene(tmp_path):
    raiz, manifest = _repo(tmp_path, ids_train=range(1, 31), ids_val=range(1, 11))
    with pytest.raises(ErrorMuestra):
        preparar(manifest, raiz, {ARCHIVO_TRAIN: 30, ARCHIVO_VAL: 10})


# --- Campos y formato ---


def test_seleccionar_solo_id_nom_estab_y_codigo_act():
    regs = [_reg(1, "Taller inventado", "811111")]
    salida = seleccionar(regs, [0])
    assert tuple(salida[0]) == CAMPOS_HUMO == ("id", "nom_estab", "codigo_act")
    assert not set(salida[0]) & {"llave", "sin_nombre", "generico", "igual_titulo"}
    with pytest.raises(ErrorMuestra):
        seleccionar([{"id": "1", "nom_estab": "X"}], [0])  # falta codigo_act
    with pytest.raises(ErrorMuestra):
        seleccionar([{"id": 1, "nom_estab": "X", "codigo_act": "811111"}], [0])  # id no es texto


def test_serializar_bytes():
    regs = [
        {"id": "9", "nom_estab": "Café Ñandú", "codigo_act": "811111"},
        {"id": "10", "nom_estab": "Taller inventado", "codigo_act": "811112"},
    ]
    datos = serializar(regs)
    assert not datos.startswith(b"\xef\xbb\xbf") and b"\r" not in datos and datos.endswith(b"\n")
    assert "Café Ñandú".encode("utf-8") in datos  # sin escapar los acentos
    assert datos.count(b"\n") == 2
    with pytest.raises(ErrorMuestra):
        serializar([{"id": "1", "nom_estab": "X", "codigo_act": "811111", "llave": "X"}])
    with pytest.raises(ErrorMuestra):
        serializar([{"nom_estab": "X", "id": "1", "codigo_act": "811111"}])  # orden distinto


def test_parsear_acepta_un_origen_valido_con_u2028():
    regs = [_reg(9, "Taller inventado", "811111"), _reg(10, "Café", "811112")]
    salida = parsear(_jsonl(regs))
    assert len(salida) == 2 and salida[0]["nom_estab"] == "Taller inventado"
    assert all(tuple(r) == CAMPOS_ORIGEN for r in salida)


@pytest.mark.parametrize(
    "datos",
    [
        b"\xef\xbb\xbf" + _jsonl([_reg(1, "A", "811111")]),  # BOM
        _jsonl([_reg(1, "A", "811111")])[:-1],  # sin salto final
        b"",  # vacío
        _jsonl([_reg(10, "A", "811111"), _reg(9, "B", "811111")]),  # desordenado por int(id)
        _jsonl([_reg(1, "A", "811111"), _reg(1, "B", "811111")]),  # id repetido
        _jsonl([{**_reg(1, "A", "811111"), "id": "٣"}]),  # dígito no ASCII
        _jsonl([{"nom_estab": "A", **_reg(1, "A", "811111")}]),  # claves en otro orden
        _jsonl([{"id": "1", "nom_estab": "A", "codigo_act": "811111"}]),  # sin llave ni marcas
    ],
)
def test_parsear_rechaza_origenes_invalidos(datos):
    with pytest.raises(ErrorMuestra):
        parsear(datos)


def test_hash_de_origen_alterado_se_detiene(tmp_path):
    raiz, manifest = _repo(tmp_path)
    ruta = raiz / "data/splits/val.jsonl"
    ruta.write_bytes(ruta.read_bytes().replace(b"811111", b"811112", 1))
    with pytest.raises(ErrorMuestra, match="SHA-256"):
        preparar(manifest, raiz, TAMANOS_PRUEBA)


# --- Metadata de Kaggle ---


def test_metadata_ascii_y_limites_de_la_cli():
    datos = metadata_kaggle()
    datos.decode("ascii")  # la CLI lo lee con open() sin codificación (cp1252 en Windows)
    assert b"\r" not in datos and datos.endswith(b"\n")
    meta = json.loads(datos)
    assert meta["id"] == ID_KAGGLE == "samcaballero/denue-scian-humo"
    assert 6 <= len(meta["title"]) <= 50 and 6 <= len(meta["id"].split("/")[1]) <= 50
    assert len(meta["licenses"]) == 1 and meta["licenses"][0]["name"]
    assert "Fuente: INEGI, DENUE" in meta["description"]
    assert json.loads(datos.decode("cp1252")) == meta  # mismo contenido, con acentos, leído en cp1252
    assert "Estadístico" in meta["description"]


# --- Candado ---


def test_candado_carpeta_limpia_pasa(tmp_path):
    raiz, manifest = _repo(tmp_path)
    datos, _, _ = preparar(manifest, raiz, TAMANOS_PRUEBA)
    sha_test = manifest["splits"]["clasificador_test"]["sha256"]
    carpeta = tmp_path / "kaggle-datasets" / "humo"
    escribir_carpeta(datos, carpeta, raiz)
    assert candado_carpeta(carpeta, sha_test, raiz) == candado_bytes(datos, sha_test)


@pytest.mark.parametrize("cambio", ["extra", "oculto", "subcarpeta", "falta"])
def test_candado_rechaza_lo_que_no_es_la_lista_permitida(tmp_path, cambio):
    raiz, manifest = _repo(tmp_path)
    datos, _, _ = preparar(manifest, raiz, TAMANOS_PRUEBA)
    sha_test = manifest["splits"]["clasificador_test"]["sha256"]
    carpeta = tmp_path / "kaggle-datasets" / "humo"
    escribir_carpeta(datos, carpeta, raiz)
    if cambio == "extra":
        (carpeta / "notas.txt").write_bytes(b"x\n")
    elif cambio == "oculto":
        (carpeta / ".oculto").write_bytes(b"x\n")
    elif cambio == "subcarpeta":
        (carpeta / "sub").mkdir()
    else:
        (carpeta / ARCHIVO_VAL).unlink()
    with pytest.raises(ErrorCandado):
        candado_carpeta(carpeta, sha_test, raiz)


def test_candado_rechaza_el_hash_de_test(tmp_path):
    raiz, manifest = _repo(tmp_path)
    datos, _, _ = preparar(manifest, raiz, TAMANOS_PRUEBA)
    bytes_test = _jsonl([_reg(5000, "Taller de prueba", "811111")])  # test inventado
    sha_test = _sha(bytes_test)
    con_test = {**datos, ARCHIVO_VAL: bytes_test}
    with pytest.raises(ErrorCandado, match="test"):
        candado_bytes(con_test, sha_test)
    carpeta = tmp_path / "kaggle-datasets" / "humo"
    escribir_carpeta(con_test, carpeta, raiz)
    with pytest.raises(ErrorCandado, match="test"):
        candado_carpeta(carpeta, sha_test, raiz)


def test_carpeta_dentro_del_repo_se_rechaza(tmp_path):
    raiz, manifest = _repo(tmp_path)
    datos, _, _ = preparar(manifest, raiz, TAMANOS_PRUEBA)
    dentro = raiz / "data" / "kaggle"
    with pytest.raises(ErrorCandado):
        escribir_carpeta(datos, dentro, raiz)
    assert not dentro.exists()
    with pytest.raises(ErrorCandado):
        verificar_fuera_del_repo(raiz, raiz)
    verificar_fuera_del_repo(CARPETA)  # la carpeta real del proyecto queda fuera del repo


def test_el_flujo_completo_nunca_abre_test(tmp_path, monkeypatch):
    raiz, manifest = _repo(tmp_path)
    sha_test = manifest["splits"]["clasificador_test"]["sha256"]
    carpeta = tmp_path / "kaggle-datasets" / "humo"
    abiertos: list[str] = []
    original = io.open

    def espia(archivo, *args, **kwargs):
        if not isinstance(archivo, int):
            abiertos.append(Path(os.fspath(archivo)).name)
        return original(archivo, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", espia)
    monkeypatch.setattr(io, "open", espia)
    datos, _, _ = preparar(manifest, raiz, TAMANOS_PRUEBA)
    candado_bytes(datos, sha_test)
    escribir_carpeta(datos, carpeta, raiz)
    candado_carpeta(carpeta, sha_test, raiz)
    monkeypatch.undo()
    assert {"val.jsonl", "train_candidatos.jsonl", ARCHIVO_VAL} <= set(abiertos)  # el espía sí ve las aperturas
    assert "test.jsonl" not in abiertos


def test_hashes_del_manifest_recorre_todo():
    manifest = {
        "a": {"sha256": "h1", "fuentes": {"denue_cdmx_sha256": "h2", "otro": "h3"}},
        "b": [{"sha256": "h1"}],
    }
    assert hashes_del_manifest(manifest) == {"h1": ["a.sha256", "b[0].sha256"], "h2": ["a.fuentes.denue_cdmx_sha256"]}


# --- Escritura y manifest ---


def test_escribir_carpeta_no_sobrescribe(tmp_path):
    raiz, manifest = _repo(tmp_path)
    datos, _, _ = preparar(manifest, raiz, TAMANOS_PRUEBA)
    carpeta = tmp_path / "kaggle-datasets" / "humo"
    escribir_carpeta(datos, carpeta, raiz)
    with pytest.raises(ErrorMuestra):
        escribir_carpeta({ARCHIVO_VAL: b"otro\n"}, carpeta, raiz)
    assert (carpeta / ARCHIVO_VAL).read_bytes() == datos[ARCHIVO_VAL]
    otra = tmp_path / "kaggle-datasets" / "otra"
    otra.mkdir()
    (otra / "ajeno.txt").write_bytes(b"x\n")
    with pytest.raises(ErrorMuestra):
        escribir_carpeta(datos, otra, raiz)
    with pytest.raises(ErrorCandado):
        escribir_carpeta({"notas.txt": b"x\n"}, tmp_path / "kaggle-datasets" / "nueva", raiz)


def test_entrada_kaggle_y_conteos(tmp_path):
    raiz, manifest = _repo(tmp_path)
    datos, regs, _ = preparar(manifest, raiz, TAMANOS_PRUEBA)
    e = entrada_kaggle(manifest=manifest, datos=datos, regs=regs, commit="abc", fecha="2026-01-01")
    assert e["id_kaggle"] == ID_KAGGLE and e["privado"] is True and e["semilla"] == 42
    assert e["campos"] == list(CAMPOS_HUMO)
    for archivo in (ARCHIVO_TRAIN, ARCHIVO_VAL):
        a = e["archivos"][archivo]
        assert a["sha256"] == _sha(datos[archivo]) and a["tamano_bytes"] == len(datos[archivo])
        assert a["filas"] == TAMANOS_PRUEBA[archivo]
        assert a["clases"] == len({r["codigo_act"] for r in regs[archivo]})
    assert e["archivos"][ARCHIVO_VAL]["origen_sha256"] == manifest["splits"]["clasificador_validacion"]["sha256"]
    assert e["metadata"]["sha256"] == _sha(datos[ARCHIVO_METADATA])
    nombres = {r["nom_estab"] for rs in regs.values() for r in rs}
    assert not {t for _, t in textos(e)} & nombres  # ningún nom_estab en la entrada
    c = conteos(regs)
    assert c["por_archivo"][ARCHIVO_VAL]["filas"] == 5
    assert c["val_clases_ausentes_de_train"] >= 0


def test_conteos_clases_ausentes():
    regs = {
        ARCHIVO_TRAIN: [{"id": "1", "nom_estab": "A", "codigo_act": "111111"}],
        ARCHIVO_VAL: [
            {"id": "2", "nom_estab": "B", "codigo_act": "111111"},
            {"id": "3", "nom_estab": "C", "codigo_act": "222222"},
            {"id": "4", "nom_estab": "D", "codigo_act": "222222"},
        ],
    }
    c = conteos(regs)
    assert c["por_archivo"] == {ARCHIVO_TRAIN: {"filas": 1, "clases": 1}, ARCHIVO_VAL: {"filas": 3, "clases": 2}}
    assert (c["val_clases_ausentes_de_train"], c["val_filas_de_clases_ausentes"]) == (1, 2)


def test_agregar_entrada_solo_agrega_la_seccion_nueva():
    manifest = {"fuentes": {"x": {"sha256": "h"}}, "splits": {"clasificador_test": {"sha256": "t"}}}
    original = json.loads(json.dumps(manifest))
    nuevo = agregar_entrada(manifest, {"id_kaggle": ID_KAGGLE})
    assert manifest == original  # no muta el original
    assert nuevo[SECCION][ENTRADA] == {"id_kaggle": ID_KAGGLE}
    assert {k: v for k, v in nuevo.items() if k != SECCION} == original
    with pytest.raises(ErrorMuestra):
        agregar_entrada(nuevo, {"id_kaggle": "otro"})
    cambiado = json.loads(json.dumps(nuevo))
    cambiado["splits"]["clasificador_test"]["sha256"] = "otro"
    with pytest.raises(ErrorMuestra):
        otras_iguales(manifest, cambiado)
    otras_iguales(manifest, nuevo)
