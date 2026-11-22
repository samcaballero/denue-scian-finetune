"""Pruebas del paquete del clasificador para Kaggle (Fase 1, paso 8).

Solo usan nombres, llaves, ids y códigos inventados en carpetas temporales; nunca leen el
DENUE ni los splits reales.
"""

import builtins
import hashlib
import io
import json
import os
from pathlib import Path

import pytest

from src.exploracion_clasificador import textos
from src.muestra_humo import SECCION, ErrorCandado, ErrorMuestra, agregar_entrada, escribir_carpeta
from src.paquete_clasificador import (
    ARCHIVO_METADATA,
    ARCHIVO_TRAIN,
    ARCHIVO_VAL,
    CAMPOS,
    CARPETA,
    ENTRADA,
    ID_KAGGLE,
    PERMITIDOS,
    TITULO,
    ErrorPaquete,
    candado_en_carpeta,
    candado_en_memoria,
    candado_manifest,
    conteos,
    entrada_kaggle,
    preparar,
    verificar,
)
from src.muestra_humo import verificar_fuera_del_repo
from src.splits import serializar as serializar_split


def _sha(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def _reg(i: int, nombre: str, codigo: str) -> dict:
    """Registro con los campos y el orden de src/splits.py."""
    return {"id": str(i), "nom_estab": nombre, "codigo_act": codigo, "llave": nombre.upper(),
            "sin_nombre": False, "generico": False, "igual_titulo": False}


def _repo(tmp_path: Path, train=None, val=None):
    """Repo de juguete con train final, validación, test y hard inventados, y su manifest."""
    raiz = tmp_path / "repo"
    splits = raiz / "data" / "splits"
    splits.mkdir(parents=True)
    archivos = {
        "train_final.jsonl": train if train is not None else
        [_reg(i, f"Taller inventado {i}", f"{811110 + i % 4}") for i in range(3, 300, 7)],
        "val.jsonl": val if val is not None else [_reg(1000 + i, f"Café inventado {i}", f"{811110 + i % 5}") for i in range(12)],
        "test.jsonl": [_reg(5000 + i, f"Prueba inventada {i}", "811111") for i in range(4)],
        "hard.jsonl": [_reg(6000 + i, f"Difícil inventado {i}", "811112") for i in range(3)],
    }
    shas = {}
    for nombre, regs in archivos.items():
        datos = serializar_split(regs)
        (splits / nombre).write_bytes(datos)
        shas[nombre] = _sha(datos)
    entradas = {"train_final.jsonl": "clasificador_train_final", "val.jsonl": "clasificador_validacion",
                "test.jsonl": "clasificador_test", "hard.jsonl": "clasificador_hard"}
    manifest = {
        "version": 1,
        "splits": {e: {"ruta": f"data/splits/{n}", "sha256": shas[n], "filas": len(archivos[n])}
                   for n, e in entradas.items()},
        "kaggle_datasets": {"denue_scian_humo": {"archivos": {"humo_val.jsonl": {"sha256": "e" * 64}}}},
    }
    return raiz, manifest


def _lineas(datos: bytes) -> list[dict]:
    return [json.loads(l) for l in datos.decode("utf-8").split("\n")[:-1]]


# --- Contenido ---


def test_preparar_todas_las_filas_con_tres_campos(tmp_path):
    raiz, manifest = _repo(tmp_path)
    datos, regs = preparar(manifest, raiz)
    assert set(datos) == PERMITIDOS
    for archivo, origen in ((ARCHIVO_TRAIN, "train_final.jsonl"), (ARCHIVO_VAL, "val.jsonl")):
        completos = _lineas((raiz / "data" / "splits" / origen).read_bytes())
        paquete = _lineas(datos[archivo])
        assert [r["id"] for r in paquete] == [r["id"] for r in completos]  # exactamente los mismos ids, en orden
        assert all(tuple(r) == CAMPOS == ("id", "nom_estab", "codigo_act") for r in paquete)
        assert paquete == [{c: r[c] for c in CAMPOS} for r in completos]
        assert not datos[archivo].startswith(b"\xef\xbb\xbf") and b"\r" not in datos[archivo]
    assert preparar(manifest, raiz)[0] == datos  # mismos bytes desde cero
    c = conteos(regs)
    assert c["por_archivo"][ARCHIVO_VAL] == {"filas": 12, "clases": 5}
    assert c["val_clases_ausentes_de_train"] == 1  # 811114 no está en train


def test_preparar_rechaza_ids_o_llaves_compartidos_filas_y_hash(tmp_path):
    raiz, manifest = _repo(tmp_path, val=[_reg(3, "Café inventado", "811111")])  # id 3 también en train
    with pytest.raises(ErrorPaquete, match="id"):
        preparar(manifest, raiz)
    raiz, manifest = _repo(tmp_path / "llave", val=[_reg(2000, "Taller inventado 3", "811111")])
    with pytest.raises(ErrorPaquete, match="llaves"):
        preparar(manifest, raiz)
    raiz, manifest = _repo(tmp_path / "filas")
    manifest["splits"]["clasificador_train_final"]["filas"] = 1
    with pytest.raises(ErrorPaquete, match="filas"):
        preparar(manifest, raiz)
    raiz, manifest = _repo(tmp_path / "hash")
    ruta = raiz / "data" / "splits" / "val.jsonl"
    ruta.write_bytes(ruta.read_bytes().replace(b"811111", b"811119", 1))
    with pytest.raises(ErrorMuestra, match="SHA-256"):
        preparar(manifest, raiz)


def test_metadata_ascii_privado_y_credito(tmp_path):
    raiz, manifest = _repo(tmp_path)
    datos, _ = preparar(manifest, raiz)
    meta_bytes = datos[ARCHIVO_METADATA]
    meta_bytes.decode("ascii")  # la CLI lo lee con open() sin codificación (cp1252 en Windows)
    meta = json.loads(meta_bytes)
    assert meta["id"] == ID_KAGGLE == "samcaballero/denue-scian-clasificador"
    assert meta["title"] == TITULO and 6 <= len(TITULO) <= 50 and 6 <= len(ID_KAGGLE.split("/")[1]) <= 50
    assert meta["licenses"] == [{"name": "other"}]
    assert meta["description"].startswith("Fuente: INEGI, DENUE") and "Términos de Libre Uso" in meta["description"]
    assert "isPrivate" not in meta and "public" not in json.dumps(meta).lower()  # privado: se sube sin -u


# --- Candado ---


def test_candado_en_memoria_pasa_y_rechaza(tmp_path):
    raiz, manifest = _repo(tmp_path)
    datos, _ = preparar(manifest, raiz)
    hashes = candado_en_memoria(datos, manifest)
    assert hashes == {n: _sha(b) for n, b in datos.items()}
    splits = raiz / "data" / "splits"
    casos = {
        "test": (splits / "test.jsonl").read_bytes(),  # bytes del test inventado (en la prueba sí se leen)
        "hard": (splits / "hard.jsonl").read_bytes(),
        "train_final completo": (splits / "train_final.jsonl").read_bytes(),  # otra entrada del manifest
    }
    for nombre, b in casos.items():
        with pytest.raises(ErrorCandado):
            candado_en_memoria({**datos, ARCHIVO_VAL: b}, manifest)
    with pytest.raises(ErrorCandado, match="congelado"):
        candado_en_memoria({**datos, ARCHIVO_TRAIN: casos["hard"]}, manifest)
    with pytest.raises(ErrorCandado):
        candado_en_memoria({**datos, "notas.txt": b"x\n"}, manifest)
    with pytest.raises(ErrorCandado):
        candado_en_memoria({k: v for k, v in datos.items() if k != ARCHIVO_VAL}, manifest)


def test_candado_manifest_excluye_solo_la_entrada_propia():
    m = {"splits": {"clasificador_test": {"sha256": "a" * 64}, "clasificador_hard": {"sha256": "b" * 64}},
         SECCION: {ENTRADA: {"archivos": {ARCHIVO_VAL: {"sha256": "c" * 64}}},
                   "otra": {"archivos": {"x": {"origen_sha256": "d" * 64}}}}}
    candado_manifest({ARCHIVO_VAL: "c" * 64}, m)  # su propio hash registrado
    for h in ("a" * 64, "b" * 64, "d" * 64):
        with pytest.raises(ErrorCandado):
            candado_manifest({ARCHIVO_VAL: h}, m)


def test_candado_en_carpeta_y_verificar(tmp_path):
    raiz, manifest = _repo(tmp_path)
    datos, regs = preparar(manifest, raiz)
    carpeta = tmp_path / "kaggle-datasets" / "clasificador"
    escribir_carpeta(datos, carpeta, raiz, permitidos=PERMITIDOS)
    hashes = candado_en_carpeta(carpeta, manifest, raiz)
    assert hashes == candado_en_memoria(datos, manifest)
    assert verificar(manifest, carpeta, raiz) == (hashes, False)
    e = entrada_kaggle(manifest=manifest, datos=datos, regs=regs, commit="abc", fecha="2026-01-01")
    registrado = agregar_entrada(manifest, e, ENTRADA)
    assert verificar(registrado, carpeta, raiz) == (hashes, True)
    otro = json.loads(json.dumps(registrado))
    otro[SECCION][ENTRADA]["archivos"][ARCHIVO_VAL]["sha256"] = "f" * 64
    with pytest.raises(ErrorCandado, match="registrados"):
        verificar(otro, carpeta, raiz)


@pytest.mark.parametrize("cambio", ["extra", "oculto", "subcarpeta", "falta", "test", "hard"])
def test_candado_en_carpeta_rechaza(tmp_path, cambio):
    raiz, manifest = _repo(tmp_path)
    datos, _ = preparar(manifest, raiz)
    carpeta = tmp_path / "kaggle-datasets" / "clasificador"
    escribir_carpeta(datos, carpeta, raiz, permitidos=PERMITIDOS)
    if cambio == "extra":
        (carpeta / "notas.txt").write_bytes(b"x\n")
    elif cambio == "oculto":
        (carpeta / ".oculto").write_bytes(b"x\n")
    elif cambio == "subcarpeta":
        (carpeta / "sub").mkdir()
    elif cambio == "falta":
        (carpeta / ARCHIVO_VAL).unlink()
    else:
        (carpeta / ARCHIVO_VAL).write_bytes((raiz / "data" / "splits" / f"{cambio}.jsonl").read_bytes())
    with pytest.raises(ErrorCandado):
        candado_en_carpeta(carpeta, manifest, raiz)


def test_carpeta_dentro_del_repo_y_no_sobrescribe(tmp_path):
    raiz, manifest = _repo(tmp_path)
    datos, _ = preparar(manifest, raiz)
    with pytest.raises(ErrorCandado):
        escribir_carpeta(datos, raiz / "data" / "kaggle", raiz, permitidos=PERMITIDOS)
    with pytest.raises(ErrorCandado):
        candado_en_carpeta(raiz, manifest, raiz)
    verificar_fuera_del_repo(CARPETA)  # la carpeta real del proyecto queda fuera del repo
    assert CARPETA.name == "denue-scian-clasificador"
    carpeta = tmp_path / "kaggle-datasets" / "clasificador"
    escribir_carpeta(datos, carpeta, raiz, permitidos=PERMITIDOS)
    with pytest.raises(ErrorMuestra):
        escribir_carpeta(datos, carpeta, raiz, permitidos=PERMITIDOS)
    assert (carpeta / ARCHIVO_VAL).read_bytes() == datos[ARCHIVO_VAL]


# --- Manifest ---


def test_entrada_kaggle_sin_nombres(tmp_path):
    raiz, manifest = _repo(tmp_path)
    datos, regs = preparar(manifest, raiz)
    e = entrada_kaggle(manifest=manifest, datos=datos, regs=regs, commit="abc", fecha="2026-01-01")
    assert e["id_kaggle"] == ID_KAGGLE and e["privado"] is True and e["desechable"] is False
    assert e["campos"] == list(CAMPOS)
    for archivo, entrada in ((ARCHIVO_TRAIN, "clasificador_train_final"), (ARCHIVO_VAL, "clasificador_validacion")):
        a = e["archivos"][archivo]
        assert a["sha256"] == _sha(datos[archivo]) and a["tamano_bytes"] == len(datos[archivo])
        assert a["origen_sha256"] == manifest["splits"][entrada]["sha256"]
        assert a["filas"] == len(regs[archivo])
    assert e["metadata"]["sha256"] == _sha(datos[ARCHIVO_METADATA])
    prohibidos = {r[c] for rs in regs.values() for r in rs for c in ("id", "nom_estab", "llave")}
    assert not {t for _, t in textos(e)} & prohibidos
    nuevo = agregar_entrada(manifest, e, ENTRADA)
    assert nuevo[SECCION]["denue_scian_humo"] == manifest[SECCION]["denue_scian_humo"]
    assert {k: v for k, v in nuevo.items() if k != SECCION} == {k: v for k, v in manifest.items() if k != SECCION}
    with pytest.raises(ErrorMuestra):
        agregar_entrada(nuevo, e, ENTRADA)


# --- Test y hard nunca abiertos ---


def test_el_flujo_completo_nunca_abre_test_ni_hard(tmp_path, monkeypatch):
    raiz, manifest = _repo(tmp_path)
    carpeta = tmp_path / "kaggle-datasets" / "clasificador"
    abiertos: list[str] = []
    original = io.open

    def espia(archivo, *args, **kwargs):
        if not isinstance(archivo, int):
            abiertos.append(Path(os.fspath(archivo)).name)
        return original(archivo, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", espia)
    monkeypatch.setattr(io, "open", espia)
    datos, regs = preparar(manifest, raiz)
    candado_en_memoria(datos, manifest)
    escribir_carpeta(datos, carpeta, raiz, permitidos=PERMITIDOS)
    e = entrada_kaggle(manifest=manifest, datos=datos, regs=regs, commit="abc", fecha="2026-01-01")
    verificar(agregar_entrada(manifest, e, ENTRADA), carpeta, raiz)
    monkeypatch.undo()
    assert {"train_final.jsonl", "val.jsonl", ARCHIVO_METADATA} <= set(abiertos)  # el espía sí ve las aperturas
    assert "test.jsonl" not in abiertos and "hard.jsonl" not in abiertos
