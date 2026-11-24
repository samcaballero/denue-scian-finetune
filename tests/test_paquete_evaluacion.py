"""Pruebas del paquete de evaluación para Kaggle (Fase 4, paso 14, Tema B).

Solo usan nombres, llaves, ids y códigos inventados en carpetas temporales; nunca leen el
DENUE ni los splits reales.
"""

import hashlib
import json
from pathlib import Path

import pytest

from src.exploracion_clasificador import textos
from src.muestra_humo import SECCION, ErrorCandado, ErrorMuestra, agregar_entrada, escribir_carpeta
from src.normalizacion import normalizar_nombre
from src.paquete_evaluacion import (
    ARCHIVO_HARD,
    ARCHIVO_METADATA,
    ARCHIVO_TEST,
    CAMPOS,
    CARPETA,
    ENTRADA,
    ID_KAGGLE,
    PERMITIDOS,
    TITULO,
    ErrorPaquete,
    candado_contenido,
    candado_en_carpeta,
    candado_en_memoria,
    entrada_kaggle,
    preparar,
    serializar,
    verificar,
)
from src.muestra_humo import verificar_fuera_del_repo

CAMPOS_SPLIT = ["id", "nom_estab", "codigo_act", "llave", "sin_nombre", "generico", "igual_titulo"]


def _sha(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def _reg(i: int, nombre: str, codigo: str, **extra) -> dict:
    return {"id": str(i), "nom_estab": nombre, "codigo_act": codigo, "llave": normalizar_nombre(nombre),
            "sin_nombre": False, "generico": False, "igual_titulo": False, **extra}


def _jsonl(regs) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in regs).encode("utf-8")


def _repo(tmp_path: Path, test=None, hard=None):
    """Repo de juguete con test y hard inventados (hard con tipo) y su manifest."""
    raiz = tmp_path / "repo"
    splits = raiz / "data" / "splits"
    splits.mkdir(parents=True)
    test = test if test is not None else [_reg(5000 + 3 * i, f"Prueba inventada {i}", "811111") for i in range(7)]
    hard = hard if hard is not None else [
        _reg(6000 + i, f"Difícil inventado {i}", "722511", tipo=("ambiguo" if i % 2 else "indirecto")) for i in range(4)
    ]
    manifest = {"version": 1, "splits": {}, "kaggle_datasets": {"denue_scian_clasificador": {"archivos": {
        "val.jsonl": {"sha256": "e" * 64}}}}}
    for nombre, entrada, regs, campos in (
        ("test.jsonl", "clasificador_test", test, CAMPOS_SPLIT),
        ("hard.jsonl", "clasificador_hard", hard, [*CAMPOS_SPLIT, "tipo"]),
    ):
        datos = _jsonl(regs)
        (splits / nombre).write_bytes(datos)
        manifest["splits"][entrada] = {"ruta": f"data/splits/{nombre}", "sha256": _sha(datos), "filas": len(regs),
                                       "campos": campos}
    return raiz, manifest


def _lineas(datos: bytes) -> list[dict]:
    return [json.loads(l) for l in datos.decode("utf-8").split("\n")[:-1]]


# --- Contenido ---


def test_preparar_solo_id_y_nombre_en_el_orden_del_origen(tmp_path):
    raiz, manifest = _repo(tmp_path)
    datos, filas, llaves = preparar(manifest, raiz)
    assert set(datos) == PERMITIDOS
    assert filas == {ARCHIVO_TEST: 7, ARCHIVO_HARD: 4}
    for archivo, origen in ((ARCHIVO_TEST, "test.jsonl"), (ARCHIVO_HARD, "hard.jsonl")):
        completos = _lineas((raiz / "data" / "splits" / origen).read_bytes())
        paquete = _lineas(datos[archivo])
        assert [r["id"] for r in paquete] == [r["id"] for r in completos]
        assert all(tuple(r) == CAMPOS == ("id", "nom_estab") for r in paquete)
        assert paquete == [{"id": r["id"], "nom_estab": r["nom_estab"]} for r in completos]
        for prohibido in (b"codigo_act", b"tipo", b"llave", b"811111", b"722511", b"ambiguo", b"sin_nombre"):
            assert prohibido not in datos[archivo]
        assert not datos[archivo].startswith(b"\xef\xbb\xbf") and b"\r" not in datos[archivo]
    assert "PRUEBA INVENTADA 3" in llaves  # solo para la revisión de privacidad
    assert preparar(manifest, raiz)[0] == datos  # mismos bytes desde cero
    meta = json.loads(datos[ARCHIVO_METADATA].decode("ascii"))
    assert meta["id"] == ID_KAGGLE and meta["title"] == TITULO and "INEGI" in meta["description"]


def test_preparar_rechaza_hash_filas_ids_compartidos_y_campos(tmp_path):
    raiz, manifest = _repo(tmp_path)
    ruta = raiz / "data" / "splits" / "hard.jsonl"
    ruta.write_bytes(ruta.read_bytes().replace(b"722511", b"722512", 1))
    with pytest.raises(ErrorMuestra, match="SHA-256"):
        preparar(manifest, raiz)
    raiz, manifest = _repo(tmp_path / "filas")
    manifest["splits"]["clasificador_test"]["filas"] = 8
    with pytest.raises(ErrorPaquete, match="filas"):
        preparar(manifest, raiz)
    raiz, manifest = _repo(tmp_path / "id", hard=[_reg(5000, "Difícil inventado", "722511", tipo="ambiguo")])
    with pytest.raises(ErrorPaquete, match="comparten id"):
        preparar(manifest, raiz)
    raiz, manifest = _repo(tmp_path / "campos")
    manifest["splits"]["clasificador_hard"]["campos"] = CAMPOS_SPLIT  # sin tipo: no es el formato de hard
    with pytest.raises(ErrorPaquete, match="claves"):
        preparar(manifest, raiz)


def test_serializar_solo_acepta_id_y_nombre():
    assert serializar([{"id": "1", "nom_estab": "Café Ñandú"}]) == '{"id": "1", "nom_estab": "Café Ñandú"}\n'.encode()
    for malo in ({"id": "1", "nom_estab": "A", "codigo_act": "811111"}, {"nom_estab": "A", "id": "1"}, {"id": "1"}):
        with pytest.raises(ErrorPaquete):
            serializar([malo])


# --- Candado ---


def test_candado_de_contenido_rechaza_etiquetas_y_otras_claves(tmp_path):
    raiz, manifest = _repo(tmp_path)
    datos, _, _ = preparar(manifest, raiz)
    assert candado_contenido(datos, manifest) == {ARCHIVO_TEST: 7, ARCHIVO_HARD: 4}
    lineas = _lineas(datos[ARCHIVO_TEST])
    casos = {
        "con codigo_act": [{**lineas[0], "codigo_act": "811111"}, *lineas[1:]],
        "con tipo": [{**lineas[0], "tipo": "ambiguo"}, *lineas[1:]],
        "con llave": [{**lineas[0], "llave": "PRUEBA"}, *lineas[1:]],
        "sin nombre": [{"id": lineas[0]["id"]}, *lineas[1:]],
        "claves en otro orden": [{"nom_estab": lineas[0]["nom_estab"], "id": lineas[0]["id"]}, *lineas[1:]],
    }
    for motivo, regs in casos.items():
        malos = {**datos, ARCHIVO_TEST: _jsonl(regs)}
        with pytest.raises(ErrorCandado, match="id y nom_estab"):
            candado_contenido(malos, manifest)
    with pytest.raises(ErrorCandado, match="filas"):
        candado_contenido({**datos, ARCHIVO_HARD: _jsonl(_lineas(datos[ARCHIVO_HARD])[:-1])}, manifest)
    with pytest.raises(ErrorCandado):
        candado_contenido({**datos, ARCHIVO_TEST: datos[ARCHIVO_TEST].replace(b"\n", b"\r\n")}, manifest)


def test_candado_en_memoria_rechaza_archivos_y_hashes(tmp_path):
    raiz, manifest = _repo(tmp_path)
    datos, _, _ = preparar(manifest, raiz)
    hashes = candado_en_memoria(datos, manifest)
    assert set(hashes) == PERMITIDOS
    test_completo = (raiz / "data" / "splits" / "test.jsonl").read_bytes()
    hard_completo = (raiz / "data" / "splits" / "hard.jsonl").read_bytes()
    with pytest.raises(ErrorCandado, match="no permitidos"):
        candado_en_memoria({**datos, "extra.jsonl": b"{}\n"}, manifest)
    with pytest.raises(ErrorCandado, match="faltan"):
        candado_en_memoria({k: v for k, v in datos.items() if k != ARCHIVO_HARD}, manifest)
    with pytest.raises(ErrorCandado, match="test"):
        candado_en_memoria({**datos, ARCHIVO_TEST: test_completo}, manifest)
    with pytest.raises(ErrorCandado, match="hard set"):
        candado_en_memoria({**datos, ARCHIVO_HARD: hard_completo}, manifest)
    otro = dict(manifest, kaggle_datasets={"denue_scian_clasificador": {"archivos": {
        "val.jsonl": {"sha256": _sha(datos[ARCHIVO_HARD])}}}})
    with pytest.raises(ErrorCandado, match="otra entrada"):
        candado_en_memoria(datos, otro)


def test_candado_en_carpeta_fuera_del_repo_sin_ocultos_ni_subcarpetas(tmp_path):
    raiz, manifest = _repo(tmp_path)
    datos, _, _ = preparar(manifest, raiz)
    carpeta = tmp_path / "kaggle-datasets" / "denue-scian-evaluacion"
    escribir_carpeta(datos, carpeta, raiz, permitidos=PERMITIDOS)
    assert candado_en_carpeta(carpeta, manifest, raiz) == candado_en_memoria(datos, manifest)
    with pytest.raises(ErrorMuestra, match="ya tiene archivos"):
        escribir_carpeta(datos, carpeta, raiz, permitidos=PERMITIDOS)
    dentro = raiz / "kaggle-datasets" / "denue-scian-evaluacion"
    with pytest.raises(ErrorCandado, match="dentro del repo"):
        escribir_carpeta(datos, dentro, raiz, permitidos=PERMITIDOS)
    (carpeta / ".oculto").write_bytes(b"x")
    with pytest.raises(ErrorCandado, match="no permitidos"):
        candado_en_carpeta(carpeta, manifest, raiz)
    (carpeta / ".oculto").unlink()
    (carpeta / "sub").mkdir()
    with pytest.raises(ErrorCandado, match="subcarpetas"):
        candado_en_carpeta(carpeta, manifest, raiz)
    (carpeta / "sub").rmdir()
    (carpeta / ARCHIVO_TEST).write_bytes(_jsonl([{**r, "codigo_act": "811111"} for r in _lineas(datos[ARCHIVO_TEST])]))
    with pytest.raises(ErrorCandado, match="id y nom_estab"):
        candado_en_carpeta(carpeta, manifest, raiz)


def test_carpeta_real_fuera_del_repo():
    verificar_fuera_del_repo(CARPETA)
    assert CARPETA.name == "denue-scian-evaluacion"


# --- Manifest ---


def test_entrada_kaggle_sin_clases_ni_nombres_y_verificar(tmp_path):
    raiz, manifest = _repo(tmp_path)
    datos, filas, _ = preparar(manifest, raiz)
    entrada = entrada_kaggle(manifest=manifest, datos=datos, filas=filas, commit="a" * 40, fecha="2026-11-24")
    assert entrada["privado"] is True and entrada["desechable"] is False and entrada["campos"] == ["id", "nom_estab"]
    for archivo in (ARCHIVO_TEST, ARCHIVO_HARD):
        a = entrada["archivos"][archivo]
        assert set(a) == {"ruta", "origen", "origen_sha256", "filas", "tamano_bytes", "sha256"}  # sin clases
        assert a["tamano_bytes"] == len(datos[archivo]) and a["sha256"] == _sha(datos[archivo])
    assert entrada["metadata"]["sha256"] == _sha(datos[ARCHIVO_METADATA])
    todo = " ".join(t for _, t in textos(entrada))
    for nombre in ("Prueba inventada", "Difícil", "PRUEBA", "5000", "811111", "ambiguo"):
        assert nombre not in todo
    # verificar(): con la entrada registrada, los hashes de la carpeta deben ser los del manifest.
    carpeta = tmp_path / "kaggle-datasets" / "denue-scian-evaluacion"
    escribir_carpeta(datos, carpeta, raiz, permitidos=PERMITIDOS)
    registrado = agregar_entrada(manifest, entrada, ENTRADA)
    hashes, registrada = verificar(registrado, carpeta, raiz)
    assert registrada and hashes[ARCHIVO_TEST] == entrada["archivos"][ARCHIVO_TEST]["sha256"]
    otra = json.loads(json.dumps(registrado))
    otra[SECCION][ENTRADA]["archivos"][ARCHIVO_HARD]["sha256"] = "f" * 64
    with pytest.raises(ErrorCandado, match="registrados"):
        verificar(otra, carpeta, raiz)
