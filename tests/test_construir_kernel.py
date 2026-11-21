"""Pruebas del generador del kernel de Kaggle (Fase 2, paso 6b).

Leen `data/manifest.json` (versionado, sin nombres) y los módulos de `src/`; nunca leen el
DENUE ni los splits.
"""

import ast
import hashlib
import json
import sys
from pathlib import Path

import pytest

from src import construir_kernel as ck
from src.denue import REPO_RAIZ
from src.scian import leer_manifest


@pytest.fixture(scope="module")
def manifest():
    return leer_manifest()


@pytest.fixture(scope="module")
def generados(manifest):
    return ck.generar(manifest)


@pytest.fixture(scope="module")
def espacio(generados):
    """Variables globales del kernel ejecutado como módulo (sin correr arrancar())."""
    ns = {"__name__": "kernel_en_prueba"}
    exec(compile(generados[ck.ARCHIVO_KERNEL].decode("utf-8"), ck.ARCHIVO_KERNEL, "exec"), ns)
    return ns


def test_kernel_escrito_al_dia(generados):
    """Lo que está en notebooks/kaggle-humo/ es exactamente lo que se generaría hoy."""
    assert ck.diferencias(generados, ck.CARPETA) == []


def test_fuentes_incrustadas_y_sus_hashes(espacio):
    textos = ck.leer_fuentes()
    assert list(espacio["FUENTES"]) == list(ck.FUENTES)
    assert espacio["FUENTES"] == textos
    assert espacio["SHA256_FUENTES"] == {f: hashlib.sha256(t.encode("utf-8")).hexdigest() for f, t in textos.items()}


def test_hashes_esperados_salen_del_manifest(espacio, manifest):
    archivos = manifest["kaggle_datasets"]["denue_scian_humo"]["archivos"]
    assert espacio["ESPERADOS"] == {a: archivos[a]["sha256"] for a in ("humo_train.jsonl", "humo_val.jsonl")}


def test_extraer_reproduce_los_modulos(espacio, tmp_path):
    espacio["extraer"](tmp_path)
    for f in ck.FUENTES:
        assert (tmp_path / f).read_bytes() == ck.normalizar_saltos((REPO_RAIZ / f).read_bytes())


def test_extraer_detecta_un_modulo_danado(espacio, tmp_path):
    ns = dict(espacio)
    ns["FUENTES"] = dict(espacio["FUENTES"], **{"src/__init__.py": "# cambiado\n"})
    extraer = type(espacio["extraer"])(espacio["extraer"].__code__, ns)
    with pytest.raises(SystemExit):
        extraer(tmp_path)


def test_ejecutar_como_modulo_no_arranca(espacio):
    assert "main" not in espacio and callable(espacio["arrancar"])


def test_candado(generados, manifest):
    kernel = generados[ck.ARCHIVO_KERNEL].decode("utf-8")
    assert "test.jsonl" not in kernel
    assert manifest["splits"]["clasificador_test"]["sha256"] not in kernel
    with pytest.raises(ck.ErrorKernel):
        ck.candado(kernel + "\n# test.jsonl\n", manifest)
    with pytest.raises(ck.ErrorKernel):
        ck.candado(kernel + f"\n# {manifest['splits']['clasificador_test']['sha256']}\n", manifest)
    with pytest.raises(ck.ErrorKernel):
        ck.candado(kernel + f"\n# {manifest['splits']['clasificador_validacion']['sha256']}\n", manifest)


def test_metadata_del_kernel(generados, manifest):
    datos = generados[ck.ARCHIVO_METADATA]
    datos.decode("ascii")
    meta = json.loads(datos)
    assert meta["id"] == "samcaballero/denue-scian-humo-6b"
    assert ck.slug(meta["title"]) == meta["id"].split("/")[1]
    assert meta["code_file"] == ck.ARCHIVO_KERNEL
    assert (meta["language"], meta["kernel_type"]) == ("python", "script")
    assert meta["is_private"] is True
    assert meta["enable_gpu"] is True and meta["enable_tpu"] is False
    assert meta["enable_internet"] is True
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["dataset_sources"] == [manifest["kaggle_datasets"]["denue_scian_humo"]["id_kaggle"]]
    assert meta["docker_image_pinning_type"] == "original"


def _imports_de_nivel_superior(ruta: Path) -> set[str]:
    arbol = ast.parse(ruta.read_text(encoding="utf-8"))
    res = set()
    for nodo in arbol.body:
        if isinstance(nodo, ast.Import):
            res |= {a.name for a in nodo.names}
        elif isinstance(nodo, ast.ImportFrom):
            res.add(nodo.module)
    return res


@pytest.mark.parametrize("fuente", [f for f in ck.FUENTES if f != "src/__init__.py"])
def test_modulos_incrustados_solo_usan_la_biblioteca_estandar(fuente):
    """Arriba solo importan la biblioteca estándar y otros módulos incrustados."""
    incrustados = {f.removesuffix(".py").replace("/", ".") for f in ck.FUENTES}
    for m in _imports_de_nivel_superior(REPO_RAIZ / fuente):
        if m.startswith("src."):
            assert m in incrustados, m
        else:
            assert m.split(".")[0] in sys.stdlib_module_names | {"__future__"}, m


def test_leer_fuente_normaliza_y_rechaza(tmp_path):
    p = tmp_path / "m.py"
    p.write_bytes(b"a = 1\r\nb = 'n'\r\n")
    assert ck.leer_fuente(p) == "a = 1\nb = 'n'\n"
    p.write_bytes(b"a = 1\rb = 2\n")
    with pytest.raises(ck.ErrorKernel):
        ck.leer_fuente(p)
    p.write_bytes(b"\xef\xbb\xbfa = 1\n")
    with pytest.raises(ck.ErrorKernel):
        ck.leer_fuente(p)


@pytest.mark.parametrize(
    "texto",
    ["", "x", "una\n", "dos\nlineas", "comillas ' \" ''' \"\"\"\n", "barra \\n \\\\ \t tab\n", "ñ é   sep\n\n\n"],
)
def test_literal_ida_y_vuelta(texto):
    assert eval(ck._literal(texto, "    ")) == texto


def test_generar_es_determinista(manifest):
    assert ck.generar(manifest) == ck.generar(manifest)
