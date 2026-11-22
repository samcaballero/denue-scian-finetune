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


# --- Generador generalizado (paso 9): todos los kernels ---

# SHA-256 de notebooks/kaggle-humo/humo_6b.py tal como se lanzó en Kaggle (commit 1b4e41f).
SHA256_KERNEL_HUMO_LANZADO = "582e004ce536925e4ad314a310f6ed6732c9a8098f6b26355f816cb6c1eaedaa"
ESPEC_06B = ck.KERNELS["06b-inicial"]


@pytest.fixture(scope="module")
def generados_por_kernel(manifest):
    return {clave: ck.generar(manifest, espec=e) for clave, e in ck.KERNELS.items()}


@pytest.fixture(scope="module")
def espacio_06b(generados_por_kernel):
    ns = {"__name__": "kernel_en_prueba"}
    texto = generados_por_kernel["06b-inicial"][ESPEC_06B.archivo_kernel].decode("utf-8")
    exec(compile(texto, ESPEC_06B.archivo_kernel, "exec"), ns)
    return ns


def test_registro_de_kernels():
    assert list(ck.KERNELS) == ["humo-6b", "06b-inicial"]
    assert ck.KERNELS["humo-6b"] is ck.HUMO and ck.HUMO.corrida is None
    assert (ck.CARPETA, ck.ARCHIVO_KERNEL, ck.ID_KERNEL, ck.FUENTES) == (
        ck.HUMO.carpeta, ck.HUMO.archivo_kernel, ck.HUMO.id_kernel, ck.HUMO.fuentes
    )


def test_kernel_de_la_prueba_igual_al_lanzado(generados_por_kernel):
    """Generalizar el generador no cambia ni un byte del kernel de la prueba, que ya corrió."""
    datos = generados_por_kernel["humo-6b"][ck.HUMO.archivo_kernel]
    assert hashlib.sha256(datos).hexdigest() == SHA256_KERNEL_HUMO_LANZADO


@pytest.mark.parametrize("clave", list(ck.KERNELS))
def test_cada_kernel_escrito_al_dia(generados_por_kernel, clave):
    assert ck.diferencias(generados_por_kernel[clave], ck.KERNELS[clave].carpeta) == []


@pytest.mark.parametrize("clave", list(ck.KERNELS))
def test_cada_kernel_extrae_sus_modulos(generados_por_kernel, clave, tmp_path):
    espec = ck.KERNELS[clave]
    ns = {"__name__": "kernel_en_prueba"}
    exec(compile(generados_por_kernel[clave][espec.archivo_kernel].decode("utf-8"), espec.archivo_kernel, "exec"), ns)
    assert list(ns["FUENTES"]) == list(espec.fuentes)
    ns["extraer"](tmp_path)
    for f in espec.fuentes:
        assert (tmp_path / f).read_bytes() == ck.normalizar_saltos((REPO_RAIZ / f).read_bytes())
    assert "main" not in ns and callable(ns["arrancar"])


def test_kernel_06b_lleva_la_corrida_y_sus_datos(espacio_06b, manifest, generados_por_kernel):
    assert espacio_06b["CORRIDA"] == dict(ESPEC_06B.corrida)
    archivos = manifest["kaggle_datasets"]["denue_scian_clasificador"]["archivos"]
    assert espacio_06b["ESPERADOS"] == {a: archivos[a]["sha256"] for a in ("train_final.jsonl", "val.jsonl")}
    texto = generados_por_kernel["06b-inicial"][ESPEC_06B.archivo_kernel].decode("utf-8")
    assert "from src.clasificador_kaggle import main" in texto
    assert "main(ESPERADOS, T_ARRANQUE, SHA256_FUENTES, CORRIDA)" in texto


def test_candado_del_kernel_06b(generados_por_kernel, manifest):
    kernel = generados_por_kernel["06b-inicial"][ESPEC_06B.archivo_kernel].decode("utf-8")
    ck.candado(kernel, manifest, ESPEC_06B)
    for texto in ("test.jsonl", "hard.jsonl"):
        assert texto not in kernel
        with pytest.raises(ck.ErrorKernel):
            ck.candado(kernel + f"\n# {texto}\n", manifest, ESPEC_06B)
    splits = manifest["splits"]
    humo = manifest["kaggle_datasets"]["denue_scian_humo"]["archivos"]
    ajenos = [
        splits["clasificador_test"]["sha256"],
        splits["clasificador_hard"]["sha256"],
        splits["clasificador_validacion"]["sha256"],
        splits["clasificador_train_final"]["sha256"],
        humo["humo_train.jsonl"]["sha256"],
    ]
    for h in ajenos:
        assert h not in kernel
        with pytest.raises(ck.ErrorKernel):
            ck.candado(kernel + f"\n# {h}\n", manifest, ESPEC_06B)
    with pytest.raises(ck.ErrorKernel):  # el kernel de la prueba no lleva los hashes del clasificador
        ck.candado(generados_por_kernel["humo-6b"][ck.HUMO.archivo_kernel].decode("utf-8"), manifest, ESPEC_06B)


def test_metadata_del_kernel_06b(generados_por_kernel, manifest):
    datos = generados_por_kernel["06b-inicial"][ck.ARCHIVO_METADATA]
    datos.decode("ascii")
    meta = json.loads(datos)
    assert meta["id"] == "samcaballero/denue-scian-06b-inicial"
    assert ck.slug(meta["title"]) == meta["id"].split("/")[1]
    assert meta["code_file"] == "clasificador_06b_inicial.py"
    assert meta["is_private"] is True and meta["enable_gpu"] is True and meta["enable_internet"] is True
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["dataset_sources"] == ["samcaballero/denue-scian-clasificador"]
    assert meta["dataset_sources"] == [manifest["kaggle_datasets"]["denue_scian_clasificador"]["id_kaggle"]]
    assert meta["kernel_sources"] == meta["competition_sources"] == meta["model_sources"] == []
    assert meta["docker_image_pinning_type"] == "original"


@pytest.mark.parametrize(
    "clave,fuente",
    [(c, f) for c, e in ck.KERNELS.items() for f in e.fuentes if f != "src/__init__.py"],
)
def test_modulos_de_cada_kernel_solo_usan_la_biblioteca_estandar(clave, fuente):
    incrustados = {f.removesuffix(".py").replace("/", ".") for f in ck.KERNELS[clave].fuentes}
    for m in _imports_de_nivel_superior(REPO_RAIZ / fuente):
        if m.startswith("src."):
            assert m in incrustados, m
        else:
            assert m.split(".")[0] in sys.stdlib_module_names | {"__future__"}, m


def test_corrida_invalida_no_genera(manifest):
    mala = dict(ESPEC_06B.corrida, semilla=7)
    espec = ck.EspecKernel(**{**ESPEC_06B.__dict__, "corrida": mala})
    with pytest.raises(ck.ErrorKernel):
        ck.generar(manifest, espec=espec)


def test_cli_verifica_todos_los_kernels(capsys):
    ck.main(["--verificar"])
    salida = capsys.readouterr().out
    for espec in ck.KERNELS.values():
        assert f"{espec.carpeta_relativa}: los archivos escritos son exactamente lo generado." in salida
    ck.main(["--verificar", "--kernel", "06b-inicial"])
    assert "humo-6b" not in capsys.readouterr().out
