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
    assert list(ck.KERNELS) == ["humo-6b", "06b-inicial", "06b-barrido"]
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


# --- Barrido del 0.6B (paso 11) ---

# SHA-256 de lo que se lanzó en Kaggle (kernel y metadata), de sus commits: 1b4e41f y 54374f9.
LANZADOS = {
    "humo-6b": {
        "humo_6b.py": "582e004ce536925e4ad314a310f6ed6732c9a8098f6b26355f816cb6c1eaedaa",
        "kernel-metadata.json": "c1b5e3aae412903b16d9b87cb4deedfe22a5958b57ece7a860265192087809ac",
    },
    "06b-inicial": {
        "clasificador_06b_inicial.py": "570a3aa8e7c0f1ef180d4adfecedebfcc432bd616f47f31e3d240b34c7dd2975",
        "kernel-metadata.json": "55bbfde5cff87cb5d853c8331fd3cd2019ebcba01942baae3333d707671373be",
    },
}
ESPEC_BARRIDO = ck.KERNELS["06b-barrido"]
PERMITIDAS = {"nombre", "descripcion", "learning_rate", "lora.r", "lora.lora_alpha"}


def _diferencias(a, b, prefijo=""):
    """Claves (con punto para lo anidado) en que difieren dos corridas."""
    res = set()
    for k in set(a) | set(b):
        ruta = f"{prefijo}{k}"
        if isinstance(a.get(k), dict) and isinstance(b.get(k), dict):
            res |= _diferencias(a[k], b[k], ruta + ".")
        elif a.get(k) != b.get(k) or type(a.get(k)) is not type(b.get(k)):
            res.add(ruta)
    return res


@pytest.mark.parametrize("clave", list(LANZADOS))
def test_kernels_lanzados_no_cambian_ni_un_byte(generados_por_kernel, clave):
    for archivo, sha in LANZADOS[clave].items():
        assert hashlib.sha256(generados_por_kernel[clave][archivo]).hexdigest() == sha, archivo


@pytest.mark.parametrize("i", range(4))
def test_cada_corrida_del_barrido_difiere_de_la_inicial_solo_en_tasa_rank_alpha_y_nombres(i):
    inicial = dict(ESPEC_06B.corrida)
    c = dict(ESPEC_BARRIDO.corridas[i])
    difieren = _diferencias(inicial, c)
    assert difieren <= PERMITIDAS, difieren - PERMITIDAS
    assert c["lora"]["lora_alpha"] == c["lora"]["r"]
    assert c["lora"]["target_modules"] == inicial["lora"]["target_modules"]
    for clave in ("modelo", "max_seq_length", "lote", "gradient_accumulation_steps", "epocas", "optim",
                  "lr_scheduler_type", "fraccion_warmup", "eval_cada_pasos", "lote_eval", "semilla"):
        assert c[clave] == inicial[clave], clave


def test_el_barrido_es_la_rejilla_completa():
    corridas = ESPEC_BARRIDO.corridas
    assert ESPEC_BARRIDO.corrida is None and len(corridas) == 4
    assert {(c["learning_rate"], c["lora"]["r"]) for c in corridas} == {(1e-4, 8), (1e-4, 32), (2e-4, 8), (2e-4, 32)}
    assert [c["nombre"] for c in corridas] == ["06b-lr1e-4-r8", "06b-lr1e-4-r32", "06b-lr2e-4-r8", "06b-lr2e-4-r32"]
    assert all(c["learning_rate"] in (1e-4, 2e-4) and type(c["learning_rate"]) is float for c in corridas)
    assert (ESPEC_06B.corrida["learning_rate"], ESPEC_06B.corrida["lora"]["r"]) not in {
        (c["learning_rate"], c["lora"]["r"]) for c in corridas
    }


def test_kernel_del_barrido_lleva_las_corridas_y_sus_datos(generados_por_kernel, manifest):
    texto = generados_por_kernel["06b-barrido"][ESPEC_BARRIDO.archivo_kernel].decode("utf-8")
    ns = {"__name__": "kernel_en_prueba"}
    exec(compile(texto, ESPEC_BARRIDO.archivo_kernel, "exec"), ns)
    assert ns["CORRIDAS"] == [dict(c) for c in ESPEC_BARRIDO.corridas]
    assert "CORRIDA" not in ns
    archivos = manifest["kaggle_datasets"]["denue_scian_clasificador"]["archivos"]
    assert ns["ESPERADOS"] == {a: archivos[a]["sha256"] for a in ("train_final.jsonl", "val.jsonl")}
    assert "from src.barrido_kaggle import main" in texto
    assert "main(ESPERADOS, T_ARRANQUE, SHA256_FUENTES, CORRIDAS)" in texto
    assert ESPEC_BARRIDO.fuentes[-1] == "src/barrido_kaggle.py"
    assert set(ESPEC_06B.fuentes) < set(ESPEC_BARRIDO.fuentes)


def test_candado_del_kernel_del_barrido(generados_por_kernel, manifest):
    kernel = generados_por_kernel["06b-barrido"][ESPEC_BARRIDO.archivo_kernel].decode("utf-8")
    ck.candado(kernel, manifest, ESPEC_BARRIDO)
    for texto in ("test.jsonl", "hard.jsonl"):
        assert texto not in kernel
    splits = manifest["splits"]
    for e in ("clasificador_test", "clasificador_hard", "clasificador_validacion", "clasificador_train_final"):
        with pytest.raises(ck.ErrorKernel):
            ck.candado(kernel + f"\n# {splits[e]['sha256']}\n", manifest, ESPEC_BARRIDO)


def test_metadata_del_barrido_fija_la_imagen_de_06b_inicial(generados_por_kernel, manifest):
    meta = json.loads(generados_por_kernel["06b-barrido"][ck.ARCHIVO_METADATA].decode("ascii"))
    assert meta["id"] == "samcaballero/denue-scian-06b-barrido" and ck.slug(meta["title"]) == "denue-scian-06b-barrido"
    assert meta["code_file"] == "clasificador_06b_barrido.py"
    assert meta["is_private"] is True and meta["enable_gpu"] is True and meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["dataset_sources"] == [manifest["kaggle_datasets"]["denue_scian_clasificador"]["id_kaggle"]]
    assert meta["docker_image_pinning_type"] == "original"
    assert meta["docker_image"] == ck.DOCKER_06B_INICIAL
    assert ck.DOCKER_06B_INICIAL.split("@sha256:")[1].isalnum() and len(ck.DOCKER_06B_INICIAL.split("@sha256:")[1]) == 64
    for clave in ("humo-6b", "06b-inicial"):  # los kernels ya lanzados no llevan docker_image
        assert "docker_image" not in json.loads(generados_por_kernel[clave][ck.ARCHIVO_METADATA])


def test_barrido_con_nombres_repetidos_o_con_corrida_y_corridas_no_genera(manifest):
    repetidas = (ESPEC_BARRIDO.corridas[0], ESPEC_BARRIDO.corridas[0])
    with pytest.raises(ck.ErrorKernel, match="nombres distintos"):
        ck.generar(manifest, espec=ck.EspecKernel(**{**ESPEC_BARRIDO.__dict__, "corridas": repetidas}))
    with pytest.raises(ck.ErrorKernel, match="no los dos"):
        ck.generar(manifest, espec=ck.EspecKernel(**{**ESPEC_BARRIDO.__dict__, "corrida": ESPEC_06B.corrida}))
    mala = dict(ESPEC_BARRIDO.corridas[1], semilla=7)
    with pytest.raises(ck.ErrorKernel):
        ck.generar(manifest, espec=ck.EspecKernel(**{**ESPEC_BARRIDO.__dict__, "corridas": (mala,)}))
